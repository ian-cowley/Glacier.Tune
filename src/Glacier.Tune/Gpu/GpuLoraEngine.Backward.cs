namespace Glacier.Tune.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Core;

public sealed unsafe partial class GpuLoraEngine
{
    public bool SupportsBackwardGemm(GgufType type) =>
        type is GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.Q4_0 or GgufType.F32;

    public void LaunchGemmBwd(
        GgufType type,
        IntPtr dDX,
        IntPtr dDY,
        IntPtr dW,
        int kCols,
        int mRows,
        int batchSize,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBwd,
            GgufType.Q6_K => _fnGemmQ6KBwd,
            GgufType.Q8_0 => _fnGemmQ8_0Bwd,
            GgufType.Q4_0 => _fnGemmQ4_0Bwd,
            GgufType.F32 => _fnGemmF32Bwd,
            _ => throw new NotSupportedException($"GPU backward GEMM does not support type {type}")
        };

        bool isSuperBlock = type is GgufType.Q4_K or GgufType.Q6_K;
        uint blockSize = isSuperBlock ? 256u : 128u;
        uint gridX = (uint)((kCols + (int)blockSize - 1) / (int)blockSize);
        uint gridY = (uint)((batchSize + 7) / 8);

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dDX;
        pArgs[1] = &dDY;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_bwd)");
    }

    public void BackwardProjection(
        GgufType type,
        IntPtr dW,
        Tensor<float> dY,
        Tensor<float> dXInput,
        int inFeatures,
        int outFeatures,
        int seqLen)
    {
        nuint bytesDX = (nuint)((long)seqLen * inFeatures * sizeof(float));
        nuint bytesDY = (nuint)((long)seqLen * outFeatures * sizeof(float));

        lock (_scratchLock)
        {
            if (bytesDX > _scratchXCap)
            {
                _gpu.FreeDevice(_dScratchX);
                _dScratchX = _gpu.AllocateDevice(bytesDX * 2);
                _scratchXCap = bytesDX * 2;
            }
            if (bytesDY > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(bytesDY * 2);
                _scratchYCap = bytesDY * 2;
            }

            _gpu.CopyToDevice(_dScratchX, (IntPtr)dXInput.DataPointer, bytesDX);
            _gpu.CopyToDevice(_dScratchY, (IntPtr)dY.DataPointer, bytesDY);

            LaunchGemmBwd(type, _dScratchX, _dScratchY, dW, inFeatures, outFeatures, seqLen);

            _gpu.CopyToHost((IntPtr)dXInput.DataPointer, _dScratchX, bytesDX);
        }
    }

    /// <summary>
    /// Computes exact tiled online streaming Cross-Entropy loss and hidden-state gradients in GPU VRAM.
    /// Keeps peak logit VRAM buffer bounded to [validCount x vocabTileSize] (&lt; 16 MB instead of 622 MB).
    /// </summary>
    public float ComputeTiledLmHeadLossAndGradients(
        GgufType lmHeadType,
        float* pValidHidden,
        float* pValidGradOut,
        int* validTargetIds,
        int hiddenDim,
        int vocabSize,
        int validCount,
        int vocabTileSize,
        delegate*<float*, int, float> vectorizedMax,
        delegate*<float*, int, float, double> vectorizedSumExp)
    {
        int actualTileSize = Math.Min(vocabTileSize, vocabSize);
        int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
        int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

        nuint hiddenBytes = (nuint)((long)validCount * hiddenDim * sizeof(float));
        nuint tileLogitBytes = (nuint)((long)validCount * actualTileSize * sizeof(float));
        float invValid = 1.0f / validCount;

        new Span<float>(pValidGradOut, validCount * hiddenDim).Clear();

        float* pTileLogits = (float*)NativeMemory.Alloc((nuint)((long)validCount * actualTileSize), sizeof(float));
        float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
        double* runningSumExp = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
        float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
        double* logSumExp = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));

        try
        {
            for (int i = 0; i < validCount; i++)
            {
                runningMax[i] = float.NegativeInfinity;
                runningSumExp[i] = 0.0;
                targetLogits[i] = 0.0f;
            }

            double totalLoss = 0.0;

            lock (_scratchLock)
            {
                EnsureFfnCapacity(hiddenBytes, tileLogitBytes);
                if (tileLogitBytes > _scratchYCap)
                {
                    _gpu.FreeDevice(_dScratchY);
                    _dScratchY = _gpu.AllocateDevice(tileLogitBytes * 2);
                    _scratchYCap = tileLogitBytes * 2;
                }

                // Upload valid hidden states and zeroed gradient accumulator ONCE
                _gpu.CopyToDevice(_dFfnNormX, (IntPtr)pValidHidden, hiddenBytes);
                _gpu.CopyToDevice(_dFfnDownOut, (IntPtr)pValidGradOut, hiddenBytes);

                // PASS 1: Tiled online Log-Sum-Exp in VRAM
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    IntPtr dTileW = _dLmHeadWeight + (nint)((long)vStart * rowBytes);
                    nuint curTileBytes = (nuint)((long)validCount * vCount * sizeof(float));

                    LaunchGemmBatch(lmHeadType, _dScratchY, _dFfnNormX, dTileW, hiddenDim, vCount, validCount);
                    _gpu.CopyToHost((IntPtr)pTileLogits, _dScratchY, curTileBytes);

                    for (int i = 0; i < validCount; i++)
                    {
                        float* logits = pTileLogits + (long)i * vCount;
                        int targetId = validTargetIds[i];
                        if (targetId >= vStart && targetId < vStart + vCount)
                        {
                            targetLogits[i] = logits[targetId - vStart];
                        }

                        float tileMax = vectorizedMax(logits, vCount);
                        if (float.IsNegativeInfinity(runningMax[i]))
                        {
                            runningMax[i] = tileMax;
                            runningSumExp[i] = vectorizedSumExp(logits, vCount, tileMax);
                        }
                        else
                        {
                            float mNew = MathF.Max(runningMax[i], tileMax);
                            double factor = Math.Exp((double)runningMax[i] - mNew);
                            double tileSum = vectorizedSumExp(logits, vCount, mNew);
                            runningSumExp[i] = runningSumExp[i] * factor + tileSum;
                            runningMax[i] = mNew;
                        }
                    }
                }

                for (int i = 0; i < validCount; i++)
                {
                    double lse = runningMax[i] + Math.Log(runningSumExp[i]);
                    logSumExp[i] = lse;
                    totalLoss += lse - targetLogits[i];
                }

                // PASS 2: Exact unbiased analytical gradient backpropagation via VRAM transpose-GEMM
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    IntPtr dTileW = _dLmHeadWeight + (nint)((long)vStart * rowBytes);
                    nuint curTileBytes = (nuint)((long)validCount * vCount * sizeof(float));

                    if (numTiles > 1)
                    {
                        LaunchGemmBatch(lmHeadType, _dScratchY, _dFfnNormX, dTileW, hiddenDim, vCount, validCount);
                        _gpu.CopyToHost((IntPtr)pTileLogits, _dScratchY, curTileBytes);
                    }

                    int vStartSnap = vStart;
                    int vCountSnap = vCount;
                    Parallel.For(0, validCount, i =>
                    {
                        int targetId = validTargetIds[i];
                        double lse = logSumExp[i];
                        float* logits = pTileLogits + (long)i * vCountSnap;
                        for (int v = 0; v < vCountSnap; v++)
                        {
                            if (vStartSnap + v == targetId)
                            {
                                logits[v] = (float)(Math.Exp(logits[v] - lse) - 1.0) * invValid;
                            }
                            else
                            {
                                logits[v] = MathF.Exp((float)(logits[v] - lse)) * invValid;
                            }
                        }
                    });

                    _gpu.CopyToDevice(_dScratchY, (IntPtr)pTileLogits, curTileBytes);
                    LaunchGemmBwd(lmHeadType, _dFfnDownOut, _dScratchY, dTileW, hiddenDim, vCount, validCount);
                }

                // Download accumulated hidden gradients ONCE
                _gpu.CopyToHost((IntPtr)pValidGradOut, _dFfnDownOut, hiddenBytes);
            }

            return (float)(totalLoss * invValid);
        }
        finally
        {
            NativeMemory.Free(pTileLogits);
            NativeMemory.Free(runningMax);
            NativeMemory.Free(runningSumExp);
            NativeMemory.Free(targetLogits);
            NativeMemory.Free(logSumExp);
        }
    }

    public void LaunchAttnTrainBwd(
        IntPtr dDOut, IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dProbs,
        IntPtr dDq, IntPtr dDk, IntPtr dDv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim, float scale)
    {
        if (headDim <= 0 || headDim > 1024)
            throw new ArgumentOutOfRangeException(nameof(headDim), $"GPU training attention requires 1 <= headDim <= 1024, got {headDim}.");

        uint blockSize = (uint)(((headDim + 31) / 32) * 32);
        uint gridX = (uint)nHeadsQ;
        uint gridY = (uint)seqLen;
        uint sharedMemBytes = (uint)(seqLen * sizeof(float));

        IntPtr dAttnDs = _dAttnDs;
        void** pArgsDq = stackalloc void*[11];
        pArgsDq[0] = &dDOut;
        pArgsDq[1] = &dK;
        pArgsDq[2] = &dV;
        pArgsDq[3] = &dProbs;
        pArgsDq[4] = &dDq;
        pArgsDq[5] = &dAttnDs;
        pArgsDq[6] = &seqLen;
        pArgsDq[7] = &nHeadsQ;
        pArgsDq[8] = &nHeadsKv;
        pArgsDq[9] = &headDim;
        pArgsDq[10] = &scale;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainBwdDqDs,
            gridX, gridY, 1,
            blockSize, 1, 1,
            sharedMemBytes, IntPtr.Zero,
            (IntPtr)pArgsDq,
            IntPtr.Zero), "LaunchKernel(tune_attn_causal_gqa_bwd_dq_ds)");

        uint gridX_kv = (uint)nHeadsKv;
        uint gridY_kv = (uint)seqLen;

        void** pArgsKv = stackalloc void*[10];
        pArgsKv[0] = &dQ;
        pArgsKv[1] = &dDOut;
        pArgsKv[2] = &dProbs;
        pArgsKv[3] = &dAttnDs;
        pArgsKv[4] = &dDk;
        pArgsKv[5] = &dDv;
        pArgsKv[6] = &seqLen;
        pArgsKv[7] = &nHeadsQ;
        pArgsKv[8] = &nHeadsKv;
        pArgsKv[9] = &headDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainBwdDkDv,
            gridX_kv, gridY_kv, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgsKv,
            IntPtr.Zero), "LaunchKernel(tune_attn_causal_gqa_bwd_dk_dv)");
    }

    public void BackwardAttention(
        int layerIndex,
        Tensor<float> dAttnOut,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float> attnProbs,
        Tensor<float> dq,
        Tensor<float> dk,
        Tensor<float> dv,
        int seqLen,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        float ropeFreqBase)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;
        nuint qBytes = (nuint)((long)seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)((long)seqLen * kvDim * sizeof(float));
        nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            _gpu.CopyToDevice(_dAttnOut, (IntPtr)dAttnOut.DataPointer, qBytes);
            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dAttnProbs, (IntPtr)attnProbs.DataPointer, probsBytes);

            LaunchAttnTrainBwd(
                _dAttnOut, _dQBuf, _dKBuf, _dVBuf, _dAttnProbs,
                _dDq, _dDk, _dDv,
                seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            LaunchRopeBatch(_dDq, _dDk, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, -1.0f);

            _gpu.CopyToHost((IntPtr)dq.DataPointer, _dDq, qBytes);
            _gpu.CopyToHost((IntPtr)dk.DataPointer, _dDk, kvBytes);
            _gpu.CopyToHost((IntPtr)dv.DataPointer, _dDv, kvBytes);
        }
    }
}
