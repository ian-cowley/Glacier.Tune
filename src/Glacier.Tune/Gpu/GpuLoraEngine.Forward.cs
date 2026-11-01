namespace Glacier.Tune.Gpu;

using System;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;

public sealed unsafe partial class GpuLoraEngine
{
    public bool SupportsForwardGemm(GgufType type) =>
        type is GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.Q4_0 or GgufType.F32;

    public void ForwardProjection(
        GgufType type,
        IntPtr dW,
        IntPtr dBias,
        Tensor<float> x,
        Tensor<float> output,
        int inFeatures,
        int outFeatures,
        int seqLen)
    {
        nuint bytesIn = (nuint)((long)seqLen * inFeatures * sizeof(float));
        nuint bytesOut = (nuint)((long)seqLen * outFeatures * sizeof(float));

        lock (_scratchLock)
        {
            if (bytesIn > _scratchXCap)
            {
                _gpu.FreeDevice(_dScratchX);
                _dScratchX = _gpu.AllocateDevice(bytesIn * 2);
                _scratchXCap = bytesIn * 2;
            }
            if (bytesOut > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(bytesOut * 2);
                _scratchYCap = bytesOut * 2;
            }

            _gpu.CopyToDevice(_dScratchX, (IntPtr)x.DataPointer, bytesIn);
            LaunchGemmBatch(type, _dScratchY, _dScratchX, dW, inFeatures, outFeatures, seqLen, dBias);
            _gpu.CopyToHost((IntPtr)output.DataPointer, _dScratchY, bytesOut);
        }
    }

    public void ComputeLmHeadLogits(
        GgufType lmHeadType,
        IntPtr dLmHeadWeight,
        float* pValidHidden,
        float* pValidLogits,
        int hiddenDim,
        int vocabSize,
        int validCount)
    {
        nuint bytesIn = (nuint)((long)validCount * hiddenDim * sizeof(float));
        nuint bytesOut = (nuint)((long)validCount * vocabSize * sizeof(float));

        lock (_scratchLock)
        {
            if (bytesIn > _scratchXCap)
            {
                _gpu.FreeDevice(_dScratchX);
                _dScratchX = _gpu.AllocateDevice(bytesIn * 2);
                _scratchXCap = bytesIn * 2;
            }
            if (bytesOut > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(bytesOut * 2);
                _scratchYCap = bytesOut * 2;
            }

            _gpu.CopyToDevice(_dScratchX, (IntPtr)pValidHidden, bytesIn);
            LaunchGemmBatch(lmHeadType, _dScratchY, _dScratchX, dLmHeadWeight, hiddenDim, vocabSize, validCount);
            _gpu.CopyToHost((IntPtr)pValidLogits, _dScratchY, bytesOut);
        }
    }

    public void LaunchGemmBatch(
        GgufType type,
        IntPtr dY,
        IntPtr dX,
        IntPtr dW,
        int kCols,
        int mRows,
        int batchSize,
        IntPtr dBias = default,
        IntPtr dResidual = default,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBatch,
            GgufType.Q6_K => _fnGemmQ6KBatch,
            GgufType.Q8_0 => _fnGemmQ8_0Batch,
            GgufType.Q4_0 => _fnGemmQ4_0Batch,
            GgufType.F32 => _fnGemmF32Batch,
            _ => throw new NotSupportedException($"GPU batch GEMM does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        int tileTokens = type == GgufType.Q4_K ? 16 : 8;
        uint gridY = (uint)((batchSize + tileTokens - 1) / tileTokens);

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dBias;
        pArgs[7] = &dResidual;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_batch)");
    }

    public void LaunchSwiglu(IntPtr dDst, IntPtr dGate, IntPtr dUp, int size)
    {
        uint blockSize = 256;
        uint gridX = (uint)((size + 255) / 256);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dGate;
        pArgs[1] = &dUp;
        pArgs[2] = &dDst;
        pArgs[3] = &size;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwiglu,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(swiglu_kernel)");
    }

    public void LaunchGemmSwigluBatch(
        IntPtr dDst,
        IntPtr dX,
        IntPtr dWGate,
        IntPtr dWUp,
        int kCols,
        int mRows,
        int batchSize)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dDst;
        pArgs[1] = &dX;
        pArgs[2] = &dWGate;
        pArgs[3] = &dWUp;
        pArgs[4] = &kCols;
        pArgs[5] = &mRows;
        pArgs[6] = &batchSize;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmSwigluBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q4_k_swiglu_batch)");
    }

    public void LaunchVecAdd(IntPtr dA, IntPtr dB, int size)
    {
        uint blockSize = 256;
        uint gridX = (uint)((size + 255) / 256);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dA;
        pArgs[1] = &dB;
        pArgs[2] = &size;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAdd,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add_kernel)");
    }

    public void ForwardQKV(
        int layerIndex,
        Tensor<float> norm1X,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        LoraAdapter qLora,
        LoraAdapter kLora,
        LoraAdapter vLora,
        int seqLen)
    {
        int hiddenDim = _weights.EmbeddingLength;
        int headDim = _weights.HeadDim;
        int qDim = _weights.HeadCount * headDim;
        int kvDim = _weights.HeadCountKv * headDim;

        nuint normBytes = (nuint)((long)seqLen * hiddenDim * sizeof(float));
        nuint qBytes = (nuint)((long)seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)((long)seqLen * kvDim * sizeof(float));

        lock (_scratchLock)
        {
            EnsureQkvCapacity(normBytes, qBytes, kvBytes);

            _gpu.CopyToDevice(_dFfnNormX, (IntPtr)norm1X.DataPointer, normBytes);

            var lw = _weights.Layers[layerIndex];

            LaunchGemmBatch(lw.QType, _dQBuf, _dFfnNormX, _dQLayers[layerIndex], hiddenDim, qDim, seqLen, _dQBiasLayers[layerIndex]);
            LaunchGemmBatch(lw.KType, _dKBuf, _dFfnNormX, _dKLayers[layerIndex], hiddenDim, kvDim, seqLen, _dKBiasLayers[layerIndex]);
            LaunchGemmBatch(lw.VType, _dVBuf, _dFfnNormX, _dVLayers[layerIndex], hiddenDim, kvDim, seqLen, _dVBiasLayers[layerIndex]);

            _gpu.CopyToHost((IntPtr)q.DataPointer, _dQBuf, qBytes);
            _gpu.CopyToHost((IntPtr)k.DataPointer, _dKBuf, kvBytes);
            _gpu.CopyToHost((IntPtr)v.DataPointer, _dVBuf, kvBytes);
        }

        AddLoraDeltaHost(norm1X, qLora, q);
        AddLoraDeltaHost(norm1X, kLora, k);
        AddLoraDeltaHost(norm1X, vLora, v);
    }

    public void ForwardFfn(
        int layerIndex,
        Tensor<float> norm2X,
        Tensor<float>? outGate,
        Tensor<float>? outUp,
        Tensor<float>? outHidden,
        Tensor<float> outDown,
        LoraAdapter gateLora,
        LoraAdapter upLora,
        LoraAdapter downLora,
        int seqLen)
    {
        int hiddenDim = _weights.EmbeddingLength;
        int ffnDim = _weights.FeedForwardLength;

        nuint normBytes = (nuint)((long)seqLen * hiddenDim * sizeof(float));
        nuint ffnBytes = (nuint)((long)seqLen * ffnDim * sizeof(float));
        int totalFfnElements = seqLen * ffnDim;

        var lw = _weights.Layers[layerIndex];
        bool gateZero = gateLora.IsZeroDelta;
        bool upZero = upLora.IsZeroDelta;
        bool downZero = downLora.IsZeroDelta;

        // Fast path: when Gate & Up LoRA deltas are identically zero (e.g., step 0 cold start or disabled adapters)
        if (gateZero && upZero)
        {
            Tensor<float>? tempHidden = (!downZero && outHidden == null) ? new Tensor<float>(seqLen, ffnDim) : null;
            try
            {
                lock (_scratchLock)
                {
                    EnsureFfnCapacity(normBytes, ffnBytes);
                    _gpu.CopyToDevice(_dFfnNormX, (IntPtr)norm2X.DataPointer, normBytes);

                    if (outGate == null && outUp == null && lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
                    {
                        LaunchGemmSwigluBatch(_dFfnHidden, _dFfnNormX, _dGateLayers[layerIndex], _dUpLayers[layerIndex], hiddenDim, ffnDim, seqLen);
                    }
                    else
                    {
                        LaunchGemmBatch(lw.FfnGateType, _dFfnGate, _dFfnNormX, _dGateLayers[layerIndex], hiddenDim, ffnDim, seqLen);
                        LaunchGemmBatch(lw.FfnUpType, _dFfnUp, _dFfnNormX, _dUpLayers[layerIndex], hiddenDim, ffnDim, seqLen);

                        if (outGate != null) _gpu.CopyToHost((IntPtr)outGate.DataPointer, _dFfnGate, ffnBytes);
                        if (outUp != null) _gpu.CopyToHost((IntPtr)outUp.DataPointer, _dFfnUp, ffnBytes);

                        LaunchSwiglu(_dFfnHidden, _dFfnGate, _dFfnUp, totalFfnElements);
                    }

                    Tensor<float>? hiddenTarget = outHidden ?? tempHidden;
                    if (hiddenTarget != null)
                    {
                        _gpu.CopyToHost((IntPtr)hiddenTarget.DataPointer, _dFfnHidden, ffnBytes);
                    }

                    LaunchGemmBatch(lw.FfnDownType, _dFfnDownOut, _dFfnHidden, _dDownLayers[layerIndex], ffnDim, hiddenDim, seqLen);
                    _gpu.CopyToHost((IntPtr)outDown.DataPointer, _dFfnDownOut, normBytes);
                }

                if (!downZero)
                {
                    AddLoraDeltaHost((outHidden ?? tempHidden)!, downLora, outDown);
                }
            }
            finally
            {
                tempHidden?.Dispose();
            }
            return;
        }

        // Exact trained-adapter path: apply Gate & Up LoRA deltas BEFORE SwiGLU and Down projection
        Tensor<float> gateBuf = outGate ?? new Tensor<float>(seqLen, ffnDim);
        Tensor<float> upBuf = outUp ?? new Tensor<float>(seqLen, ffnDim);
        Tensor<float> hiddenBuf = outHidden ?? new Tensor<float>(seqLen, ffnDim);

        try
        {
            lock (_scratchLock)
            {
                EnsureFfnCapacity(normBytes, ffnBytes);
                _gpu.CopyToDevice(_dFfnNormX, (IntPtr)norm2X.DataPointer, normBytes);
                LaunchGemmBatch(lw.FfnGateType, _dFfnGate, _dFfnNormX, _dGateLayers[layerIndex], hiddenDim, ffnDim, seqLen);
                LaunchGemmBatch(lw.FfnUpType, _dFfnUp, _dFfnNormX, _dUpLayers[layerIndex], hiddenDim, ffnDim, seqLen);
                _gpu.CopyToHost((IntPtr)gateBuf.DataPointer, _dFfnGate, ffnBytes);
                _gpu.CopyToHost((IntPtr)upBuf.DataPointer, _dFfnUp, ffnBytes);
            }

            if (!gateZero) AddLoraDeltaHost(norm2X, gateLora, gateBuf);
            if (!upZero) AddLoraDeltaHost(norm2X, upLora, upBuf);

            SwiGluKernel.Forward(
                (float*)gateBuf.DataPointer,
                (float*)upBuf.DataPointer,
                (float*)hiddenBuf.DataPointer,
                totalFfnElements);

            lock (_scratchLock)
            {
                EnsureFfnCapacity(normBytes, ffnBytes);
                _gpu.CopyToDevice(_dFfnHidden, (IntPtr)hiddenBuf.DataPointer, ffnBytes);
                LaunchGemmBatch(lw.FfnDownType, _dFfnDownOut, _dFfnHidden, _dDownLayers[layerIndex], ffnDim, hiddenDim, seqLen);
                _gpu.CopyToHost((IntPtr)outDown.DataPointer, _dFfnDownOut, normBytes);
            }

            if (!downZero)
            {
                AddLoraDeltaHost(hiddenBuf, downLora, outDown);
            }
        }
        finally
        {
            if (outGate == null) gateBuf.Dispose();
            if (outUp == null) upBuf.Dispose();
            if (outHidden == null) hiddenBuf.Dispose();
        }
    }

    private static void AddLoraDeltaHost(Tensor<float> x, LoraAdapter adapter, Tensor<float> output)
    {
        if (!adapter.IsEnabled || adapter.IsZeroDelta) return;

        if (x.IsContiguous && output.IsContiguous && adapter.AdapterA.IsContiguous && adapter.AdapterB.IsContiguous)
        {
            int seqLen = x.Shape[0];
            int inFeatures = adapter.AdapterA.Shape[0];
            int outFeatures = adapter.AdapterB.Shape[1];
            int rank = adapter.AdapterA.Shape[1];
            LoraKernels.ApplyLoraDelta(
                x.DataPointer,
                adapter.AdapterA.DataPointer,
                adapter.AdapterB.DataPointer,
                output.DataPointer,
                seqLen,
                inFeatures,
                outFeatures,
                rank,
                adapter.Scaling);
            return;
        }

        using var lowRank = TensorOps.MatMul(x, adapter.AdapterA, GpuTarget.Cpu);
        using var delta = TensorOps.MatMul(lowRank, adapter.AdapterB, GpuTarget.Cpu);
        using var scaled = TensorOps.Scale(delta, adapter.Scaling);

        var outSpan = output.AsSpan();
        var deltaSpan = scaled.AsSpan();
        for (int i = 0; i < outSpan.Length; i++)
        {
            outSpan[i] += deltaSpan[i];
        }
    }

    public void LaunchRopeBatch(IntPtr dQ, IntPtr dK, int nHeadsQ, int nHeadsKv, int headDim, int seqLen, float freqBase, float freqScale = 1.0f)
    {
        int halfDim = headDim / 2;
        int totalHalf = (nHeadsQ + nHeadsKv) * halfDim;
        int totalAll = totalHalf * seqLen;
        uint blockSize = 256;
        uint gridSize = (uint)((totalAll + (int)blockSize - 1) / (int)blockSize);

        int startPos = 0;
        int batchSize = seqLen;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &freqBase;
        pArgs[8] = &freqScale;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRopeBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_batch)");
    }

    public void LaunchAttnTrainFwd(
        IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dAttnOut, IntPtr dProbs,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim, float scale)
    {
        if (headDim <= 0 || headDim > 1024)
            throw new ArgumentOutOfRangeException(nameof(headDim), $"GPU training attention requires 1 <= headDim <= 1024, got {headDim}.");

        uint blockSize = (uint)(((headDim + 31) / 32) * 32);
        uint gridX = (uint)nHeadsQ;
        uint gridY = (uint)seqLen;
        uint sharedMemBytes = dProbs != IntPtr.Zero ? (uint)(seqLen * sizeof(float)) : 0u;

        void** pArgs = stackalloc void*[10];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dAttnOut;
        pArgs[4] = &dProbs;
        pArgs[5] = &seqLen;
        pArgs[6] = &nHeadsQ;
        pArgs[7] = &nHeadsKv;
        pArgs[8] = &headDim;
        pArgs[9] = &scale;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainFwd,
            gridX, gridY, 1,
            blockSize, 1, 1,
            sharedMemBytes, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(tune_attn_causal_gqa_fwd)");
    }

    public void ForwardAttention(
        int layerIndex,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float> attnOut,
        Tensor<float>? attnProbs,
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
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);

            LaunchRopeBatch(_dQBuf, _dKBuf, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, 1.0f);

            if (attnProbs != null)
            {
                _gpu.CopyToHost((IntPtr)q.DataPointer, _dQBuf, qBytes);
                _gpu.CopyToHost((IntPtr)k.DataPointer, _dKBuf, kvBytes);
            }

            IntPtr dProbs = attnProbs != null ? _dAttnProbs : IntPtr.Zero;
            LaunchAttnTrainFwd(_dQBuf, _dKBuf, _dVBuf, _dAttnOut, dProbs, seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            _gpu.CopyToHost((IntPtr)attnOut.DataPointer, _dAttnOut, qBytes);
            if (attnProbs != null)
            {
                nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
                _gpu.CopyToHost((IntPtr)attnProbs.DataPointer, _dAttnProbs, probsBytes);
            }
        }
    }
}
