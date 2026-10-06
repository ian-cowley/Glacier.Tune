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
        nuint bytesIn = (nuint)(seqLen * inFeatures * sizeof(float));
        nuint bytesOut = (nuint)(seqLen * outFeatures * sizeof(float));

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

            // Copy input activations to GPU scratch buffer
            _gpu.CopyToDevice(_dScratchX, (IntPtr)x.DataPointer, bytesIn);

            // Execute batched quantized GEMM kernel in VRAM
            LaunchGemmBatch(type, _dScratchY, _dScratchX, dW, inFeatures, outFeatures, seqLen, dBias);

            // Copy output activations back to host tensor
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
        nuint bytesIn = (nuint)(validCount * hiddenDim * sizeof(float));
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

        nuint normBytes = (nuint)(seqLen * hiddenDim * sizeof(float));
        nuint qBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));

        lock (_scratchLock)
        {
            EnsureQkvCapacity(normBytes, qBytes, kvBytes);

            // Copy norm1X to device ONCE (eliminates 2 redundant PCIe copies)
            _gpu.CopyToDevice(_dFfnNormX, (IntPtr)norm1X.DataPointer, normBytes);

            var lw = _weights.Layers[layerIndex];

            // Execute Q, K, V projections in VRAM
            LaunchGemmBatch(lw.QType, _dQBuf, _dFfnNormX, _dQLayers[layerIndex], hiddenDim, qDim, seqLen, _dQBiasLayers[layerIndex]);
            LaunchGemmBatch(lw.KType, _dKBuf, _dFfnNormX, _dKLayers[layerIndex], hiddenDim, kvDim, seqLen, _dKBiasLayers[layerIndex]);
            LaunchGemmBatch(lw.VType, _dVBuf, _dFfnNormX, _dVLayers[layerIndex], hiddenDim, kvDim, seqLen, _dVBiasLayers[layerIndex]);

            // Copy back to host tensors
            _gpu.CopyToHost((IntPtr)q.DataPointer, _dQBuf, qBytes);
            _gpu.CopyToHost((IntPtr)k.DataPointer, _dKBuf, kvBytes);
            _gpu.CopyToHost((IntPtr)v.DataPointer, _dVBuf, kvBytes);
        }

        // Add LoRA adapter deltas on host
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

        nuint normBytes = (nuint)(seqLen * hiddenDim * sizeof(float));
        nuint ffnBytes = (nuint)(seqLen * ffnDim * sizeof(float));
        int totalFfnElements = seqLen * ffnDim;

        lock (_scratchLock)
        {
            EnsureFfnCapacity(normBytes, ffnBytes);

            // 1. Copy norm2X to GPU ONCE
            _gpu.CopyToDevice(_dFfnNormX, (IntPtr)norm2X.DataPointer, normBytes);

            // 2. Launch Gate and Up base GEMMs in VRAM
            var lw = _weights.Layers[layerIndex];
            if (outGate == null && outUp == null && lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
            {
                // Fused Gate + Up + SwiGLU: SiLU(Gate) * Up in registers in a single kernel pass!
                LaunchGemmSwigluBatch(_dFfnHidden, _dFfnNormX, _dGateLayers[layerIndex], _dUpLayers[layerIndex], hiddenDim, ffnDim, seqLen);
            }
            else
            {
                LaunchGemmBatch(lw.FfnGateType, _dFfnGate, _dFfnNormX, _dGateLayers[layerIndex], hiddenDim, ffnDim, seqLen);
                LaunchGemmBatch(lw.FfnUpType, _dFfnUp, _dFfnNormX, _dUpLayers[layerIndex], hiddenDim, ffnDim, seqLen);

                // 3. Save Gate & Up to host if required for activation checkpointing backward pass
                if (outGate != null) _gpu.CopyToHost((IntPtr)outGate.DataPointer, _dFfnGate, ffnBytes);
                if (outUp != null) _gpu.CopyToHost((IntPtr)outUp.DataPointer, _dFfnUp, ffnBytes);

                // 4. Compute SwiGLU in VRAM: Hidden = SiLU(Gate) * Up
                LaunchSwiglu(_dFfnHidden, _dFfnGate, _dFfnUp, totalFfnElements);
            }

            // 5. Save Hidden to host if required
            if (outHidden != null) _gpu.CopyToHost((IntPtr)outHidden.DataPointer, _dFfnHidden, ffnBytes);

            // 6. Launch Down base GEMM in VRAM: DownOut = Hidden * W_down
            LaunchGemmBatch(lw.FfnDownType, _dFfnDownOut, _dFfnHidden, _dDownLayers[layerIndex], ffnDim, hiddenDim, seqLen);

            // 7. Copy DownOut back to host (1 PCIe transfer)
            _gpu.CopyToHost((IntPtr)outDown.DataPointer, _dFfnDownOut, normBytes);
        }

        // 8. Add LoRA adapter deltas
        if (outGate != null) AddLoraDeltaHost(norm2X, gateLora, outGate);
        if (outUp != null) AddLoraDeltaHost(norm2X, upLora, outUp);
        if (outHidden != null) AddLoraDeltaHost(outHidden, downLora, outDown);
    }

    private static void AddLoraDeltaHost(Tensor<float> x, LoraAdapter adapter, Tensor<float> output)
    {
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

        // output += (alpha / r) * (x * A) * B
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
        uint blockSize = (uint)headDim;
        uint gridX = (uint)nHeadsQ;
        uint gridY = (uint)seqLen;
        uint sharedMemBytes = dProbs != IntPtr.Zero ? (uint)((4 + seqLen) * sizeof(float)) : 16;

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

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainFwd,
            gridX, gridY, 1,
            blockSize, 1, 1,
            sharedMemBytes, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_causal_gqa_train_fwd)");
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
        nuint qBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            // Copy Q, K, V to VRAM buffers
            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);

            // 1. RoPE in VRAM (GPU batch kernel)
            LaunchRopeBatch(_dQBuf, _dKBuf, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, 1.0f);

            // Copy RoPE-applied Q and K back to host if needed for backward
            if (attnProbs != null)
            {
                _gpu.CopyToHost((IntPtr)q.DataPointer, _dQBuf, qBytes);
                _gpu.CopyToHost((IntPtr)k.DataPointer, _dKBuf, kvBytes);
            }

            // 2. Causal FlashAttention in VRAM
            IntPtr dProbs = attnProbs != null ? _dAttnProbs : IntPtr.Zero;
            LaunchAttnTrainFwd(_dQBuf, _dKBuf, _dVBuf, _dAttnOut, dProbs, seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            // Copy results back
            _gpu.CopyToHost((IntPtr)attnOut.DataPointer, _dAttnOut, qBytes);
            if (attnProbs != null)
            {
                nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
                _gpu.CopyToHost((IntPtr)attnProbs.DataPointer, _dAttnProbs, probsBytes);
            }
        }
    }
}
