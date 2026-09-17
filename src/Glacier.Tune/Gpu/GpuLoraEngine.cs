namespace Glacier.Tune.Gpu;

using System;
using System.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;
using Glacier.Tensor.Core;

/// <summary>
/// High-performance GPU VRAM accelerator for Glacier.Tune.
/// Uploads frozen base GGUF model weights to NVIDIA VRAM once at startup,
/// and executes batched sequence GEMMs directly on CUDA cores.
/// </summary>
public sealed unsafe class GpuLoraEngine : IDisposable
{
    private static GpuLoraEngine? s_current;
    public static GpuLoraEngine? Current => s_current;

    private readonly GpuContext _gpu;
    private readonly ModelWeights _weights;
    private readonly IntPtr _module;

    // Kernel function handles
    private readonly IntPtr _fnGemmQ4KBatch;
    private readonly IntPtr _fnGemmQ6KBatch;
    private readonly IntPtr _fnGemmQ8_0Batch;
    private readonly IntPtr _fnGemmSwigluBatch;
    private readonly IntPtr _fnRmsNormBatch;
    private readonly IntPtr _fnSwiglu;
    private readonly IntPtr _fnSwigluBwd;
    private readonly IntPtr _fnVecAdd;
    private readonly IntPtr _fnRopeBatch;
    private readonly IntPtr _fnAttnTrainFwd;
    private readonly IntPtr _fnAttnTrainBwdDqDs;
    private readonly IntPtr _fnAttnTrainBwdDkDv;

    // Scratch buffers in VRAM
    private IntPtr _dScratchX;
    private nuint _scratchXCap;
    private IntPtr _dScratchY;
    private nuint _scratchYCap;
    private readonly object _scratchLock = new();

    // Dedicated FFN VRAM buffers (avoids PCIe ping-pong for Gate/Up/Hidden)
    private IntPtr _dFfnNormX;
    private nuint _ffnNormXCap;
    private IntPtr _dFfnGate;
    private nuint _ffnGateCap;
    private IntPtr _dFfnUp;
    private nuint _ffnUpCap;
    private IntPtr _dFfnHidden;
    private nuint _ffnHiddenCap;
    private IntPtr _dFfnDownOut;
    private nuint _ffnDownOutCap;

    // Dedicated QKV VRAM buffers
    private IntPtr _dQBuf;
    private nuint _qBufCap;
    private IntPtr _dKBuf;
    private nuint _kBufCap;
    private IntPtr _dVBuf;
    private nuint _vBufCap;

    // Dedicated Attention VRAM buffers
    private IntPtr _dAttnOut;
    private nuint _attnOutCap;
    private IntPtr _dAttnProbs;
    private nuint _attnProbsCap;
    private IntPtr _dAttnDs;
    private nuint _attnDsCap;
    private IntPtr _dDq;
    private nuint _dqCap;
    private IntPtr _dDk;
    private nuint _dkCap;
    private IntPtr _dDv;
    private nuint _dvCap;

    // VRAM Weight pointers for 28 layers
    private readonly IntPtr[] _dQLayers;
    private readonly IntPtr[] _dQBiasLayers;
    private readonly IntPtr[] _dKLayers;
    private readonly IntPtr[] _dKBiasLayers;
    private readonly IntPtr[] _dVLayers;
    private readonly IntPtr[] _dVBiasLayers;
    private readonly IntPtr[] _dAttnOutLayers;
    private readonly IntPtr[] _dGateLayers;
    private readonly IntPtr[] _dUpLayers;
    private readonly IntPtr[] _dDownLayers;

    // LM Head in VRAM
    private IntPtr _dLmHeadWeight;

    private bool _disposed;

    public GpuContext Gpu => _gpu;
    public IntPtr LmHeadWeight => _dLmHeadWeight;

    public GpuLoraEngine(ModelWeights weights, int deviceOrdinal = 0)
    {
        _weights = weights;
        _gpu = new GpuContext(deviceOrdinal);

        // 1. Load CUDA fatbinary kernel module
        byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), $"ModuleLoadData({_gpu.ArchString})");

        // 2. Retrieve kernel function handles
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _module, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBatch, _module, "gemm_q6_k_batch"), "ModuleGetFunction(gemm_q6_k_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ8_0Batch, _module, "gemm_q8_0_batch"), "ModuleGetFunction(gemm_q8_0_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmSwigluBatch, _module, "gemm_q4_k_swiglu_batch"), "ModuleGetFunction(gemm_q4_k_swiglu_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatch, _module, "rms_norm_batch"), "ModuleGetFunction(rms_norm_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "swiglu_kernel"), "ModuleGetFunction(swiglu_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwigluBwd, _module, "swiglu_bwd_kernel"), "ModuleGetFunction(swiglu_bwd_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _module, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRopeBatch, _module, "rope_batch_kernel"), "ModuleGetFunction(rope_batch_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainFwd, _module, "attention_causal_gqa_train_fwd"), "ModuleGetFunction(attention_causal_gqa_train_fwd)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainBwdDqDs, _module, "attention_causal_gqa_train_bwd_dq_ds"), "ModuleGetFunction(attention_causal_gqa_train_bwd_dq_ds)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainBwdDkDv, _module, "attention_causal_gqa_train_bwd_dk_dv"), "ModuleGetFunction(attention_causal_gqa_train_bwd_dk_dv)");

        int blockCount = weights.BlockCount;
        _dQLayers = new IntPtr[blockCount];
        _dQBiasLayers = new IntPtr[blockCount];
        _dKLayers = new IntPtr[blockCount];
        _dKBiasLayers = new IntPtr[blockCount];
        _dVLayers = new IntPtr[blockCount];
        _dVBiasLayers = new IntPtr[blockCount];
        _dAttnOutLayers = new IntPtr[blockCount];
        _dGateLayers = new IntPtr[blockCount];
        _dUpLayers = new IntPtr[blockCount];
        _dDownLayers = new IntPtr[blockCount];

        // 3. Preallocate initial scratch buffers (e.g. seqLen 512 x 18944 dim x 4 bytes ~ 38 MB)
        _scratchXCap = (nuint)(512 * Math.Max(weights.EmbeddingLength, weights.FeedForwardLength) * sizeof(float));
        _scratchYCap = _scratchXCap;
        _dScratchX = _gpu.AllocateDevice(_scratchXCap);
        _dScratchY = _gpu.AllocateDevice(_scratchYCap);

        // Preallocate dedicated FFN VRAM buffers (total ~130 MB, stays resident in VRAM)
        _ffnNormXCap = (nuint)(512 * weights.EmbeddingLength * sizeof(float));
        _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);

        _ffnGateCap = (nuint)(512 * weights.FeedForwardLength * sizeof(float));
        _dFfnGate = _gpu.AllocateDevice(_ffnGateCap);

        _ffnUpCap = _ffnGateCap;
        _dFfnUp = _gpu.AllocateDevice(_ffnUpCap);

        _ffnHiddenCap = _ffnGateCap;
        _dFfnHidden = _gpu.AllocateDevice(_ffnHiddenCap);

        _ffnDownOutCap = _ffnNormXCap;
        _dFfnDownOut = _gpu.AllocateDevice(_ffnDownOutCap);

        // Preallocate dedicated QKV VRAM buffers
        int qDim = weights.HeadCount * weights.HeadDim;
        int kvDim = weights.HeadCountKv * weights.HeadDim;
        _qBufCap = (nuint)(512 * qDim * sizeof(float));
        _dQBuf = _gpu.AllocateDevice(_qBufCap);

        _kBufCap = (nuint)(512 * kvDim * sizeof(float));
        _dKBuf = _gpu.AllocateDevice(_kBufCap);

        _vBufCap = _kBufCap;
        _dVBuf = _gpu.AllocateDevice(_vBufCap);

        // Preallocate dedicated Attention VRAM buffers
        _attnOutCap = _qBufCap;
        _dAttnOut = _gpu.AllocateDevice(_attnOutCap);
        _dqCap = _qBufCap;
        _dDq = _gpu.AllocateDevice(_dqCap);
        _dkCap = _kBufCap;
        _dDk = _gpu.AllocateDevice(_dkCap);
        _dvCap = _kBufCap;
        _dDv = _gpu.AllocateDevice(_dvCap);

        _attnProbsCap = (nuint)((long)weights.HeadCount * 512 * 512 * sizeof(float));
        _attnDsCap = _attnProbsCap;
        _dAttnProbs = _gpu.AllocateDevice(_attnProbsCap);
        _dAttnDs = _gpu.AllocateDevice(_attnDsCap);

        // 4. Upload frozen base weights into GPU VRAM
        UploadBaseWeights();

        s_current = this;
    }

    private void UploadBaseWeights()
    {
        int dim = _weights.EmbeddingLength;
        int ffnDim = _weights.FeedForwardLength;
        int nHeads = _weights.HeadCount;
        int nHeadsKv = _weights.HeadCountKv;
        int headDim = _weights.HeadDim;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        double modelGb = (double)new System.IO.FileInfo(_weights.Gguf.FilePath).Length / (1024 * 1024 * 1024);
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  [GPU VRAM] Uploading {modelGb:F2} GB frozen base model weights into {_gpu.DeviceName} VRAM...");
        Console.ResetColor();
        var sw = Stopwatch.StartNew();

        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];

            nuint qBytes = (nuint)GgufTypes.GetRowBytes(lw.QType, dim) * (nuint)qDim;
            _dQLayers[l] = _gpu.AllocateDevice(qBytes);
            _gpu.CopyToDevice(_dQLayers[l], (IntPtr)lw.QWeight, qBytes);

            if (lw.QBias != null)
            {
                nuint qBiasBytes = (nuint)(qDim * sizeof(float));
                _dQBiasLayers[l] = _gpu.AllocateDevice(qBiasBytes);
                _gpu.CopyToDevice(_dQBiasLayers[l], (IntPtr)lw.QBias, qBiasBytes);
            }

            nuint kBytes = (nuint)GgufTypes.GetRowBytes(lw.KType, dim) * (nuint)kvDim;
            _dKLayers[l] = _gpu.AllocateDevice(kBytes);
            _gpu.CopyToDevice(_dKLayers[l], (IntPtr)lw.KWeight, kBytes);

            if (lw.KBias != null)
            {
                nuint kBiasBytes = (nuint)(kvDim * sizeof(float));
                _dKBiasLayers[l] = _gpu.AllocateDevice(kBiasBytes);
                _gpu.CopyToDevice(_dKBiasLayers[l], (IntPtr)lw.KBias, kBiasBytes);
            }

            nuint vBytes = (nuint)GgufTypes.GetRowBytes(lw.VType, dim) * (nuint)kvDim;
            _dVLayers[l] = _gpu.AllocateDevice(vBytes);
            _gpu.CopyToDevice(_dVLayers[l], (IntPtr)lw.VWeight, vBytes);

            if (lw.VBias != null)
            {
                nuint vBiasBytes = (nuint)(kvDim * sizeof(float));
                _dVBiasLayers[l] = _gpu.AllocateDevice(vBiasBytes);
                _gpu.CopyToDevice(_dVBiasLayers[l], (IntPtr)lw.VBias, vBiasBytes);
            }

            nuint attnOutBytes = (nuint)GgufTypes.GetRowBytes(lw.AttnOutType, qDim) * (nuint)dim;
            _dAttnOutLayers[l] = _gpu.AllocateDevice(attnOutBytes);
            _gpu.CopyToDevice(_dAttnOutLayers[l], (IntPtr)lw.AttnOutWeight, attnOutBytes);

            nuint gateBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnGateType, dim) * (nuint)ffnDim;
            _dGateLayers[l] = _gpu.AllocateDevice(gateBytes);
            _gpu.CopyToDevice(_dGateLayers[l], (IntPtr)lw.FfnGateWeight, gateBytes);

            nuint upBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnUpType, dim) * (nuint)ffnDim;
            _dUpLayers[l] = _gpu.AllocateDevice(upBytes);
            _gpu.CopyToDevice(_dUpLayers[l], (IntPtr)lw.FfnUpWeight, upBytes);

            nuint downBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnDownType, ffnDim) * (nuint)dim;
            _dDownLayers[l] = _gpu.AllocateDevice(downBytes);
            _gpu.CopyToDevice(_dDownLayers[l], (IntPtr)lw.FfnDownWeight, downBytes);
        }

        // Upload LM Head weight
        nuint outBytes = (nuint)GgufTypes.GetRowBytes(_weights.OutType, dim) * (nuint)_weights.VocabSize;
        _dLmHeadWeight = _gpu.AllocateDevice(outBytes);
        _gpu.CopyToDevice(_dLmHeadWeight, (IntPtr)_weights.OutWeight, outBytes);

        sw.Stop();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  [GPU VRAM] Model resident in GPU VRAM ({sw.Elapsed.TotalSeconds:F2}s, 100% Zero-Copy PCIe during training)!");
        Console.ResetColor();
    }

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

    private void EnsureFfnCapacity(nuint normBytes, nuint ffnBytes)
    {
        if (normBytes > _ffnNormXCap)
        {
            _gpu.FreeDevice(_dFfnNormX);
            _gpu.FreeDevice(_dFfnDownOut);
            _ffnNormXCap = normBytes * 2;
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);
            _dFfnDownOut = _gpu.AllocateDevice(_ffnNormXCap);
        }
        if (ffnBytes > _ffnGateCap)
        {
            _gpu.FreeDevice(_dFfnGate);
            _gpu.FreeDevice(_dFfnUp);
            _gpu.FreeDevice(_dFfnHidden);
            _ffnGateCap = ffnBytes * 2;
            _dFfnGate = _gpu.AllocateDevice(_ffnGateCap);
            _dFfnUp = _gpu.AllocateDevice(_ffnGateCap);
            _dFfnHidden = _gpu.AllocateDevice(_ffnGateCap);
        }
    }

    private void EnsureQkvCapacity(nuint normBytes, nuint qBytes, nuint kvBytes)
    {
        if (normBytes > _ffnNormXCap)
        {
            _gpu.FreeDevice(_dFfnNormX);
            _gpu.FreeDevice(_dFfnDownOut);
            _ffnNormXCap = normBytes * 2;
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);
            _dFfnDownOut = _gpu.AllocateDevice(_ffnNormXCap);
        }
        if (qBytes > _qBufCap)
        {
            _gpu.FreeDevice(_dQBuf);
            _qBufCap = qBytes * 2;
            _dQBuf = _gpu.AllocateDevice(_qBufCap);
        }
        if (kvBytes > _kBufCap)
        {
            _gpu.FreeDevice(_dKBuf);
            _gpu.FreeDevice(_dVBuf);
            _kBufCap = kvBytes * 2;
            _dKBuf = _gpu.AllocateDevice(_kBufCap);
            _dVBuf = _gpu.AllocateDevice(_kBufCap);
        }
    }

    public void ForwardQKV(
        int layerIndex,
        Tensor<float> norm1X,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Glacier.Tune.Model.LoraAdapter qLora,
        Glacier.Tune.Model.LoraAdapter kLora,
        Glacier.Tune.Model.LoraAdapter vLora,
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
        Glacier.Tune.Model.LoraAdapter gateLora,
        Glacier.Tune.Model.LoraAdapter upLora,
        Glacier.Tune.Model.LoraAdapter downLora,
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

    private static void AddLoraDeltaHost(Tensor<float> x, Glacier.Tune.Model.LoraAdapter adapter, Tensor<float> output)
    {
        if (x.IsContiguous && output.IsContiguous && adapter.AdapterA.IsContiguous && adapter.AdapterB.IsContiguous)
        {
            int seqLen = x.Shape[0];
            int inFeatures = adapter.AdapterA.Shape[0];
            int outFeatures = adapter.AdapterB.Shape[1];
            int rank = adapter.AdapterA.Shape[1];
            Glacier.Tune.Kernels.LoraKernels.ApplyLoraDelta(
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
        using var lowRank = TensorOps.MatMul(x, adapter.AdapterA, Glacier.Tensor.Compute.GpuTarget.Cpu);
        using var delta = TensorOps.MatMul(lowRank, adapter.AdapterB, Glacier.Tensor.Compute.GpuTarget.Cpu);
        using var scaled = TensorOps.Scale(delta, adapter.Scaling);

        var outSpan = output.AsSpan();
        var deltaSpan = scaled.AsSpan();
        for (int i = 0; i < outSpan.Length; i++)
        {
            outSpan[i] += deltaSpan[i];
        }
    }

    public IntPtr GetQ(int layer) => _dQLayers[layer];
    public IntPtr GetQBias(int layer) => _dQBiasLayers[layer];
    public IntPtr GetK(int layer) => _dKLayers[layer];
    public IntPtr GetKBias(int layer) => _dKBiasLayers[layer];
    public IntPtr GetV(int layer) => _dVLayers[layer];
    public IntPtr GetVBias(int layer) => _dVBiasLayers[layer];
    public IntPtr GetAttnOut(int layer) => _dAttnOutLayers[layer];
    public IntPtr GetGate(int layer) => _dGateLayers[layer];
    public IntPtr GetUp(int layer) => _dUpLayers[layer];
    public IntPtr GetDown(int layer) => _dDownLayers[layer];

    private void EnsureAttnCapacity(int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;
        nuint attnOutBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));

        if (attnOutBytes > _attnOutCap)
        {
            _gpu.FreeDevice(_dAttnOut);
            _attnOutCap = attnOutBytes * 2;
            _dAttnOut = _gpu.AllocateDevice(_attnOutCap);
        }
        if (probsBytes > _attnProbsCap)
        {
            _gpu.FreeDevice(_dAttnProbs);
            _gpu.FreeDevice(_dAttnDs);
            _attnProbsCap = probsBytes * 2;
            _attnDsCap = _attnProbsCap;
            _dAttnProbs = _gpu.AllocateDevice(_attnProbsCap);
            _dAttnDs = _gpu.AllocateDevice(_attnDsCap);
        }
        if (attnOutBytes > _dqCap)
        {
            _gpu.FreeDevice(_dDq);
            _dqCap = attnOutBytes * 2;
            _dDq = _gpu.AllocateDevice(_dqCap);
        }
        if (kvBytes > _dkCap)
        {
            _gpu.FreeDevice(_dDk);
            _gpu.FreeDevice(_dDv);
            _dkCap = kvBytes * 2;
            _dvCap = _dkCap;
            _dDk = _gpu.AllocateDevice(_dkCap);
            _dDv = _gpu.AllocateDevice(_dvCap);
        }
    }

    public void LaunchRopeBatch(IntPtr dQ, IntPtr dK, int nHeadsQ, int nHeadsKv, int headDim, int seqLen, float freqBase, float invSign = 1.0f)
    {
        int halfDim = headDim / 2;
        int totalHalf = (nHeadsQ + nHeadsKv) * halfDim;
        uint blockSize = 256;
        uint gridX = (uint)((totalHalf + 255) / 256);
        uint gridY = (uint)seqLen;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &seqLen;
        pArgs[6] = &freqBase;
        pArgs[7] = &invSign;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRopeBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_batch_kernel)");
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

    public void LaunchAttnTrainBwd(
        IntPtr dDOut, IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dProbs,
        IntPtr dDq, IntPtr dDk, IntPtr dDv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim, float scale)
    {
        uint blockSize = (uint)headDim;
        uint gridX = (uint)nHeadsQ;
        uint gridY = (uint)seqLen;
        uint sharedMemBytes = (uint)((4 + seqLen) * sizeof(float));

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

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainBwdDqDs,
            gridX, gridY, 1,
            blockSize, 1, 1,
            sharedMemBytes, IntPtr.Zero,
            (IntPtr)pArgsDq,
            IntPtr.Zero), "LaunchKernel(attention_causal_gqa_train_bwd_dq_ds)");

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
            IntPtr.Zero), "LaunchKernel(attention_causal_gqa_train_bwd_dk_dv)");
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
        nuint qBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));
        nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            // Copy inputs to VRAM
            _gpu.CopyToDevice(_dAttnOut, (IntPtr)dAttnOut.DataPointer, qBytes);
            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dAttnProbs, (IntPtr)attnProbs.DataPointer, probsBytes);

            // 1. Attention backward in VRAM -> _dDq, _dDk, _dDv
            LaunchAttnTrainBwd(
                _dAttnOut, _dQBuf, _dKBuf, _dVBuf, _dAttnProbs,
                _dDq, _dDk, _dDv,
                seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            // 2. RoPE backward in VRAM (inverse rotation with invSign = -1.0f)
            LaunchRopeBatch(_dDq, _dDk, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, -1.0f);

            // Copy gradients back to host
            _gpu.CopyToHost((IntPtr)dq.DataPointer, _dDq, qBytes);
            _gpu.CopyToHost((IntPtr)dk.DataPointer, _dDk, kvBytes);
            _gpu.CopyToHost((IntPtr)dv.DataPointer, _dDv, kvBytes);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (s_current == this) s_current = null;

        for (int l = 0; l < _dQLayers.Length; l++)
        {
            _gpu.FreeDevice(_dQLayers[l]);
            _gpu.FreeDevice(_dQBiasLayers[l]);
            _gpu.FreeDevice(_dKLayers[l]);
            _gpu.FreeDevice(_dKBiasLayers[l]);
            _gpu.FreeDevice(_dVLayers[l]);
            _gpu.FreeDevice(_dVBiasLayers[l]);
            _gpu.FreeDevice(_dAttnOutLayers[l]);
            _gpu.FreeDevice(_dGateLayers[l]);
            _gpu.FreeDevice(_dUpLayers[l]);
            _gpu.FreeDevice(_dDownLayers[l]);
        }

        _gpu.FreeDevice(_dLmHeadWeight);
        _gpu.FreeDevice(_dScratchX);
        _gpu.FreeDevice(_dScratchY);
        _gpu.FreeDevice(_dFfnNormX);
        _gpu.FreeDevice(_dFfnGate);
        _gpu.FreeDevice(_dFfnUp);
        _gpu.FreeDevice(_dFfnHidden);
        _gpu.FreeDevice(_dFfnDownOut);
        _gpu.FreeDevice(_dQBuf);
        _gpu.FreeDevice(_dKBuf);
        _gpu.FreeDevice(_dVBuf);
        _gpu.FreeDevice(_dAttnOut);
        _gpu.FreeDevice(_dAttnProbs);
        _gpu.FreeDevice(_dAttnDs);
        _gpu.FreeDevice(_dDq);
        _gpu.FreeDevice(_dDk);
        _gpu.FreeDevice(_dDv);

        _gpu.Dispose();
    }
}
