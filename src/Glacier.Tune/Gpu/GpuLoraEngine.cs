namespace Glacier.Tune.Gpu;

using System;
using System.Diagnostics;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;
using Glacier.Tune.Diagnostics;

/// <summary>
/// High-performance GPU VRAM accelerator for Glacier.Tune.
/// Uploads frozen base GGUF model weights to NVIDIA VRAM once at startup,
/// and executes batched sequence GEMMs directly on CUDA cores.
/// </summary>
public sealed unsafe partial class GpuLoraEngine : IDisposable
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
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRopeBatch, _module, "rope_batch"), "ModuleGetFunction(rope_batch)");
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

        double modelGb = (double)new FileInfo(_weights.Gguf.FilePath).Length / (1024 * 1024 * 1024);
        GlacierDiagnostics.LogInformation($"  [GPU VRAM] Uploading {modelGb:F2} GB frozen base model weights into {_gpu.DeviceName} VRAM...");
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
        GlacierDiagnostics.LogInformation($"  [GPU VRAM] Model resident in GPU VRAM ({sw.Elapsed.TotalSeconds:F2}s, 100% Zero-Copy PCIe during training)!");
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
