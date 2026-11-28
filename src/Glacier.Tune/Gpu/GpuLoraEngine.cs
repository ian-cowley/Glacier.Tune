namespace Glacier.Tune.Gpu;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Model;

/// <summary>
/// High-performance GPU VRAM accelerator for Glacier.Tune.
/// Uploads frozen base GGUF model weights to NVIDIA VRAM once at startup,
/// and executes batched sequence forward and backward GEMMs directly on CUDA cores.
/// </summary>
public sealed unsafe partial class GpuLoraEngine : IDisposable
{
    private static GpuLoraEngine? s_current;
    public static GpuLoraEngine? Current => s_current;

    private readonly GpuContext _gpu;
    private readonly ModelWeights _weights;
    private readonly IntPtr _module;
    private readonly IntPtr _bwdModule;

    // Forward & ops kernel function handles (from Glacier.Inference fatbinary)
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
    private readonly IntPtr _fnAttnBwdPreprocessD;
    private readonly IntPtr _fnAttnTrainBwdDq;
    private readonly IntPtr _fnAttnTrainBwdDkDv;

    // Extended forward, norm, LoRA, CE & transpose-backward kernel handles (from Glacier.Tune tune_bwd.cubin)
    private readonly IntPtr _fnGemmF32Batch;
    private readonly IntPtr _fnGemmQ4_0Batch;
    private readonly IntPtr _fnGemmQ4KBwd;
    private readonly IntPtr _fnGemmQ6KBwd;
    private readonly IntPtr _fnGemmQ8_0Bwd;
    private readonly IntPtr _fnGemmQ4_0Bwd;
    private readonly IntPtr _fnGemmF32Bwd;
    private readonly IntPtr _fnRmsNormBwd;
    private readonly IntPtr _fnResAddOut;
    private readonly IntPtr _fnScaleBuf;
    private readonly IntPtr _fnLoraProjLr;
    private readonly IntPtr _fnLoraAddB;
    private readonly IntPtr _fnLoraBwdDb;
    private readonly IntPtr _fnLoraBwdDlr;
    private readonly IntPtr _fnLoraBwdDa;
    private readonly IntPtr _fnLoraBwdDx;
    private readonly IntPtr _fnCeTileStats;
    private readonly IntPtr _fnCeFinalizeLoss;
    private readonly IntPtr _fnCeWriteGrad;

    // Scratch buffers in VRAM
    private IntPtr _dScratchX;
    private nuint _scratchXCap;
    private IntPtr _dScratchY;
    private nuint _scratchYCap;
    private readonly object _scratchLock = new();

    // Dedicated FFN VRAM buffers
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

    // Dedicated Attention VRAM buffers (O(H * S) FlashAttention-2 LSE + D)
    private IntPtr _dAttnOut;
    private nuint _attnOutCap;
    private IntPtr _dAttnLse;
    private nuint _attnLseCap;
    private IntPtr _dAttnD;
    private nuint _attnDCap;
    private IntPtr _dDq;
    private nuint _dqCap;
    private IntPtr _dDk;
    private nuint _dkCap;
    private IntPtr _dDv;
    private nuint _dvCap;

    // Dedicated Cross-Entropy & LoRA VRAM scratch buffers
    private IntPtr _dCeMax;
    private IntPtr _dCeArgMax;
    private IntPtr _dCeSumOther;
    private IntPtr _dCeTargetLogits;
    private IntPtr _dCeInvDenom;
    private IntPtr _dCeLosses;
    private IntPtr _dCeTargetIds;
    private int _ceTokenCap;

    private IntPtr _dLoraTemp;
    private nuint _loraTempCap;

    private sealed class LoraGpuBuffers
    {
        public IntPtr DA;
        public IntPtr DB;
        public IntPtr DGradA;
        public IntPtr DGradB;
        public nuint ABytes;
        public nuint BBytes;
    }

    private readonly Dictionary<LoraAdapter, LoraGpuBuffers> _loraBuffers = new();

    // VRAM Weight pointers for transformer layers
    private readonly IntPtr[] _dQLayers;
    private readonly IntPtr[] _dQBiasLayers;
    private readonly IntPtr[] _dKLayers;
    private readonly IntPtr[] _dKBiasLayers;
    private readonly IntPtr[] _dVLayers;
    private readonly IntPtr[] _dVBiasLayers;
    private readonly IntPtr[] _dAttnOutLayers;
    private readonly IntPtr[] _dAttnOutBiasLayers;
    private readonly IntPtr[] _dGateLayers;
    private readonly IntPtr[] _dUpLayers;
    private readonly IntPtr[] _dDownLayers;
    private readonly IntPtr[] _dAttnNormLayers;
    private readonly IntPtr[] _dFfnNormLayers;

    // LM Head & Final RMSNorm weight in VRAM
    private IntPtr _dLmHeadWeight;
    private IntPtr _dOutNormWeight;

    private bool _disposed;

    public GpuContext Gpu => _gpu;
    public IntPtr LmHeadWeight => _dLmHeadWeight;
    public IntPtr OutNormWeight => _dOutNormWeight;

    public bool SupportsAttention(int seqLen, int headDim) =>
        seqLen > 0 && headDim is > 0 and <= 1024;

    public GpuLoraEngine(ModelWeights weights, int deviceOrdinal = 0)
    {
        _weights = weights;
        int blockCount = weights.BlockCount;
        _dQLayers = new IntPtr[blockCount];
        _dQBiasLayers = new IntPtr[blockCount];
        _dKLayers = new IntPtr[blockCount];
        _dKBiasLayers = new IntPtr[blockCount];
        _dVLayers = new IntPtr[blockCount];
        _dVBiasLayers = new IntPtr[blockCount];
        _dAttnOutLayers = new IntPtr[blockCount];
        _dAttnOutBiasLayers = new IntPtr[blockCount];
        _dGateLayers = new IntPtr[blockCount];
        _dUpLayers = new IntPtr[blockCount];
        _dDownLayers = new IntPtr[blockCount];
        _dAttnNormLayers = new IntPtr[blockCount];
        _dFfnNormLayers = new IntPtr[blockCount];

        _gpu = new GpuContext(deviceOrdinal);

        try
        {
            // 1. Load primary CUDA fatbinary kernel module
            byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
            CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), $"ModuleLoadData({_gpu.ArchString})");

            // 2. Load Glacier.Tune backward/extended GEMM CUDA fatbinary module
            byte[] bwdCubin = LoadTuneBackwardCubin();
            CuDriver.Check(CuDriver.ModuleLoadData(out _bwdModule, bwdCubin), $"ModuleLoadData(tune_bwd:{_gpu.ArchString})");

            // 3. Retrieve kernel function handles
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _module, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBatch, _module, "gemm_q6_k_batch"), "ModuleGetFunction(gemm_q6_k_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ8_0Batch, _module, "gemm_q8_0_batch"), "ModuleGetFunction(gemm_q8_0_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmSwigluBatch, _module, "gemm_q4_k_swiglu_batch"), "ModuleGetFunction(gemm_q4_k_swiglu_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatch, _module, "rms_norm_batch"), "ModuleGetFunction(rms_norm_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "swiglu_kernel"), "ModuleGetFunction(swiglu_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwigluBwd, _module, "swiglu_bwd_kernel"), "ModuleGetFunction(swiglu_bwd_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _module, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRopeBatch, _module, "rope_batch"), "ModuleGetFunction(rope_batch)");

            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainFwd, _bwdModule, "tune_attn_causal_gqa_fwd"), "ModuleGetFunction(tune_attn_causal_gqa_fwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnBwdPreprocessD, _bwdModule, "tune_attn_bwd_preprocess_d"), "ModuleGetFunction(tune_attn_bwd_preprocess_d)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainBwdDq, _bwdModule, "tune_attn_causal_gqa_bwd_dq"), "ModuleGetFunction(tune_attn_causal_gqa_bwd_dq)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnTrainBwdDkDv, _bwdModule, "tune_attn_causal_gqa_bwd_dk_dv"), "ModuleGetFunction(tune_attn_causal_gqa_bwd_dk_dv)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmF32Batch, _bwdModule, "gemm_f32_batch"), "ModuleGetFunction(gemm_f32_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4_0Batch, _bwdModule, "gemm_q4_0_batch"), "ModuleGetFunction(gemm_q4_0_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBwd, _bwdModule, "gemm_q4_k_bwd"), "ModuleGetFunction(gemm_q4_k_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBwd, _bwdModule, "gemm_q6_k_bwd"), "ModuleGetFunction(gemm_q6_k_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ8_0Bwd, _bwdModule, "gemm_q8_0_bwd"), "ModuleGetFunction(gemm_q8_0_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4_0Bwd, _bwdModule, "gemm_q4_0_bwd"), "ModuleGetFunction(gemm_q4_0_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmF32Bwd, _bwdModule, "gemm_f32_bwd"), "ModuleGetFunction(gemm_f32_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBwd, _bwdModule, "tune_rmsnorm_bwd"), "ModuleGetFunction(tune_rmsnorm_bwd)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnResAddOut, _bwdModule, "tune_res_add_out"), "ModuleGetFunction(tune_res_add_out)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnScaleBuf, _bwdModule, "tune_scale_buf"), "ModuleGetFunction(tune_scale_buf)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraProjLr, _bwdModule, "tune_lora_proj_lr"), "ModuleGetFunction(tune_lora_proj_lr)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraAddB, _bwdModule, "tune_lora_add_b"), "ModuleGetFunction(tune_lora_add_b)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraBwdDb, _bwdModule, "tune_lora_bwd_db"), "ModuleGetFunction(tune_lora_bwd_db)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraBwdDlr, _bwdModule, "tune_lora_bwd_dlr"), "ModuleGetFunction(tune_lora_bwd_dlr)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraBwdDa, _bwdModule, "tune_lora_bwd_da"), "ModuleGetFunction(tune_lora_bwd_da)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLoraBwdDx, _bwdModule, "tune_lora_bwd_dx"), "ModuleGetFunction(tune_lora_bwd_dx)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnCeTileStats, _bwdModule, "tune_ce_tile_stats"), "ModuleGetFunction(tune_ce_tile_stats)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnCeFinalizeLoss, _bwdModule, "tune_ce_finalize_loss"), "ModuleGetFunction(tune_ce_finalize_loss)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnCeWriteGrad, _bwdModule, "tune_ce_write_grad"), "ModuleGetFunction(tune_ce_write_grad)");

            int initSeq = Math.Min(512, Math.Max(16, weights.ContextLength));
            _scratchXCap = (nuint)((long)initSeq * Math.Max(weights.EmbeddingLength, weights.FeedForwardLength) * sizeof(float));
            _scratchYCap = _scratchXCap;
            _dScratchX = _gpu.AllocateDevice(_scratchXCap);
            _dScratchY = _gpu.AllocateDevice(_scratchYCap);

            _ffnNormXCap = (nuint)((long)initSeq * weights.EmbeddingLength * sizeof(float));
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);

            _ffnGateCap = (nuint)((long)initSeq * weights.FeedForwardLength * sizeof(float));
            _dFfnGate = _gpu.AllocateDevice(_ffnGateCap);

            _ffnUpCap = _ffnGateCap;
            _dFfnUp = _gpu.AllocateDevice(_ffnUpCap);

            _ffnHiddenCap = _ffnGateCap;
            _dFfnHidden = _gpu.AllocateDevice(_ffnHiddenCap);

            _ffnDownOutCap = _ffnNormXCap;
            _dFfnDownOut = _gpu.AllocateDevice(_ffnDownOutCap);

            int qDim = weights.HeadCount * weights.HeadDim;
            int kvDim = weights.HeadCountKv * weights.HeadDim;
            _qBufCap = (nuint)((long)initSeq * qDim * sizeof(float));
            _dQBuf = _gpu.AllocateDevice(_qBufCap);

            _kBufCap = (nuint)((long)initSeq * kvDim * sizeof(float));
            _dKBuf = _gpu.AllocateDevice(_kBufCap);

            _vBufCap = _kBufCap;
            _dVBuf = _gpu.AllocateDevice(_vBufCap);

            _attnOutCap = _qBufCap;
            _dAttnOut = _gpu.AllocateDevice(_attnOutCap);
            _dqCap = _qBufCap;
            _dDq = _gpu.AllocateDevice(_dqCap);
            _dkCap = _kBufCap;
            _dDk = _gpu.AllocateDevice(_dkCap);
            _dvCap = _kBufCap;
            _dDv = _gpu.AllocateDevice(_dvCap);

            _attnLseCap = (nuint)((long)weights.HeadCount * initSeq * sizeof(float));
            _attnDCap = _attnLseCap;
            _dAttnLse = _gpu.AllocateDevice(_attnLseCap);
            _dAttnD = _gpu.AllocateDevice(_attnDCap);

            _loraTempCap = (nuint)((long)initSeq * 128 * sizeof(float));
            _dLoraTemp = _gpu.AllocateDevice(_loraTempCap);

            EnsureCeCapacity(initSeq);

            UploadBaseWeights();
            s_current = this;
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static byte[] LoadTuneBackwardCubin()
    {
        var asm = typeof(GpuLoraEngine).Assembly;
        using var stream = asm.GetManifestResourceStream("Glacier.Tune.Gpu.Kernels.tune_bwd.cubin");
        if (stream != null)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            return ms.ToArray();
        }

        string localPath = Path.Combine(AppContext.BaseDirectory, "Gpu", "Kernels", "tune_bwd.cubin");
        if (File.Exists(localPath))
        {
            return File.ReadAllBytes(localPath);
        }

        throw new FileNotFoundException("Embedded CUDA training module 'Glacier.Tune.Gpu.Kernels.tune_bwd.cubin' not found.");
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

            if (lw.AttnOutBias != null)
            {
                nuint attnOutBiasBytes = (nuint)(dim * sizeof(float));
                _dAttnOutBiasLayers[l] = _gpu.AllocateDevice(attnOutBiasBytes);
                _gpu.CopyToDevice(_dAttnOutBiasLayers[l], (IntPtr)lw.AttnOutBias, attnOutBiasBytes);
            }

            nuint gateBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnGateType, dim) * (nuint)ffnDim;
            _dGateLayers[l] = _gpu.AllocateDevice(gateBytes);
            _gpu.CopyToDevice(_dGateLayers[l], (IntPtr)lw.FfnGateWeight, gateBytes);

            nuint upBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnUpType, dim) * (nuint)ffnDim;
            _dUpLayers[l] = _gpu.AllocateDevice(upBytes);
            _gpu.CopyToDevice(_dUpLayers[l], (IntPtr)lw.FfnUpWeight, upBytes);

            nuint downBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnDownType, ffnDim) * (nuint)dim;
            _dDownLayers[l] = _gpu.AllocateDevice(downBytes);
            _gpu.CopyToDevice(_dDownLayers[l], (IntPtr)lw.FfnDownWeight, downBytes);

            if (lw.AttnNormWeight != null)
            {
                nuint normBytes = (nuint)(dim * sizeof(float));
                _dAttnNormLayers[l] = _gpu.AllocateDevice(normBytes);
                _gpu.CopyToDevice(_dAttnNormLayers[l], (IntPtr)lw.AttnNormWeight, normBytes);
            }

            if (lw.FfnNormWeight != null)
            {
                nuint normBytes = (nuint)(dim * sizeof(float));
                _dFfnNormLayers[l] = _gpu.AllocateDevice(normBytes);
                _gpu.CopyToDevice(_dFfnNormLayers[l], (IntPtr)lw.FfnNormWeight, normBytes);
            }
        }

        if (_weights.OutNormWeight != null)
        {
            nuint outNormBytes = (nuint)(dim * sizeof(float));
            _dOutNormWeight = _gpu.AllocateDevice(outNormBytes);
            _gpu.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, outNormBytes);
        }

        nuint outBytes = (nuint)GgufTypes.GetRowBytes(_weights.OutType, dim) * (nuint)_weights.VocabSize;
        _dLmHeadWeight = _gpu.AllocateDevice(outBytes);
        _gpu.CopyToDevice(_dLmHeadWeight, (IntPtr)_weights.OutWeight, outBytes);

        sw.Stop();
        GlacierDiagnostics.LogInformation($"  [GPU VRAM] Model resident in GPU VRAM ({sw.Elapsed.TotalSeconds:F2}s, 100% Zero-Copy PCIe base weights)!");
    }

    public IntPtr GetQ(int layer) => _dQLayers[layer];
    public IntPtr GetQBias(int layer) => _dQBiasLayers[layer];
    public IntPtr GetK(int layer) => _dKLayers[layer];
    public IntPtr GetKBias(int layer) => _dKBiasLayers[layer];
    public IntPtr GetV(int layer) => _dVLayers[layer];
    public IntPtr GetVBias(int layer) => _dVBiasLayers[layer];
    public IntPtr GetAttnOut(int layer) => _dAttnOutLayers[layer];
    public IntPtr GetAttnOutBias(int layer) => _dAttnOutBiasLayers[layer];
    public IntPtr GetGate(int layer) => _dGateLayers[layer];
    public IntPtr GetUp(int layer) => _dUpLayers[layer];
    public IntPtr GetDown(int layer) => _dDownLayers[layer];
    public IntPtr GetAttnNorm(int layer) => _dAttnNormLayers[layer];
    public IntPtr GetFfnNorm(int layer) => _dFfnNormLayers[layer];

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (s_current == this) s_current = null;

        if (_gpu.Handle != IntPtr.Zero)
        {
            CuDriver.CtxSetCurrent(_gpu.Handle);
        }

        foreach (var kv in _loraBuffers)
        {
            _gpu.FreeDevice(kv.Value.DA);
            _gpu.FreeDevice(kv.Value.DB);
            _gpu.FreeDevice(kv.Value.DGradA);
            _gpu.FreeDevice(kv.Value.DGradB);
        }
        _loraBuffers.Clear();

        for (int l = 0; l < _dQLayers.Length; l++)
        {
            _gpu.FreeDevice(_dQLayers[l]);
            _gpu.FreeDevice(_dQBiasLayers[l]);
            _gpu.FreeDevice(_dKLayers[l]);
            _gpu.FreeDevice(_dKBiasLayers[l]);
            _gpu.FreeDevice(_dVLayers[l]);
            _gpu.FreeDevice(_dVBiasLayers[l]);
            _gpu.FreeDevice(_dAttnOutLayers[l]);
            _gpu.FreeDevice(_dAttnOutBiasLayers[l]);
            _gpu.FreeDevice(_dGateLayers[l]);
            _gpu.FreeDevice(_dUpLayers[l]);
            _gpu.FreeDevice(_dDownLayers[l]);
            _gpu.FreeDevice(_dAttnNormLayers[l]);
            _gpu.FreeDevice(_dFfnNormLayers[l]);
        }

        _gpu.FreeDevice(_dOutNormWeight);
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
        _gpu.FreeDevice(_dAttnLse);
        _gpu.FreeDevice(_dAttnD);
        _gpu.FreeDevice(_dDq);
        _gpu.FreeDevice(_dDk);
        _gpu.FreeDevice(_dDv);
        _gpu.FreeDevice(_dLoraTemp);
        _gpu.FreeDevice(_dCeMax);
        _gpu.FreeDevice(_dCeArgMax);
        _gpu.FreeDevice(_dCeSumOther);
        _gpu.FreeDevice(_dCeTargetLogits);
        _gpu.FreeDevice(_dCeInvDenom);
        _gpu.FreeDevice(_dCeLosses);
        _gpu.FreeDevice(_dCeTargetIds);

        if (_gpu.Handle != IntPtr.Zero)
        {
            CuDriver.CtxSetCurrent(_gpu.Handle);
            if (_bwdModule != IntPtr.Zero) CuDriver.ModuleUnload(_bwdModule);
            if (_module != IntPtr.Zero) CuDriver.ModuleUnload(_module);
        }

        _gpu.Dispose();
    }
}
