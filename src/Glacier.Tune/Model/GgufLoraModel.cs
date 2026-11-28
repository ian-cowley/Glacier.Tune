using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Tokenizer;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Kernels;

namespace Glacier.Tune.Model;

/// <summary>
/// Full-scale Causal Transformer Language Model with PEFT LoRA adapters.
/// Supports full-sequence forward pass, activation checkpointing, fused large-vocabulary cross-entropy loss,
/// and exact reverse-mode backpropagation through both frozen quantized base projections and LoRA adapters.
/// </summary>
public sealed unsafe class GgufLoraModel : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly BpeTokenizer _tokenizer;
    private readonly LoraConfig _config;
    private readonly GpuTarget _target;
    private readonly Glacier.Tune.Gpu.GpuLoraEngine? _gpuEngine;
    private readonly List<TransformerBlock> _blocks = [];
    private readonly List<GdnBlock> _gdnBlocks = [];
    private readonly List<ITuneBlock> _layerBlocks = [];
    private readonly List<Tensor<float>> _trainableParameters = [];
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public LoraConfig Config => _config;
    public GpuTarget Target => _target;
    public IReadOnlyList<TransformerBlock> Blocks => _blocks;
    public IReadOnlyList<GdnBlock> GdnBlocks => _gdnBlocks;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;
    public bool IsHybrid => _gdnBlocks.Count > 0;
    public bool IsGpuAccelerated => _gpuEngine != null;

    public int LayerCount => _weights.BlockCount;
    public int HiddenDim => _weights.EmbeddingLength;
    public int FfnDim => _weights.FeedForwardLength;
    public int VocabSize => _weights.VocabSize;

    private GgufLoraModel(GgufFile gguf, ModelWeights weights, BpeTokenizer tokenizer, LoraConfig config, GpuTarget target = GpuTarget.Auto)
    {
        _gguf = gguf;
        _weights = weights;
        _tokenizer = tokenizer;
        _config = config;

        if (target is GpuTarget.Direct3D12 or GpuTarget.Vulkan or GpuTarget.Amd or GpuTarget.DualGpu)
        {
            throw new NotSupportedException(
                $"GPU backend '{target}' is not supported by Glacier.Tune. GpuLoraEngine supports NVIDIA CUDA or CPU execution only.");
        }

        if (target is GpuTarget.Nvidia or GpuTarget.NvidiaTensorCore)
        {
            if (!SupportsGpuEngine(weights, out string unsupportedReason))
            {
                throw new NotSupportedException(
                    $"Explicit GPU target '{target}' does not support this model architecture ({unsupportedReason}). Use device=\"auto\" or \"cpu\" for CPU execution.");
            }

            if (!Glacier.Inference.Gpu.GpuContext.IsSupported)
            {
                throw new PlatformNotSupportedException(
                    $"Explicit GPU target '{target}' was requested, but no CUDA-capable NVIDIA GPU or driver is available.");
            }

            _target = target;
            _gpuEngine = new Glacier.Tune.Gpu.GpuLoraEngine(weights);
        }
        else if (target == GpuTarget.Auto)
        {
            if (Glacier.Inference.Gpu.GpuContext.IsSupported)
            {
                if (SupportsGpuEngine(weights, out string autoUnsupportedReason))
                {
                    try
                    {
                        _gpuEngine = new Glacier.Tune.Gpu.GpuLoraEngine(weights);
                        _target = GpuTarget.Nvidia;
                    }
                    catch (Exception ex)
                    {
                        GlacierDiagnostics.LogWarning($"[GPU WARNING] Failed to initialize GPU VRAM base engine ({ex.Message}). Falling back to CPU memory-mapped base execution.", ex);
                        _gpuEngine = null;
                        _target = GpuTarget.Cpu;
                    }
                }
                else
                {
                    GlacierDiagnostics.LogInformation($"[INFO] Architecture uses unsupported GPU features ({autoUnsupportedReason}); executing in pure-C# CPU mode.");
                    _target = GpuTarget.Cpu;
                }
            }
            else
            {
                _target = GpuTarget.Cpu;
            }
        }
        else
        {
            _target = GpuTarget.Cpu;
        }

        try
        {
            InitializeBlocks();
            if (_gpuEngine != null)
            {
                for (int l = 0; l < _layerBlocks.Count; l++)
                {
                    _layerBlocks[l].ConnectGpuWeights(_gpuEngine);
                }
            }
        }
        catch
        {
            _gpuEngine?.Dispose();
            _gpuEngine = null;
            foreach (var b in _layerBlocks) b.Dispose();
            _layerBlocks.Clear();
            _blocks.Clear();
            _gdnBlocks.Clear();
            _trainableParameters.Clear();
            throw;
        }
    }

    private static bool SupportsGpuEngine(ModelWeights weights, out string unsupportedReason)
    {
        var reasons = new List<string>();
        if (weights.IsHybridSsm) reasons.Add("hybrid-SSM");
        if (weights.IsMoe) reasons.Add("MoE");
        if (weights.IsMla) reasons.Add("MLA");
        if (weights.IsDsa) reasons.Add("DSA");
        if (weights.HasRopeFreqs) reasons.Add("custom-rope-freqs");
        if (weights.RopeDimensionCount > 0 && weights.HeadDim > 0 && weights.RopeDimensionCount < weights.HeadDim)
        {
            reasons.Add("partial-RoPE");
        }

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw == null)
            {
                reasons.Add($"missing-layer-{l}");
                break;
            }
            if (lw.IsGdn) reasons.Add("GDN");
            if (lw.HasQGate) reasons.Add("Q-Gate");
            if (lw.QWeight == null || lw.KWeight == null || lw.VWeight == null) reasons.Add("fused-QKV");
            if (lw.AttnQNormWeight != null || lw.AttnKNormWeight != null) reasons.Add("QK-Norm");
            if (lw.AttnSinksWeight != null) reasons.Add("Attn-Sinks");
            if (lw.HasFusedGateUp || lw.FfnGateWeight == null || lw.FfnUpWeight == null || lw.FfnDownWeight == null) reasons.Add("fused-or-missing-FFN");
            if ((lw.HeadDim > 0 && weights.HeadDim > 0 && lw.HeadDim != weights.HeadDim) ||
                (lw.HeadsKv > 0 && lw.HeadsKv != weights.HeadCountKv))
            {
                reasons.Add("heterogeneous-heads");
            }
            if (lw.HasHyperConnections) reasons.Add("hyper-connections");
            if (lw.IsQsa) reasons.Add("QSA");
            if (lw.HasPle) reasons.Add("PLE");

            if (reasons.Count > 0)
            {
                break;
            }
        }

        if (reasons.Count == 0)
        {
            unsupportedReason = string.Empty;
            return true;
        }

        unsupportedReason = string.Join(", ", reasons);
        return false;
    }

    public static GgufLoraModel Load(string ggufPath, LoraConfig? config = null, string? device = "auto")
    {
        var target = ResolveTarget(device);
        if (target is GpuTarget.Direct3D12 or GpuTarget.Vulkan or GpuTarget.Amd or GpuTarget.DualGpu)
        {
            throw new NotSupportedException(
                $"GPU backend '{target}' is not supported by Glacier.Tune. GpuLoraEngine supports NVIDIA CUDA or CPU execution only.");
        }

        var gguf = new GgufFile(ggufPath);
        try
        {
            var weights = new ModelWeights(gguf);
            var tokenizer = new BpeTokenizer(gguf);
            return new GgufLoraModel(gguf, weights, tokenizer, config ?? new LoraConfig(), target);
        }
        catch
        {
            gguf.Dispose();
            throw;
        }
    }

    public static GpuTarget ResolveTarget(string? device)
    {
        if (string.IsNullOrWhiteSpace(device) || device.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return GpuTarget.Auto;
        }

        return device.Trim().ToLowerInvariant() switch
        {
            "cpu" => GpuTarget.Cpu,
            "cuda" or "gpu" or "nvidia" => GpuTarget.Nvidia,
            "tensorcore" => GpuTarget.NvidiaTensorCore,
            "d3d12" or "directml" => GpuTarget.Direct3D12,
            "vulkan" => GpuTarget.Vulkan,
            "amd" or "rocm" or "hip" => GpuTarget.Amd,
            "dualgpu" => GpuTarget.DualGpu,
            _ => throw new ArgumentException(
                $"Unknown compute device '{device}'. Supported values: auto, cpu, cuda, gpu, nvidia, tensorcore.",
                nameof(device))
        };
    }

    private void InitializeBlocks()
    {
        int defaultHeadDim = _weights.HeadDim > 0
            ? _weights.HeadDim
            : (_weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128);

        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l] ?? throw new InvalidDataException($"Missing layer weights for block {l}.");

            if (lw.IsGdn)
            {
                var gdn = new GdnBlock(
                    layerIndex: l,
                    weights: lw,
                    modelWeights: _weights,
                    config: _config,
                    target: _target);

                _gdnBlocks.Add(gdn);
                _layerBlocks.Add(gdn);
                _trainableParameters.AddRange(gdn.TrainableParameters);
            }
            else
            {
                int layerHeadDim = lw.HeadDim > 0 ? lw.HeadDim : defaultHeadDim;
                int layerHeadsKv = lw.HeadsKv > 0 ? lw.HeadsKv : _weights.HeadCountKv;
                int ropeDim = _weights.RopeDimensionCount > 0 ? Math.Min(_weights.RopeDimensionCount, layerHeadDim) : layerHeadDim;

                var block = new TransformerBlock(
                    layerIndex: l,
                    weights: lw,
                    hiddenDim: _weights.EmbeddingLength,
                    ffnDim: _weights.FeedForwardLength,
                    nHeadsQ: _weights.HeadCount,
                    nHeadsKv: layerHeadsKv,
                    headDim: layerHeadDim,
                    ropeFreqBase: _weights.RopeFreqBase,
                    rmsNormEps: _weights.RmsNormEps,
                    config: _config,
                    target: _target,
                    ropeDim: ropeDim,
                    ropeFreqs: _weights.RopeFreqsWeight);

                _blocks.Add(block);
                _layerBlocks.Add(block);
                _trainableParameters.AddRange(block.TrainableParameters);
            }
        }
    }

    /// <summary>
    /// Computes forward-only vocabulary logits [seqLen, VocabSize] across the full model and active LoRA adapters.
    /// </summary>
    public Tensor<float> ForwardLogits(ReadOnlySpan<int> inputTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (inputTokens.IsEmpty)
        {
            throw new ArgumentException("Input token sequence must not be empty.", nameof(inputTokens));
        }

        int seqLen = inputTokens.Length;
        for (int t = 0; t < seqLen; t++)
        {
            int token = inputTokens[t];
            if ((uint)token >= (uint)VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(inputTokens),
                    $"Input token ID {token} at index {t} is outside vocabulary bounds [0, {VocabSize}).");
            }
        }

        var currentX = new Tensor<float>(seqLen, HiddenDim);
        float* pX0 = (float*)currentX.DataPointer;
        for (int t = 0; t < seqLen; t++)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, inputTokens[t], pX0 + (long)t * HiddenDim, HiddenDim);
        }

        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            var nextX = _layerBlocks[l].Forward(currentX, seqLen, saveActivations: false, out _);
            currentX.Dispose();
            currentX = nextX;
        }

        using var finalNormOut = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNorm(currentX, wFinalNorm, finalNormOut, _weights.RmsNormEps);
        }
        currentX.Dispose();

        var logits = new Tensor<float>(seqLen, VocabSize);
        if (_gpuEngine != null && _gpuEngine.LmHeadWeight != IntPtr.Zero && _gpuEngine.SupportsForwardGemm(_weights.OutType))
        {
            _gpuEngine.ComputeLmHeadLogits(
                _weights.OutType,
                _gpuEngine.LmHeadWeight,
                (float*)finalNormOut.DataPointer,
                (float*)logits.DataPointer,
                HiddenDim,
                VocabSize,
                seqLen);
        }
        else
        {
            QuantKernels.MatMulBatch(
                _weights.OutType,
                _weights.OutWeight,
                (float*)finalNormOut.DataPointer,
                (float*)logits.DataPointer,
                HiddenDim,
                VocabSize,
                seqLen);
        }

        return logits;
    }

    /// <summary>
    /// Computes forward-only vocabulary logits [1, VocabSize] for the final token position only,
    /// avoiding full-sequence LM-head projection during autoregressive decoding.
    /// </summary>
    public Tensor<float> ForwardLastTokenLogits(ReadOnlySpan<int> inputTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (inputTokens.IsEmpty)
        {
            throw new ArgumentException("Input token sequence must not be empty.", nameof(inputTokens));
        }

        int seqLen = inputTokens.Length;
        for (int t = 0; t < seqLen; t++)
        {
            int token = inputTokens[t];
            if ((uint)token >= (uint)VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(inputTokens),
                    $"Input token ID {token} at index {t} is outside vocabulary bounds [0, {VocabSize}).");
            }
        }

        var currentX = new Tensor<float>(seqLen, HiddenDim);
        float* pX0 = (float*)currentX.DataPointer;
        for (int t = 0; t < seqLen; t++)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, inputTokens[t], pX0 + (long)t * HiddenDim, HiddenDim);
        }

        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            var nextX = _layerBlocks[l].Forward(currentX, seqLen, saveActivations: false, out _);
            currentX.Dispose();
            currentX = nextX;
        }

        using var lastHidden = new Tensor<float>(1, HiddenDim);
        float* pLastSrc = (float*)currentX.DataPointer + (long)(seqLen - 1) * HiddenDim;
        Buffer.MemoryCopy(pLastSrc, (float*)lastHidden.DataPointer, (ulong)(HiddenDim * sizeof(float)), (ulong)(HiddenDim * sizeof(float)));
        currentX.Dispose();

        using var finalNormOut = new Tensor<float>(1, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNorm(lastHidden, wFinalNorm, finalNormOut, _weights.RmsNormEps);
        }

        var logits = new Tensor<float>(1, VocabSize);
        if (_gpuEngine != null && _gpuEngine.LmHeadWeight != IntPtr.Zero && _gpuEngine.SupportsForwardGemm(_weights.OutType))
        {
            _gpuEngine.ComputeLmHeadLogits(
                _weights.OutType,
                _gpuEngine.LmHeadWeight,
                (float*)finalNormOut.DataPointer,
                (float*)logits.DataPointer,
                HiddenDim,
                VocabSize,
                1);
        }
        else
        {
            QuantKernels.MatMulBatch(
                _weights.OutType,
                _weights.OutWeight,
                (float*)finalNormOut.DataPointer,
                (float*)logits.DataPointer,
                HiddenDim,
                VocabSize,
                1);
        }

        return logits;
    }

    /// <summary>
    /// Computes forward-only cross-entropy validation loss over <paramref name="inputTokens"/> and <paramref name="targetTokens"/>
    /// without allocating or mutating gradient buffers.
    /// </summary>
    public float EvaluateLoss(int[] inputTokens, int[] targetTokens)
        => EvaluateLoss(inputTokens, targetTokens, out _);

    /// <summary>
    /// Computes forward-only cross-entropy validation loss over <paramref name="inputTokens"/> and <paramref name="targetTokens"/>
    /// without allocating or mutating gradient buffers, returning the count of valid supervised tokens.
    /// </summary>
    public unsafe float EvaluateLoss(int[] inputTokens, int[] targetTokens, out int validTokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputTokens);
        ArgumentNullException.ThrowIfNull(targetTokens);

        int seqLen = inputTokens.Length;
        if (seqLen < 2)
        {
            validTokenCount = 0;
            return 0f;
        }

        for (int t = 0; t < seqLen; t++)
        {
            int token = inputTokens[t];
            if ((uint)token >= (uint)VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(inputTokens),
                    $"Input token ID {token} at index {t} is outside vocabulary bounds [0, {VocabSize}).");
            }
        }

        if (_gpuEngine != null && CanUseResidentGpuPipeline(seqLen))
        {
            return EvaluateLossResident(_gpuEngine, inputTokens, targetTokens, seqLen, out validTokenCount);
        }

        var currentX = new Tensor<float>(seqLen, HiddenDim);
        float* pX0 = (float*)currentX.DataPointer;
        for (int t = 0; t < seqLen; t++)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, inputTokens[t], pX0 + (long)t * HiddenDim, HiddenDim);
        }

        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            var nextX = _layerBlocks[l].Forward(currentX, seqLen, saveActivations: false, out _);
            currentX.Dispose();
            currentX = nextX;
        }

        using var finalNormOut = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNorm(currentX, wFinalNorm, finalNormOut, _weights.RmsNormEps);
        }
        currentX.Dispose();

        return FusedCrossEntropyLoss.ComputeLoss(
            (float*)finalNormOut.DataPointer,
            targetTokens,
            _weights.OutType,
            _weights.OutWeight,
            seqLen,
            HiddenDim,
            VocabSize,
            out validTokenCount,
            _gpuEngine);
    }

    private float EvaluateLossResident(
        Glacier.Tune.Gpu.GpuLoraEngine gpu,
        int[] inputTokens,
        int[] targetTokens,
        int seqLen,
        out int validTokenCount)
    {
        gpu.BeginResidentStep(EnumerateAdapters());

        int x0Count = checked(seqLen * HiddenDim);
        float[] x0Host = ArrayPool<float>.Shared.Rent(x0Count);
        DeviceResidentTensor<float>? currentX = null;
        try
        {
            fixed (float* pX0 = x0Host)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, inputTokens[t], pX0 + (long)t * HiddenDim, HiddenDim);
                }
                currentX = gpu.AllocateResident(seqLen, HiddenDim);
                gpu.CopyHostToResident(pX0, currentX);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x0Host);
        }

        try
        {
            for (int l = 0; l < _blocks.Count; l++)
            {
                var nextX = _blocks[l].ForwardResident(gpu, currentX, seqLen, saveActivations: false, out _);
                currentX.Dispose();
                currentX = nextX;
            }

            using var finalNormOut = gpu.AllocateResident(seqLen, HiddenDim);
            gpu.LaunchRmsNormBatch(currentX.DevicePointer, gpu.OutNormWeight, finalNormOut.DevicePointer, seqLen, HiddenDim, _weights.RmsNormEps);

            return gpu.ComputeTiledLmHeadLossResident(
                _weights.OutType,
                finalNormOut,
                targetTokens,
                HiddenDim,
                VocabSize,
                seqLen,
                FusedCrossEntropyLoss.VocabTileSize,
                out validTokenCount);
        }
        finally
        {
            currentX?.Dispose();
        }
    }

    private bool CanUseResidentGpuPipeline(int seqLen)
    {
        if (_gpuEngine == null || _layerBlocks.Count == 0 || _gdnBlocks.Count > 0) return false;
        if (_gpuEngine.LmHeadWeight == IntPtr.Zero || _gpuEngine.OutNormWeight == IntPtr.Zero) return false;
        if (!_gpuEngine.SupportsForwardGemm(_weights.OutType) || !_gpuEngine.SupportsBackwardGemm(_weights.OutType))
            return false;

        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            if (_layerBlocks[l] is not TransformerBlock tb || !tb.SupportsFullResidentGpu(seqLen))
                return false;
        }
        return true;
    }

    private IEnumerable<LoraAdapter> EnumerateAdapters()
    {
        for (int l = 0; l < _blocks.Count; l++)
        {
            var tb = _blocks[l];
            yield return tb.QProj;
            yield return tb.KProj;
            yield return tb.VProj;
            yield return tb.OProj;
            yield return tb.GateProj;
            yield return tb.UpProj;
            yield return tb.DownProj;
        }
    }

    /// <summary>
    /// Executes the full-sequence forward pass through all layers (optionally with activation checkpointing),
    /// computes online fused cross-entropy loss against target labels, and backpropagates through
    /// all layers to accumulate exact gradients into all enabled LoRA adapters.
    /// </summary>
    public unsafe float ForwardLossAndBackward(int[] inputTokens, int[] targetTokens, bool checkpointActivations = false)
        => ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations, lossScale: 1.0f);

    /// <summary>
    /// Executes the full-sequence forward and backward pass with scalar gradient multiplier <paramref name="lossScale"/>.
    /// </summary>
    public unsafe float ForwardLossAndBackward(int[] inputTokens, int[] targetTokens, bool checkpointActivations, float lossScale)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputTokens);
        ArgumentNullException.ThrowIfNull(targetTokens);

        int seqLen = inputTokens.Length;
        if (seqLen < 2) return 0f;

        for (int t = 0; t < seqLen; t++)
        {
            int token = inputTokens[t];
            if ((uint)token >= (uint)VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(inputTokens),
                    $"Input token ID {token} at index {t} is outside vocabulary bounds [0, {VocabSize}).");
            }
        }

        if (_gpuEngine != null && CanUseResidentGpuPipeline(seqLen))
        {
            return ForwardLossAndBackwardResident(_gpuEngine, inputTokens, targetTokens, seqLen, checkpointActivations, lossScale);
        }

        // 1. Embedding lookup -> X0 [seqLen, hiddenDim]
        var x0 = new Tensor<float>(seqLen, HiddenDim);
        float* pX0 = (float*)x0.DataPointer;
        for (int t = 0; t < seqLen; t++)
        {
            int token = inputTokens[t];
            float* dest = pX0 + (long)t * HiddenDim;
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, dest, HiddenDim);
        }

        // 2. Forward pass through all layers
        var swFwd = Stopwatch.StartNew();
        var layerInputs = new Tensor<float>[LayerCount];
        var layerStates = new IDisposable?[LayerCount];
        Tensor<float> currentX = x0;

        for (int l = 0; l < LayerCount; l++)
        {
            layerInputs[l] = currentX;
            currentX = _layerBlocks[l].Forward(currentX, seqLen, saveActivations: !checkpointActivations, out layerStates[l]);
        }
        swFwd.Stop();

        Tensor<float> xFinalLayer = currentX;

        // 3. Final Output RMSNorm: normOut = RMSNorm(xFinalLayer, out_norm)
        using var finalNormOut = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNorm(xFinalLayer, wFinalNorm, finalNormOut, _weights.RmsNormEps);
        }

        // 4. Fused Cross-Entropy Loss & Output Gradient (dXFinalNorm)
        var swLoss = Stopwatch.StartNew();
        using var dFinalNormOut = new Tensor<float>(seqLen, HiddenDim);
        float loss = FusedCrossEntropyLoss.ComputeLossAndGradients(
            (float*)finalNormOut.DataPointer,
            inputTokens,
            targetTokens,
            _weights.OutType,
            _weights.OutWeight,
            (float*)dFinalNormOut.DataPointer,
            seqLen,
            HiddenDim,
            VocabSize,
            _gpuEngine);
        swLoss.Stop();

        if (lossScale != 1.0f)
        {
            var gradSpan = dFinalNormOut.AsSpan();
            for (int i = 0; i < gradSpan.Length; i++)
            {
                gradSpan[i] *= lossScale;
            }
        }

        // 5. Final Output RMSNorm Backward -> dXFinalLayer
        using var dXFinalLayer = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dFinalNormOut, xFinalLayer, wFinalNorm, dXFinalLayer, null, _weights.RmsNormEps);
        }

        // 6. Reverse-Mode Backpropagation
        var swRecompute = new Stopwatch();
        var swBwd = new Stopwatch();
        Tensor<float> currentDX = dXFinalLayer;

        for (int l = LayerCount - 1; l >= 0; l--)
        {
            var xIn = layerInputs[l];
            IDisposable state;

            if (checkpointActivations)
            {
                swRecompute.Start();
                using var recomputedOut = _layerBlocks[l].Forward(xIn, seqLen, saveActivations: true, out var st);
                swRecompute.Stop();
                state = st!;
            }
            else
            {
                state = layerStates[l]!;
            }

            // Backpropagate through layer l, updating adapters and computing dXIn
            swBwd.Start();
            var dXPrev = _layerBlocks[l].Backward(currentDX, state, xIn, seqLen);
            swBwd.Stop();
            state.Dispose();

            if (!ReferenceEquals(currentDX, dXFinalLayer))
            {
                currentDX.Dispose();
            }
            currentDX = dXPrev;
        }

        GlacierDiagnostics.LogDebug($"[PROFILE] Fwd: {swFwd.ElapsedMilliseconds} ms | Loss: {swLoss.ElapsedMilliseconds} ms | Recompute: {(checkpointActivations ? swRecompute.ElapsedMilliseconds : 0)} ms | Bwd: {swBwd.ElapsedMilliseconds} ms");

        // Cleanup activation checkpoints
        for (int l = 0; l < LayerCount; l++)
        {
            layerInputs[l].Dispose();
        }
        xFinalLayer.Dispose();
        currentDX.Dispose();

        return loss;
    }

    private float ForwardLossAndBackwardResident(
        Glacier.Tune.Gpu.GpuLoraEngine gpu,
        int[] inputTokens,
        int[] targetTokens,
        int seqLen,
        bool checkpointActivations,
        float lossScale)
    {
        gpu.BeginResidentStep(EnumerateAdapters());

        int x0Count = checked(seqLen * HiddenDim);
        DeviceResidentTensor<float> x0;
        float[] x0Host = ArrayPool<float>.Shared.Rent(x0Count);
        try
        {
            fixed (float* pX0 = x0Host)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, inputTokens[t], pX0 + (long)t * HiddenDim, HiddenDim);
                }
                x0 = gpu.AllocateResident(seqLen, HiddenDim);
                gpu.CopyHostToResident(pX0, x0);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x0Host);
        }

        var layerInputs = new DeviceResidentTensor<float>?[_blocks.Count];
        var layerStates = new ResidentBlockActivationState?[_blocks.Count];
        DeviceResidentTensor<float>? xFinalLayer = null;
        DeviceResidentTensor<float>? currentDX = null;

        try
        {
            var swFwd = Stopwatch.StartNew();
            DeviceResidentTensor<float> currentX = x0;
            for (int l = 0; l < _blocks.Count; l++)
            {
                layerInputs[l] = currentX;
                currentX = _blocks[l].ForwardResident(gpu, currentX, seqLen, saveActivations: !checkpointActivations, out layerStates[l]);
            }
            swFwd.Stop();
            xFinalLayer = currentX;

            using var finalNormOut = gpu.AllocateResident(seqLen, HiddenDim);
            gpu.LaunchRmsNormBatch(xFinalLayer.DevicePointer, gpu.OutNormWeight, finalNormOut.DevicePointer, seqLen, HiddenDim, _weights.RmsNormEps);

            var swLoss = Stopwatch.StartNew();
            using var dFinalNormOut = gpu.AllocateResident(seqLen, HiddenDim);
            float loss = gpu.ComputeTiledLmHeadLossAndGradientsResident(
                _weights.OutType,
                finalNormOut,
                dFinalNormOut,
                targetTokens,
                HiddenDim,
                VocabSize,
                seqLen,
                FusedCrossEntropyLoss.VocabTileSize,
                lossScale);
            swLoss.Stop();

            using var dXFinalLayer = gpu.AllocateResident(seqLen, HiddenDim);
            gpu.LaunchRmsNormBwd(dFinalNormOut.DevicePointer, xFinalLayer.DevicePointer, gpu.OutNormWeight, dXFinalLayer.DevicePointer, seqLen, HiddenDim, _weights.RmsNormEps);

            var swRecompute = new Stopwatch();
            var swBwd = new Stopwatch();
            currentDX = dXFinalLayer;

            for (int l = _blocks.Count - 1; l >= 0; l--)
            {
                var xIn = layerInputs[l]!;
                ResidentBlockActivationState state;

                if (checkpointActivations)
                {
                    swRecompute.Start();
                    using var recomputedOut = _blocks[l].ForwardResident(gpu, xIn, seqLen, saveActivations: true, out var st);
                    swRecompute.Stop();
                    state = st!;
                }
                else
                {
                    state = layerStates[l]!;
                    layerStates[l] = null;
                }

                try
                {
                    swBwd.Start();
                    var dXPrev = _blocks[l].BackwardResident(gpu, currentDX, state, xIn, seqLen);
                    swBwd.Stop();

                    if (!ReferenceEquals(currentDX, dXFinalLayer))
                    {
                        currentDX.Dispose();
                    }
                    currentDX = dXPrev;
                }
                finally
                {
                    state.Dispose();
                }
            }

            gpu.EndResidentStep(EnumerateAdapters());
            GlacierDiagnostics.LogDebug($"[PROFILE GPU RESIDENT] Fwd: {swFwd.ElapsedMilliseconds} ms | Loss: {swLoss.ElapsedMilliseconds} ms | Recompute: {(checkpointActivations ? swRecompute.ElapsedMilliseconds : 0)} ms | Bwd: {swBwd.ElapsedMilliseconds} ms");
            return loss;
        }
        finally
        {
            for (int l = 0; l < layerStates.Length; l++)
            {
                layerStates[l]?.Dispose();
            }
            for (int l = 0; l < layerInputs.Length; l++)
            {
                layerInputs[l]?.Dispose();
            }
            xFinalLayer?.Dispose();
            currentDX?.Dispose();
        }
    }

    public (long TrainableParams, long TotalParams, double TrainablePercent) GetParameterStats()
    {
        long trainable = 0;
        foreach (var p in _trainableParameters)
        {
            long count = 1;
            for (int d = 0; d < p.Rank; d++) count *= p.Shape[d];
            trainable += count;
        }

        int defaultHeadDim = _weights.HeadDim > 0
            ? _weights.HeadDim
            : (_weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128);

        long totalBase = 0;
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];
            if (lw != null && lw.IsGdn)
            {
                long gdnConvChannels = ((long)_weights.SsmGroupCount * 2 + _weights.SsmTimeStepRank) * _weights.SsmStateSize;
                long ssmInner = _weights.SsmInnerSize > 0
                    ? _weights.SsmInnerSize
                    : (long)_weights.SsmTimeStepRank * _weights.SsmStateSize;
                totalBase += ((long)HiddenDim * gdnConvChannels) +
                             ((long)HiddenDim * ssmInner) +
                             ((long)HiddenDim * _weights.SsmTimeStepRank * 2) +
                             (ssmInner * HiddenDim) +
                             ((long)HiddenDim * FfnDim * 3);
            }
            else
            {
                int lHeadDim = lw != null && lw.HeadDim > 0 ? lw.HeadDim : defaultHeadDim;
                int lHeadsKv = lw != null && lw.HeadsKv > 0 ? lw.HeadsKv : _weights.HeadCountKv;
                long qDim = (long)_weights.HeadCount * lHeadDim;
                long qProjOut = lw != null && lw.HasQGate ? 2 * qDim : qDim;
                long kvDim = (long)lHeadsKv * lHeadDim;
                totalBase += ((long)HiddenDim * qProjOut) +
                             (qDim * HiddenDim) +
                             ((long)HiddenDim * kvDim * 2) +
                             ((long)HiddenDim * FfnDim * 3);
            }
        }

        double pct = totalBase + trainable > 0 ? (double)trainable / (totalBase + trainable) * 100.0 : 0.0;
        return (trainable, totalBase, pct);
    }

    public void SaveAdapter(string outputDirOrPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        string outputDir;
        string binPath;
        if (outputDirOrPath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
        {
            outputDir = Path.GetDirectoryName(outputDirOrPath) ?? ".";
            binPath = outputDirOrPath;
        }
        else
        {
            outputDir = outputDirOrPath;
            binPath = Path.Combine(outputDir, "adapter_model.bin");
        }

        if (!string.IsNullOrEmpty(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        foreach (var p in _trainableParameters)
        {
            LoraBinaryFormat.ValidateFinite(p.AsSpan(), "SaveAdapter trainable parameters");
        }

        string configPath = Path.Combine(outputDir, "adapter_config.json");
        WriteAdapterConfigJson(configPath);

        int enabledTensorCount = 0;
        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            enabledTensorCount += _layerBlocks[l].CountEnabledTensors();
        }

        string binTmp = LoraBinaryFormat.CreateTempSiblingPath(binPath);
        try
        {
            using (var fs = File.Create(binTmp))
            using (var bw = new BinaryWriter(fs))
            {
                // Magic 0x41524F4C ('LORA'), Version 1
                bw.Write(LoraBinaryFormat.Magic);
                bw.Write(LoraBinaryFormat.Version);
                bw.Write(_layerBlocks.Count);
                bw.Write(_config.Rank);
                bw.Write(_config.Alpha);
                bw.Write(enabledTensorCount);

                for (int l = 0; l < _layerBlocks.Count; l++)
                {
                    _layerBlocks[l].WriteAdapterTensors(bw);
                }
                bw.Flush();
                fs.Flush(true);
            }
            File.Move(binTmp, binPath, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(binTmp)) File.Delete(binTmp); }
            catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"[GgufLoraModel] Failed to clean temp file '{binTmp}': {cleanupEx.Message}");
            }
            throw;
        }

        string manifestPath = Path.Combine(outputDir, "adapter_manifest.txt");
        string manifestTmp = LoraBinaryFormat.CreateTempSiblingPath(manifestPath);
        try
        {
            using (var writer = new StreamWriter(manifestTmp))
            {
                writer.WriteLine($"# Glacier.Tune LoRA Adapter Manifest");
                writer.WriteLine($"# Total Layers: {_layerBlocks.Count}");
                writer.WriteLine($"# Total Trainable Tensors: {_trainableParameters.Count}");
                writer.WriteLine($"# Binary Weights: {Path.GetFileName(binPath)}");
                for (int l = 0; l < _layerBlocks.Count; l++)
                {
                    _layerBlocks[l].WriteManifestEntries(writer);
                }
                writer.Flush();
            }
            File.Move(manifestTmp, manifestPath, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(manifestTmp)) File.Delete(manifestTmp); }
            catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"[GgufLoraModel] Failed to clean temp file '{manifestTmp}': {cleanupEx.Message}");
            }
            throw;
        }
    }

    private void WriteAdapterConfigJson(string path)
    {
        string tmp = LoraBinaryFormat.CreateTempSiblingPath(path);
        try
        {
            using (var fs = File.Create(tmp))
            using (var json = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
            {
                json.WriteStartObject();
                json.WriteString("base_model_name_or_path", _gguf.Architecture);
                json.WriteNumber("r", _config.Rank);
                json.WriteNumber("lora_alpha", _config.Alpha);
                json.WriteNumber("lora_dropout", _config.Dropout);
                json.WritePropertyName("target_modules");
                json.WriteStartArray();
                if (_config.TargetModules != null)
                {
                    foreach (string module in _config.TargetModules)
                    {
                        json.WriteStringValue(module);
                    }
                }
                json.WriteEndArray();
                json.WriteString("bias", _config.Bias);
                json.WriteString("peft_type", "LORA");
                json.WriteString("task_type", "CAUSAL_LM");
                json.WriteEndObject();
                json.Flush();
                fs.Flush(true);
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(tmp)) File.Delete(tmp); }
            catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"[GgufLoraModel] Failed to clean temp file '{tmp}': {cleanupEx.Message}");
            }
            throw;
        }
    }

    /// <summary>
    /// Loads saved LoRA adapter weights from an adapter binary file (or directory containing adapter_model.bin)
    /// into all transformer / GDN blocks for training resumption or evaluation.
    /// Supports both native Glacier [inDim, rank] / [rank, outDim] and PEFT [rank, inDim] / [outDim, rank] layouts.
    /// </summary>
    public void LoadAdapter(string adapterFilePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        string binPath;
        if (File.Exists(adapterFilePath))
        {
            binPath = adapterFilePath;
        }
        else if (Directory.Exists(adapterFilePath) && File.Exists(Path.Combine(adapterFilePath, "adapter_model.bin")))
        {
            binPath = Path.Combine(adapterFilePath, "adapter_model.bin");
        }
        else
        {
            throw new FileNotFoundException($"LoRA adapter file not found: {adapterFilePath}");
        }

        using var fs = File.OpenRead(binPath);
        using var br = new BinaryReader(fs);

        LoraBinaryFormat.ValidateAndReadHeader(fs, br, out int layerCount, out int rank, out float alpha, out int tensorCount);
        if (tensorCount > 0 && layerCount > _layerBlocks.Count)
        {
            throw new InvalidDataException($"LoRA adapter layer count ({layerCount}) exceeds model layer count ({_layerBlocks.Count}).");
        }
        if (rank != _config.Rank)
            throw new InvalidDataException($"LoRA adapter rank mismatch: file specifies rank {rank}, model is configured with rank {_config.Rank}.");

        int loadedCount = 0;

        for (int t = 0; t < tensorCount; t++)
        {
            string name = LoraBinaryFormat.ReadBoundedString(fs);
            if (fs.Length - fs.Position < 8)
                throw new InvalidDataException($"Truncated LoRA tensor header for '{name}'.");

            int dim0 = br.ReadInt32();
            int dim1 = br.ReadInt32();
            if (dim0 <= 0 || dim1 <= 0)
                throw new InvalidDataException($"Invalid LoRA tensor dimensions [{dim0}, {dim1}] for '{name}'.");

            long elemCount = (long)dim0 * dim1;
            if (elemCount > int.MaxValue / sizeof(float))
                throw new InvalidDataException($"LoRA tensor '{name}' element count {elemCount} exceeds maximum supported buffer size.");

            long byteCount = elemCount * sizeof(float);
            if (byteCount > fs.Length - fs.Position)
                throw new InvalidDataException($"Truncated LoRA binary payload for '{name}': expected {byteCount} bytes, only {fs.Length - fs.Position} remaining.");

            Tensor<float>? targetTensor = FindTargetTensor(name, dim0, dim1, out bool needsTranspose);
            if (targetTensor != null)
            {
                if (elemCount != targetTensor.ElementCount)
                    throw new InvalidDataException($"LoRA tensor '{name}' element count {elemCount} does not match target {targetTensor.ElementCount}.");

                int countInt = checked((int)elemCount);
                float[] temp = System.Buffers.ArrayPool<float>.Shared.Rent(countInt);
                try
                {
                    Span<float> tempSpan = temp.AsSpan(0, countInt);
                    fs.ReadExactly(MemoryMarshal.AsBytes(tempSpan));
                    LoraBinaryFormat.ValidateFinite(tempSpan, name);

                    var dst = targetTensor.AsSpan();
                    if (!needsTranspose)
                    {
                        tempSpan.CopyTo(dst);
                    }
                    else
                    {
                        for (int r = 0; r < dim0; r++)
                        {
                            for (int c = 0; c < dim1; c++)
                            {
                                dst[c * dim0 + r] = tempSpan[r * dim1 + c];
                            }
                        }
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(temp);
                }
                loadedCount++;
            }
            else
            {
                fs.Seek(byteCount, SeekOrigin.Current);
            }
        }

        if (fs.Position != fs.Length)
        {
            throw new InvalidDataException($"Unexpected {fs.Length - fs.Position} trailing byte(s) after reading {tensorCount} tensor(s) in '{binPath}'.");
        }

        if (tensorCount > 0 && loadedCount == 0)
        {
            throw new InvalidDataException($"LoRA adapter '{binPath}' matched 0/{tensorCount} tensors in the current model.");
        }

        if (rank > 0 && MathF.Abs(alpha - _config.Alpha) > 1e-6f)
        {
            _config.Alpha = alpha;
            foreach (var block in _layerBlocks)
            {
                if (block is TransformerBlock tb) tb.UpdateAlpha(alpha);
                else if (block is GdnBlock gb) gb.UpdateAlpha(alpha);
            }
        }

        GlacierDiagnostics.LogInformation($"[LORA] Successfully loaded {loadedCount}/{tensorCount} adapter tensors across {Math.Min(layerCount, _layerBlocks.Count)} layers (rank={rank}, alpha={alpha}).");
    }

    private Tensor<float>? FindTargetTensor(string name, int dim0, int dim1, out bool needsTranspose)
    {
        needsTranspose = false;
        var parts = name.Split('.');
        if (parts.Length >= 5 && parts[0] == "blk" && int.TryParse(parts[1], out int layerIdx) && layerIdx >= 0 && layerIdx < _layerBlocks.Count)
        {
            var block = _layerBlocks[layerIdx];
            string proj = parts[2];
            string matrix = parts[3];

            LoraAdapter? adapter = block.FindAdapter(proj);
            if (adapter != null && adapter.IsEnabled)
            {
                Tensor<float>? tensor = matrix switch
                {
                    "lora_a" => adapter.AdapterA,
                    "lora_b" => adapter.AdapterB,
                    _ => null
                };

                if (tensor != null)
                {
                    if (tensor.Shape[0] == dim0 && tensor.Shape[1] == dim1)
                    {
                        needsTranspose = false;
                        return tensor;
                    }
                    if (tensor.Shape[0] == dim1 && tensor.Shape[1] == dim0)
                    {
                        needsTranspose = true;
                        return tensor;
                    }
                }
            }
        }

        return null;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _gpuEngine?.Dispose();
            foreach (var b in _layerBlocks) b.Dispose();
            _layerBlocks.Clear();
            _blocks.Clear();
            _gdnBlocks.Clear();
            _trainableParameters.Clear();
            _gguf.Dispose();
        }
    }
}
