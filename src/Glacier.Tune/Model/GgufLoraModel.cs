using System;
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
        _target = target;

        InitializeBlocks();

        if ((target is GpuTarget.Nvidia or GpuTarget.NvidiaTensorCore or GpuTarget.Auto) && Glacier.Inference.Gpu.GpuContext.IsSupported)
        {
            if (SupportsGpuEngine(weights))
            {
                try
                {
                    _gpuEngine = new Glacier.Tune.Gpu.GpuLoraEngine(weights);
                    for (int l = 0; l < _layerBlocks.Count; l++)
                    {
                        _layerBlocks[l].ConnectGpuWeights(_gpuEngine);
                    }
                }
                catch (Exception ex)
                {
                    GlacierDiagnostics.LogWarning($"[GPU WARNING] Failed to initialize GPU VRAM base engine ({ex.Message}). Falling back to CPU memory-mapped base execution.", ex);
                    _gpuEngine = null;
                }
            }
            else
            {
                GlacierDiagnostics.LogInformation("[INFO] Architecture uses hybrid GDN / Q-Gate / QK-Norm / partial-RoPE / fused-QKV features; executing in pure-C# CPU mode.");
            }
        }
    }

    private static bool SupportsGpuEngine(ModelWeights weights)
    {
        if (weights.IsHybridSsm || weights.IsMoe || weights.IsMla || weights.IsDsa || weights.HasRopeFreqs)
        {
            return false;
        }

        if (weights.RopeDimensionCount > 0 && weights.HeadDim > 0 && weights.RopeDimensionCount < weights.HeadDim)
        {
            return false;
        }

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw == null ||
                lw.IsGdn ||
                lw.HasQGate ||
                lw.QWeight == null ||
                lw.KWeight == null ||
                lw.VWeight == null ||
                lw.AttnQNormWeight != null ||
                lw.AttnKNormWeight != null ||
                lw.AttnSinksWeight != null ||
                lw.HasFusedGateUp ||
                lw.FfnGateWeight == null ||
                lw.FfnUpWeight == null ||
                lw.FfnDownWeight == null ||
                (lw.HeadDim > 0 && weights.HeadDim > 0 && lw.HeadDim != weights.HeadDim) ||
                (lw.HeadsKv > 0 && lw.HeadsKv != weights.HeadCountKv) ||
                lw.HasHyperConnections ||
                lw.IsQsa ||
                lw.HasPle)
            {
                return false;
            }
        }

        return true;
    }

    public static GgufLoraModel Load(string ggufPath, LoraConfig? config = null, string? device = "auto")
    {
        var target = ResolveTarget(device);
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
        if (string.IsNullOrEmpty(device) || device.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return GpuAccelerator.HasNvidiaGpu ? GpuTarget.Nvidia : GpuTarget.Auto;
        }

        return device.ToLowerInvariant() switch
        {
            "cuda" or "gpu" or "nvidia" => GpuTarget.Nvidia,
            "tensorcore" => GpuTarget.NvidiaTensorCore,
            "d3d12" or "directml" => GpuTarget.Direct3D12,
            "vulkan" => GpuTarget.Vulkan,
            "cpu" => GpuTarget.Cpu,
            _ => GpuTarget.Auto
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

    /// <summary>
    /// Executes the full-sequence forward pass through all layers (optionally with activation checkpointing),
    /// computes online fused cross-entropy loss against target labels, and backpropagates through
    /// all layers to accumulate exact gradients into all enabled LoRA adapters.
    /// </summary>
    public unsafe float ForwardLossAndBackward(int[] inputTokens, int[] targetTokens, bool checkpointActivations = false)
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

        WriteAdapterConfigJson(Path.Combine(outputDir, "adapter_config.json"));

        int enabledTensorCount = 0;
        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            enabledTensorCount += _layerBlocks[l].CountEnabledTensors();
        }

        using (var fs = File.Create(binPath))
        using (var bw = new BinaryWriter(fs))
        {
            // Magic 0x41524F4C ('LORA'), Version 1
            bw.Write((uint)0x41524F4C);
            bw.Write((uint)1);
            bw.Write(_layerBlocks.Count);
            bw.Write(_config.Rank);
            bw.Write(_config.Alpha);
            bw.Write(enabledTensorCount);

            for (int l = 0; l < _layerBlocks.Count; l++)
            {
                _layerBlocks[l].WriteAdapterTensors(bw);
            }
        }

        string manifestPath = Path.Combine(outputDir, "adapter_manifest.txt");
        using var writer = new StreamWriter(manifestPath);
        writer.WriteLine($"# Glacier.Tune LoRA Adapter Manifest");
        writer.WriteLine($"# Total Layers: {_layerBlocks.Count}");
        writer.WriteLine($"# Total Trainable Tensors: {_trainableParameters.Count}");
        writer.WriteLine($"# Binary Weights: {Path.GetFileName(binPath)}");
        for (int l = 0; l < _layerBlocks.Count; l++)
        {
            _layerBlocks[l].WriteManifestEntries(writer);
        }
    }

    private void WriteAdapterConfigJson(string path)
    {
        using var fs = File.Create(path);
        using var json = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true });
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

        uint magic = br.ReadUInt32();
        if (magic != 0x41524F4C)
            throw new InvalidDataException($"Invalid LoRA magic header: 0x{magic:X8} (expected 0x41524F4C 'LORA')");

        uint version = br.ReadUInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported LoRA adapter version: {version}");

        int layerCount = br.ReadInt32();
        int rank = br.ReadInt32();
        float alpha = br.ReadSingle();
        int tensorCount = br.ReadInt32();
        if (tensorCount < 0)
            throw new InvalidDataException($"Invalid negative LoRA tensor count: {tensorCount}");
        if (rank != _config.Rank)
            throw new InvalidDataException($"LoRA adapter rank mismatch: file specifies rank {rank}, model is configured with rank {_config.Rank}.");
        if (rank > 0 && (!float.IsFinite(alpha) || alpha <= 0f))
            throw new InvalidDataException($"Invalid LoRA adapter alpha value: {alpha}.");

        int loadedCount = 0;

        for (int t = 0; t < tensorCount; t++)
        {
            string name = br.ReadString();
            int dim0 = br.ReadInt32();
            int dim1 = br.ReadInt32();
            if (dim0 <= 0 || dim1 <= 0)
                throw new InvalidDataException($"Invalid LoRA tensor dimensions [{dim0}, {dim1}] for '{name}'.");

            long byteCount = (long)dim0 * dim1 * sizeof(float);
            if (byteCount > fs.Length - fs.Position)
                throw new InvalidDataException($"Truncated LoRA binary payload for '{name}': expected {byteCount} bytes, only {fs.Length - fs.Position} remaining.");

            Tensor<float>? targetTensor = FindTargetTensor(name, dim0, dim1, out bool needsTranspose);
            if (targetTensor != null)
            {
                if (!needsTranspose)
                {
                    var span = targetTensor.AsSpan();
                    fs.ReadExactly(MemoryMarshal.AsBytes(span));
                }
                else
                {
                    float[] temp = new float[dim0 * dim1];
                    fs.ReadExactly(MemoryMarshal.AsBytes(temp.AsSpan()));
                    var dst = targetTensor.AsSpan();
                    for (int r = 0; r < dim0; r++)
                    {
                        for (int c = 0; c < dim1; c++)
                        {
                            dst[c * dim0 + r] = temp[r * dim1 + c];
                        }
                    }
                }
                loadedCount++;
            }
            else
            {
                fs.Seek(byteCount, SeekOrigin.Current);
            }
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
