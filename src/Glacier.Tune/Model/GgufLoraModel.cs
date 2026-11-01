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
    private readonly List<Tensor<float>> _trainableParameters = [];
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public LoraConfig Config => _config;
    public GpuTarget Target => _target;
    public IReadOnlyList<TransformerBlock> Blocks => _blocks;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;
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
            try
            {
                _gpuEngine = new Glacier.Tune.Gpu.GpuLoraEngine(weights);
                for (int l = 0; l < _blocks.Count; l++)
                {
                    _blocks[l].ConnectGpuWeights(_gpuEngine);
                }
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GPU WARNING] Failed to initialize GPU VRAM base engine ({ex.Message}). Falling back to CPU memory-mapped base execution.", ex);
                _gpuEngine = null;
            }
        }
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
        int headDim = _weights.HeadDim > 0
            ? _weights.HeadDim
            : (_weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128);

        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var block = new TransformerBlock(
                layerIndex: l,
                weights: _weights.Layers[l],
                hiddenDim: _weights.EmbeddingLength,
                ffnDim: _weights.FeedForwardLength,
                nHeadsQ: _weights.HeadCount,
                nHeadsKv: _weights.HeadCountKv,
                headDim: headDim,
                ropeFreqBase: _weights.RopeFreqBase,
                rmsNormEps: _weights.RmsNormEps,
                config: _config,
                target: _target);

            _blocks.Add(block);
            _trainableParameters.AddRange(block.TrainableParameters);
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

        for (int l = 0; l < LayerCount; l++)
        {
            var nextX = _blocks[l].Forward(currentX, seqLen, saveActivations: false, out _);
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
        var layerStates = new BlockActivationState?[LayerCount];
        Tensor<float> currentX = x0;

        for (int l = 0; l < LayerCount; l++)
        {
            layerInputs[l] = currentX;
            currentX = _blocks[l].Forward(currentX, seqLen, saveActivations: !checkpointActivations, out layerStates[l]);
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
            BlockActivationState state;

            if (checkpointActivations)
            {
                swRecompute.Start();
                using var recomputedOut = _blocks[l].Forward(xIn, seqLen, saveActivations: true, out var st);
                swRecompute.Stop();
                state = st!;
            }
            else
            {
                state = layerStates[l]!;
            }

            // Backpropagate through layer l, updating adapters and computing dXIn
            swBwd.Start();
            var dXPrev = _blocks[l].Backward(currentDX, state, xIn, seqLen);
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

        int headDim = _weights.HeadDim > 0
            ? _weights.HeadDim
            : (_weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128);
        int qDim = _weights.HeadCount * headDim;
        int kvDim = _weights.HeadCountKv * headDim;

        long totalBase = (long)_weights.BlockCount * (
            ((long)_weights.EmbeddingLength * qDim * 2) +
            ((long)_weights.EmbeddingLength * kvDim * 2) +
            ((long)_weights.EmbeddingLength * _weights.FeedForwardLength * 3)
        );

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
        for (int l = 0; l < _blocks.Count; l++)
        {
            var b = _blocks[l];
            if (b.QProj.IsEnabled) enabledTensorCount += 2;
            if (b.KProj.IsEnabled) enabledTensorCount += 2;
            if (b.VProj.IsEnabled) enabledTensorCount += 2;
            if (b.OProj.IsEnabled) enabledTensorCount += 2;
            if (b.GateProj.IsEnabled) enabledTensorCount += 2;
            if (b.UpProj.IsEnabled) enabledTensorCount += 2;
            if (b.DownProj.IsEnabled) enabledTensorCount += 2;
        }

        using (var fs = File.Create(binPath))
        using (var bw = new BinaryWriter(fs))
        {
            // Magic 0x41524F4C ('LORA'), Version 1
            bw.Write((uint)0x41524F4C);
            bw.Write((uint)1);
            bw.Write(_blocks.Count);
            bw.Write(_config.Rank);
            bw.Write(_config.Alpha);
            bw.Write(enabledTensorCount);

            for (int l = 0; l < _blocks.Count; l++)
            {
                var b = _blocks[l];
                if (b.QProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.attn_q.lora_a.weight", b.QProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.attn_q.lora_b.weight", b.QProj.AdapterB);
                }
                if (b.KProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.attn_k.lora_a.weight", b.KProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.attn_k.lora_b.weight", b.KProj.AdapterB);
                }
                if (b.VProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.attn_v.lora_a.weight", b.VProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.attn_v.lora_b.weight", b.VProj.AdapterB);
                }
                if (b.OProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.attn_output.lora_a.weight", b.OProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.attn_output.lora_b.weight", b.OProj.AdapterB);
                }
                if (b.GateProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.ffn_gate.lora_a.weight", b.GateProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.ffn_gate.lora_b.weight", b.GateProj.AdapterB);
                }
                if (b.UpProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.ffn_up.lora_a.weight", b.UpProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.ffn_up.lora_b.weight", b.UpProj.AdapterB);
                }
                if (b.DownProj.IsEnabled)
                {
                    WriteTensor(bw, $"blk.{l}.ffn_down.lora_a.weight", b.DownProj.AdapterA);
                    WriteTensor(bw, $"blk.{l}.ffn_down.lora_b.weight", b.DownProj.AdapterB);
                }
            }
        }

        string manifestPath = Path.Combine(outputDir, "adapter_manifest.txt");
        using var writer = new StreamWriter(manifestPath);
        writer.WriteLine($"# Glacier.Tune LoRA Adapter Manifest");
        writer.WriteLine($"# Total Layers: {_blocks.Count}");
        writer.WriteLine($"# Total Trainable Tensors: {_trainableParameters.Count}");
        writer.WriteLine($"# Binary Weights: {Path.GetFileName(binPath)}");
        for (int l = 0; l < _blocks.Count; l++)
        {
            var b = _blocks[l];
            if (b.QProj.IsEnabled) writer.WriteLine($"blk.{l}.q_proj: A=[{b.QProj.AdapterA.Shape[0]},{b.QProj.AdapterA.Shape[1]}], B=[{b.QProj.AdapterB.Shape[0]},{b.QProj.AdapterB.Shape[1]}]");
            if (b.KProj.IsEnabled) writer.WriteLine($"blk.{l}.k_proj: A=[{b.KProj.AdapterA.Shape[0]},{b.KProj.AdapterA.Shape[1]}], B=[{b.KProj.AdapterB.Shape[0]},{b.KProj.AdapterB.Shape[1]}]");
            if (b.VProj.IsEnabled) writer.WriteLine($"blk.{l}.v_proj: A=[{b.VProj.AdapterA.Shape[0]},{b.VProj.AdapterA.Shape[1]}], B=[{b.VProj.AdapterB.Shape[0]},{b.VProj.AdapterB.Shape[1]}]");
            if (b.OProj.IsEnabled) writer.WriteLine($"blk.{l}.o_proj: A=[{b.OProj.AdapterA.Shape[0]},{b.OProj.AdapterA.Shape[1]}], B=[{b.OProj.AdapterB.Shape[0]},{b.OProj.AdapterB.Shape[1]}]");
            if (b.GateProj.IsEnabled) writer.WriteLine($"blk.{l}.gate_proj: A=[{b.GateProj.AdapterA.Shape[0]},{b.GateProj.AdapterA.Shape[1]}], B=[{b.GateProj.AdapterB.Shape[0]},{b.GateProj.AdapterB.Shape[1]}]");
            if (b.UpProj.IsEnabled) writer.WriteLine($"blk.{l}.up_proj: A=[{b.UpProj.AdapterA.Shape[0]},{b.UpProj.AdapterA.Shape[1]}], B=[{b.UpProj.AdapterB.Shape[0]},{b.UpProj.AdapterB.Shape[1]}]");
            if (b.DownProj.IsEnabled) writer.WriteLine($"blk.{l}.down_proj: A=[{b.DownProj.AdapterA.Shape[0]},{b.DownProj.AdapterA.Shape[1]}], B=[{b.DownProj.AdapterB.Shape[0]},{b.DownProj.AdapterB.Shape[1]}]");
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
    /// into all transformer blocks for training resumption or evaluation.
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

        GlacierDiagnostics.LogInformation($"[LORA] Successfully loaded {loadedCount}/{tensorCount} adapter tensors across {Math.Min(layerCount, _blocks.Count)} layers (rank={rank}, alpha={alpha}).");
    }

    private Tensor<float>? FindTargetTensor(string name, int dim0, int dim1, out bool needsTranspose)
    {
        needsTranspose = false;
        var parts = name.Split('.');
        if (parts.Length >= 5 && parts[0] == "blk" && int.TryParse(parts[1], out int layerIdx) && layerIdx >= 0 && layerIdx < _blocks.Count)
        {
            var b = _blocks[layerIdx];
            string proj = parts[2];
            string matrix = parts[3];

            LoraAdapter? adapter = proj switch
            {
                "attn_q" or "q_proj" => b.QProj,
                "attn_k" or "k_proj" => b.KProj,
                "attn_v" or "v_proj" => b.VProj,
                "attn_output" or "o_proj" => b.OProj,
                "ffn_gate" or "gate_proj" => b.GateProj,
                "ffn_up" or "up_proj" => b.UpProj,
                "ffn_down" or "down_proj" => b.DownProj,
                _ => null
            };

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

    private static void WriteTensor(BinaryWriter bw, string name, Tensor<float> tensor)
    {
        bw.Write(name);
        bw.Write(tensor.Shape[0]);
        bw.Write(tensor.Shape[1]);
        var span = tensor.AsSpan();
        var byteSpan = MemoryMarshal.AsBytes(span);
        bw.Write(byteSpan);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _gpuEngine?.Dispose();
            foreach (var b in _blocks) b.Dispose();
            _blocks.Clear();
            _trainableParameters.Clear();
            _gguf.Dispose();
        }
    }
}
