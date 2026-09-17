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
using Glacier.Tune.Kernels;

namespace Glacier.Tune.Model;

/// <summary>
/// Full-scale 28-layer Causal Transformer Language Model with PEFT LoRA adapters.
/// Supports full-sequence forward pass, activation checkpointing, fused 152k-vocabulary cross-entropy loss,
/// and reverse-mode backpropagation through all transformer blocks.
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
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[GPU WARNING] Failed to initialize GPU VRAM base engine ({ex.Message}). Falling back to CPU memory-mapped base execution.");
                Console.ResetColor();
                _gpuEngine = null;
            }
        }
    }

    public static GgufLoraModel Load(string ggufPath, LoraConfig? config = null, string? device = "auto")
    {
        var target = ResolveTarget(device);
        var gguf = new GgufFile(ggufPath);
        var weights = new ModelWeights(gguf);
        var tokenizer = new BpeTokenizer(gguf);
        return new GgufLoraModel(gguf, weights, tokenizer, config ?? new LoraConfig(), target);
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
        int headDim = _weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128;

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
    /// Executes the full-sequence forward pass through all 28 layers with activation checkpointing,
    /// computes online fused cross-entropy loss against target labels, and backpropagates through
    /// all 28 layers to accumulate exact gradients into all LoRA adapters.
    /// </summary>
    public unsafe float ForwardLossAndBackward(int[] inputTokens, int[] targetTokens, bool checkpointActivations = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int seqLen = inputTokens.Length;
        if (seqLen < 2) return 0f;

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
            VocabSize);
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

        Console.WriteLine($"\n    [PROFILE] Fwd: {swFwd.ElapsedMilliseconds} ms | Loss: {swLoss.ElapsedMilliseconds} ms | Recompute: {(checkpointActivations ? swRecompute.ElapsedMilliseconds : 0)} ms | Bwd: {swBwd.ElapsedMilliseconds} ms");

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

        int headDim = _weights.HeadCount > 0 ? _weights.EmbeddingLength / _weights.HeadCount : 128;

        long totalBase = (long)_weights.BlockCount * (
            ((long)_weights.EmbeddingLength * _weights.EmbeddingLength * 2) +
            ((long)_weights.EmbeddingLength * (headDim * _weights.HeadCountKv) * 2) +
            ((long)_weights.EmbeddingLength * _weights.FeedForwardLength * 3)
        );

        double pct = (double)trainable / (totalBase + trainable) * 100.0;
        return (trainable, totalBase, pct);
    }

    public void SaveAdapter(string outputDir)
    {
        Directory.CreateDirectory(outputDir);

        var adapterConfig = new Dictionary<string, object>
        {
            ["base_model_name_or_path"] = _gguf.Architecture,
            ["r"] = _config.Rank,
            ["lora_alpha"] = _config.Alpha,
            ["lora_dropout"] = _config.Dropout,
            ["target_modules"] = _config.TargetModules,
            ["bias"] = _config.Bias,
            ["peft_type"] = "LORA",
            ["task_type"] = "CAUSAL_LM"
        };

        string configJson = JsonSerializer.Serialize(adapterConfig, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(outputDir, "adapter_config.json"), configJson);

        string binPath = Path.Combine(outputDir, "adapter_model.bin");
        using (var fs = File.Create(binPath))
        using (var bw = new BinaryWriter(fs))
        {
            // Magic 0x41524F4C ('LORA'), Version 1
            bw.Write((uint)0x41524F4C);
            bw.Write((uint)1);
            bw.Write(_blocks.Count);
            bw.Write(_config.Rank);
            bw.Write(_config.Alpha);
            bw.Write(_blocks.Count * 14); // 7 projections * 2 (A and B)

            for (int l = 0; l < _blocks.Count; l++)
            {
                var b = _blocks[l];
                WriteTensor(bw, $"blk.{l}.attn_q.lora_a.weight", b.QProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.attn_q.lora_b.weight", b.QProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.attn_k.lora_a.weight", b.KProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.attn_k.lora_b.weight", b.KProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.attn_v.lora_a.weight", b.VProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.attn_v.lora_b.weight", b.VProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.attn_output.lora_a.weight", b.OProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.attn_output.lora_b.weight", b.OProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.ffn_gate.lora_a.weight", b.GateProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.ffn_gate.lora_b.weight", b.GateProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.ffn_up.lora_a.weight", b.UpProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.ffn_up.lora_b.weight", b.UpProj.AdapterB);
                WriteTensor(bw, $"blk.{l}.ffn_down.lora_a.weight", b.DownProj.AdapterA);
                WriteTensor(bw, $"blk.{l}.ffn_down.lora_b.weight", b.DownProj.AdapterB);
            }
        }

        string manifestPath = Path.Combine(outputDir, "adapter_manifest.txt");
        using var writer = new StreamWriter(manifestPath);
        writer.WriteLine($"# Glacier.Tune LoRA Adapter Manifest");
        writer.WriteLine($"# Total Layers: {_blocks.Count}");
        writer.WriteLine($"# Total Trainable Parameters: {_trainableParameters.Count}");
        writer.WriteLine($"# Binary Weights: adapter_model.bin");
        for (int l = 0; l < _blocks.Count; l++)
        {
            var b = _blocks[l];
            writer.WriteLine($"blk.{l}.q_proj: A=[{b.QProj.AdapterA.Shape[0]},{b.QProj.AdapterA.Shape[1]}], B=[{b.QProj.AdapterB.Shape[0]},{b.QProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.k_proj: A=[{b.KProj.AdapterA.Shape[0]},{b.KProj.AdapterA.Shape[1]}], B=[{b.KProj.AdapterB.Shape[0]},{b.KProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.v_proj: A=[{b.VProj.AdapterA.Shape[0]},{b.VProj.AdapterA.Shape[1]}], B=[{b.VProj.AdapterB.Shape[0]},{b.VProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.o_proj: A=[{b.OProj.AdapterA.Shape[0]},{b.OProj.AdapterA.Shape[1]}], B=[{b.OProj.AdapterB.Shape[0]},{b.OProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.gate_proj: A=[{b.GateProj.AdapterA.Shape[0]},{b.GateProj.AdapterA.Shape[1]}], B=[{b.GateProj.AdapterB.Shape[0]},{b.GateProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.up_proj: A=[{b.UpProj.AdapterA.Shape[0]},{b.UpProj.AdapterA.Shape[1]}], B=[{b.UpProj.AdapterB.Shape[0]},{b.UpProj.AdapterB.Shape[1]}]");
            writer.WriteLine($"blk.{l}.down_proj: A=[{b.DownProj.AdapterA.Shape[0]},{b.DownProj.AdapterA.Shape[1]}], B=[{b.DownProj.AdapterB.Shape[0]},{b.DownProj.AdapterB.Shape[1]}]");
        }
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
