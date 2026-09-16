using System;
using System.Collections.Generic;
using System.IO;
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
    private readonly List<TransformerBlock> _blocks = [];
    private readonly List<Tensor<float>> _trainableParameters = [];
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public LoraConfig Config => _config;
    public IReadOnlyList<TransformerBlock> Blocks => _blocks;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;

    public int LayerCount => _weights.BlockCount;
    public int HiddenDim => _weights.EmbeddingLength;
    public int FfnDim => _weights.FeedForwardLength;
    public int VocabSize => _weights.VocabSize;

    private GgufLoraModel(GgufFile gguf, ModelWeights weights, BpeTokenizer tokenizer, LoraConfig config)
    {
        _gguf = gguf;
        _weights = weights;
        _tokenizer = tokenizer;
        _config = config;

        InitializeBlocks();
    }

    public static GgufLoraModel Load(string ggufPath, LoraConfig? config = null)
    {
        var gguf = new GgufFile(ggufPath);
        var weights = new ModelWeights(gguf);
        var tokenizer = new BpeTokenizer(gguf);
        return new GgufLoraModel(gguf, weights, tokenizer, config ?? new LoraConfig());
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
                config: _config);

            _blocks.Add(block);
            _trainableParameters.AddRange(block.TrainableParameters);
        }
    }

    /// <summary>
    /// Executes the full-sequence forward pass through all 28 layers with activation checkpointing,
    /// computes online fused cross-entropy loss against target labels, and backpropagates through
    /// all 28 layers to accumulate exact gradients into all LoRA adapters.
    /// </summary>
    public float ForwardLossAndBackward(int[] inputTokens, int[] targetTokens)
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

        // 2. Forward pass through all layers with Activation Checkpointing
        // Only layer inputs X_l are preserved (205 MB for 28 layers at seq 512)
        var layerInputs = new Tensor<float>[LayerCount];
        Tensor<float> currentX = x0;

        for (int l = 0; l < LayerCount; l++)
        {
            layerInputs[l] = currentX;
            currentX = _blocks[l].Forward(currentX, seqLen, saveActivations: false, out _);
        }

        Tensor<float> xFinalLayer = currentX;

        // 3. Final Output RMSNorm: normOut = RMSNorm(xFinalLayer, out_norm)
        using var finalNormOut = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNorm(xFinalLayer, wFinalNorm, finalNormOut, _weights.RmsNormEps);
        }

        // 4. Fused Cross-Entropy Loss & Output Gradient (dXFinalNorm)
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

        // 5. Final Output RMSNorm Backward -> dXFinalLayer
        using var dXFinalLayer = new Tensor<float>(seqLen, HiddenDim);
        using (var wFinalNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.OutNormWeight, HiddenDim), [HiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dFinalNormOut, xFinalLayer, wFinalNorm, dXFinalLayer, null, _weights.RmsNormEps);
        }

        // 6. Reverse-Mode Backpropagation with Activation Checkpointing
        // Recomputes layer l activations on the fly and immediately disposes them
        Tensor<float> currentDX = dXFinalLayer;

        for (int l = LayerCount - 1; l >= 0; l--)
        {
            var xIn = layerInputs[l];

            // Recompute layer l forward pass with activations saved
            using var recomputedOut = _blocks[l].Forward(xIn, seqLen, saveActivations: true, out var state);

            // Backpropagate through layer l, updating adapters and computing dXIn
            var dXPrev = _blocks[l].Backward(currentDX, state!, xIn, seqLen);
            state!.Dispose();

            if (!ReferenceEquals(currentDX, dXFinalLayer))
            {
                currentDX.Dispose();
            }
            currentDX = dXPrev;
        }

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

        string manifestPath = Path.Combine(outputDir, "adapter_manifest.txt");
        using var writer = new StreamWriter(manifestPath);
        writer.WriteLine($"# Glacier.Tune LoRA Adapter Manifest");
        writer.WriteLine($"# Total Layers: {_blocks.Count}");
        writer.WriteLine($"# Total Trainable Parameters: {_trainableParameters.Count}");
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

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            foreach (var b in _blocks) b.Dispose();
            _blocks.Clear();
            _trainableParameters.Clear();
            _gguf.Dispose();
        }
    }
}
