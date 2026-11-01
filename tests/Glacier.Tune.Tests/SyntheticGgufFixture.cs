namespace Glacier.Tune.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Model;

/// <summary>
/// High-performance synthetic GGUF model and LoRA adapter fixture.
/// Builds minimal, valid in-memory GGUF headers, tensor descriptors, and model weights in &lt;50ms,
/// decoupling unit and regression tests from multi-gigabyte external model files.
/// </summary>
public sealed class SyntheticGgufFixture : IDisposable
{
    public string DirectoryPath { get; }
    public string GgufPath { get; }
    public string AdapterPath { get; }
    public int LayerCount { get; }
    public int HiddenDim { get; }
    public int FfnDim { get; }
    public int HeadCount { get; }
    public int HeadCountKv { get; }
    public int VocabSize { get; }
    public int Rank { get; }
    public float Alpha { get; }
    public GgufType DefaultWeightType { get; }

    private bool _disposed;

    public SyntheticGgufFixture(
        string directoryPath,
        string ggufPath,
        string adapterPath,
        int layerCount,
        int hiddenDim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize,
        int rank,
        float alpha,
        GgufType defaultWeightType)
    {
        DirectoryPath = directoryPath;
        GgufPath = ggufPath;
        AdapterPath = adapterPath;
        LayerCount = layerCount;
        HiddenDim = hiddenDim;
        FfnDim = ffnDim;
        HeadCount = headCount;
        HeadCountKv = headCountKv;
        VocabSize = vocabSize;
        Rank = rank;
        Alpha = alpha;
        DefaultWeightType = defaultWeightType;
    }

    /// <summary>
    /// Creates a compact, valid synthetic Qwen2 model and LoRA adapter fixture on disk in &lt;50ms.
    /// </summary>
    public static SyntheticGgufFixture CreateTinyQwenModel(
        int layerCount = 2,
        int hiddenDim = 64,
        int ffnDim = 128,
        int nHeadsQ = 2,
        int nHeadsKv = 1,
        int vocabSize = 32,
        int rank = 4,
        float alpha = 8f,
        GgufType weightType = GgufType.F32,
        bool zeroMeanWeights = false)
    {
        string dirPath = Path.Combine(Path.GetTempPath(), "GlacierSynthetic_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dirPath);

        string ggufPath = Path.Combine(dirPath, "model.gguf");
        string adapterPath = Path.Combine(dirPath, "adapter_model.bin");

        BuildSyntheticGguf(ggufPath, layerCount, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, vocabSize, weightType, zeroMeanWeights);
        BuildSyntheticAdapter(adapterPath, dirPath, layerCount, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, rank, alpha);

        return new SyntheticGgufFixture(
            dirPath,
            ggufPath,
            adapterPath,
            layerCount,
            hiddenDim,
            ffnDim,
            nHeadsQ,
            nHeadsKv,
            vocabSize,
            rank,
            alpha,
            weightType);
    }

    /// <summary>
    /// Creates a synthetic model with 256-aligned dimensions suitable for Q4_K super-block quantization testing.
    /// </summary>
    public static SyntheticGgufFixture CreateSuperBlockQwenModel(
        int layerCount = 1,
        int hiddenDim = 256,
        int ffnDim = 512,
        int nHeadsQ = 4,
        int nHeadsKv = 2,
        int vocabSize = 32,
        int rank = 4,
        float alpha = 8f,
        GgufType weightType = GgufType.F32)
    {
        return CreateTinyQwenModel(
            layerCount,
            hiddenDim,
            ffnDim,
            nHeadsQ,
            nHeadsKv,
            vocabSize,
            rank,
            alpha,
            weightType);
    }

    /// <summary>
    /// Loads a GgufLoraModel from this fixture.
    /// </summary>
    public GgufLoraModel CreateLoraModel(LoraConfig? config = null, string device = "cpu")
    {
        var loraConfig = config ?? new LoraConfig { Rank = Rank, Alpha = Alpha };
        return GgufLoraModel.Load(GgufPath, loraConfig, device: device);
    }

    /// <summary>
    /// Creates synthetic training examples for testing forward and backward passes.
    /// </summary>
    public ChatMlDataset CreateSyntheticDataset(int exampleCount = 3, int seqLength = 8)
    {
        var dataset = new ChatMlDataset();
        var rng = new Random(42);

        for (int e = 0; e < exampleCount; e++)
        {
            int[] inputTokens = new int[seqLength];
            int[] labels = new int[seqLength];

            for (int t = 0; t < seqLength; t++)
            {
                inputTokens[t] = rng.Next(1, VocabSize);
            }

            // Next-token prediction: position t predicts position t + 1
            for (int t = 0; t < seqLength - 1; t++)
            {
                labels[t] = inputTokens[t + 1];
            }
            labels[seqLength - 1] = ChatMlDataset.IgnoreIndex;

            // Prompt masking: first half tokens are prompt (masked with -100)
            int promptLen = Math.Max(1, seqLength / 2);
            for (int p = 0; p < promptLen; p++)
            {
                labels[p] = ChatMlDataset.IgnoreIndex;
            }

            var example = new TrainingExample(inputTokens, labels);
            // Add via reflection to private field if needed, or format as ChatConversation
            AddExampleViaReflection(dataset, example);
        }

        return dataset;
    }

    private static void AddExampleViaReflection(ChatMlDataset dataset, TrainingExample example)
    {
        var field = typeof(ChatMlDataset).GetField("_examples", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field?.GetValue(dataset) is List<TrainingExample> list)
        {
            list.Add(example);
        }
    }

    private static unsafe void BuildSyntheticGguf(
        string path,
        int layerCount,
        int hiddenDim,
        int ffnDim,
        int nHeadsQ,
        int nHeadsKv,
        int vocabSize,
        GgufType weightType,
        bool zeroMeanWeights = false)
    {
        int headDim = hiddenDim / nHeadsQ;
        int kvDim = nHeadsKv * headDim;

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        // 1. Header: Magic 'GGUF' (0x46554747), Version 3
        bw.Write((uint)0x46554747);
        bw.Write((uint)3);

        // Define tensors
        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, ulong ByteSize)>();

        ulong CalcByteSize(ulong inDim, ulong outDim, GgufType type)
        {
            return type switch
            {
                GgufType.F32 => inDim * outDim * 4UL,
                GgufType.Q8_0 => (inDim / 32UL) * 34UL * outDim,
                GgufType.Q4_0 => (inDim / 32UL) * 18UL * outDim,
                GgufType.Q4_K => (inDim / 256UL) * 144UL * outDim,
                _ => inDim * outDim * 4UL
            };
        }

        // Global tensors (always F32 for embedding and norms)
        tensors.Add(("token_embd.weight", [(ulong)hiddenDim, (ulong)vocabSize], GgufType.F32, (ulong)(hiddenDim * vocabSize * 4)));
        tensors.Add(("output.weight", [(ulong)hiddenDim, (ulong)vocabSize], GgufType.F32, (ulong)(hiddenDim * vocabSize * 4)));
        tensors.Add(("output_norm.weight", [(ulong)hiddenDim], GgufType.F32, (ulong)(hiddenDim * 4)));

        // Layer tensors
        for (int l = 0; l < layerCount; l++)
        {
            tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)hiddenDim], GgufType.F32, (ulong)(hiddenDim * 4)));
            tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)hiddenDim, (ulong)hiddenDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)hiddenDim, weightType)));
            tensors.Add(($"blk.{l}.attn_k.weight", [(ulong)hiddenDim, (ulong)kvDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)kvDim, weightType)));
            tensors.Add(($"blk.{l}.attn_v.weight", [(ulong)hiddenDim, (ulong)kvDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)kvDim, weightType)));
            tensors.Add(($"blk.{l}.attn_output.weight", [(ulong)hiddenDim, (ulong)hiddenDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)hiddenDim, weightType)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", [(ulong)hiddenDim], GgufType.F32, (ulong)(hiddenDim * 4)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", [(ulong)hiddenDim, (ulong)ffnDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)ffnDim, weightType)));
            tensors.Add(($"blk.{l}.ffn_up.weight", [(ulong)hiddenDim, (ulong)ffnDim], weightType, CalcByteSize((ulong)hiddenDim, (ulong)ffnDim, weightType)));
            tensors.Add(($"blk.{l}.ffn_down.weight", [(ulong)ffnDim, (ulong)hiddenDim], weightType, CalcByteSize((ulong)ffnDim, (ulong)hiddenDim, weightType)));
        }

        bw.Write((ulong)tensors.Count); // Tensor count

        // 2. Metadata KV pairs
        var metadata = new List<Action>();

        void AddString(string key, string val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)8); // GgufValueType.String
            WriteString(bw, val);
        });

        void AddUInt32(string key, uint val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)4); // GgufValueType.UInt32
            bw.Write(val);
        });

        void AddSingle(string key, float val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)6); // GgufValueType.Float32
            bw.Write(val);
        });

        void AddStringArray(string key, string[] items) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)9); // GgufValueType.Array
            bw.Write((uint)8); // String element type
            bw.Write((ulong)items.Length);
            foreach (var s in items) WriteString(bw, s);
        });

        AddString("general.architecture", "qwen2");
        AddUInt32("qwen2.block_count", (uint)layerCount);
        AddUInt32("qwen2.context_length", 512);
        AddUInt32("qwen2.embedding_length", (uint)hiddenDim);
        AddUInt32("qwen2.feed_forward_length", (uint)ffnDim);
        AddUInt32("qwen2.attention.head_count", (uint)nHeadsQ);
        AddUInt32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        AddSingle("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        AddSingle("qwen2.rope.freq_base", 10000.0f);
        AddUInt32("tokenizer.ggml.bos_token_id", 1);
        AddUInt32("tokenizer.ggml.eos_token_id", 2);

        string[] tokens = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++) tokens[i] = $"[tok_{i}]";
        AddStringArray("tokenizer.ggml.tokens", tokens);

        bw.Write((ulong)metadata.Count);
        foreach (var writeMeta in metadata) writeMeta();

        // 3. Tensor Directory entries
        ulong currentOffset = 0;
        foreach (var (name, dims, type, byteSize) in tensors)
        {
            WriteString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (var d in dims) bw.Write(d);
            bw.Write((uint)type);
            bw.Write(currentOffset);

            // Maintain 32-byte alignment for next tensor payload
            ulong remAlign = byteSize % 32;
            ulong paddedSize = remAlign == 0 ? byteSize : byteSize + (32 - remAlign);
            currentOffset += paddedSize;
        }

        // 4. Align header end to 32 bytes before tensor payload
        long pos = fs.Position;
        long rem = pos % 32;
        if (rem != 0)
        {
            int pad = (int)(32 - rem);
            bw.Write(new byte[pad]);
        }

        // 5. Write Tensor Data (deterministic non-zero values for numerical stability)
        float* rowTmp = stackalloc float[256];
        int tensorSeedIdx = 0;
        foreach (var (tName, _, type, byteSize) in tensors)
        {
            tensorSeedIdx++;
            bool isNorm = tName.EndsWith("norm.weight", StringComparison.Ordinal);
            byte[] buf = new byte[byteSize];
            if (type == GgufType.F32)
            {
                fixed (byte* pBuf = buf)
                {
                    float* pFloats = (float*)pBuf;
                    int floatCount = (int)(byteSize / 4);
                    for (int f = 0; f < floatCount; f++)
                    {
                        pFloats[f] = isNorm
                            ? 1.0f
                            : (zeroMeanWeights
                                ? MathF.Sin((f + 1) * 1.6180339f + tensorSeedIdx * 0.73f) * 0.12f
                                : 0.02f + (f % 17) * 0.001f);
                    }
                }
            }
            else if (type == GgufType.Q8_0)
            {
                fixed (byte* pBuf = buf)
                {
                    BlockQ8_0* pBlocks = (BlockQ8_0*)pBuf;
                    int blockCount = (int)(byteSize / (ulong)sizeof(BlockQ8_0));
                    for (int b = 0; b < blockCount; b++)
                    {
                        pBlocks[b].Delta = (Half)0.01f;
                        for (int q = 0; q < 32; q++)
                        {
                            pBlocks[b].Qs[q] = zeroMeanWeights
                                ? (sbyte)Math.Round(MathF.Sin((b * 32 + q + 1) * 1.6180339f + tensorSeedIdx) * 12.0f)
                                : (sbyte)((q % 10) - 5);
                        }
                    }
                }
            }
            else if (type == GgufType.Q4_0)
            {
                fixed (byte* pBuf = buf)
                {
                    BlockQ4_0* pBlocks = (BlockQ4_0*)pBuf;
                    int blockCount = (int)(byteSize / (ulong)sizeof(BlockQ4_0));
                    for (int b = 0; b < blockCount; b++)
                    {
                        pBlocks[b].Delta = (Half)0.02f;
                        for (int q = 0; q < 16; q++)
                        {
                            if (zeroMeanWeights)
                            {
                                int q0 = Math.Clamp((int)MathF.Round(MathF.Sin((b * 32 + q + 1) * 1.6180339f + tensorSeedIdx) * 5.0f) + 8, 0, 15);
                                int q1 = Math.Clamp((int)MathF.Round(MathF.Sin((b * 32 + q + 17) * 1.6180339f + tensorSeedIdx) * 5.0f) + 8, 0, 15);
                                pBlocks[b].Qs[q] = (byte)(q0 | (q1 << 4));
                            }
                            else
                            {
                                pBlocks[b].Qs[q] = 0x88; // zero value (8 - 8 = 0)
                            }
                        }
                    }
                }
            }
            else if (type == GgufType.Q4_K)
            {
                fixed (byte* pBuf = buf)
                {
                    BlockQ4_K* pBlocks = (BlockQ4_K*)pBuf;
                    int blockCount = (int)(byteSize / (ulong)sizeof(BlockQ4_K));
                    if (zeroMeanWeights)
                    {
                        for (int b = 0; b < blockCount; b++)
                        {
                            for (int i = 0; i < 256; i++)
                            {
                                rowTmp[i] = MathF.Sin((b * 256 + i + 1) * 1.6180339f + tensorSeedIdx) * 0.12f;
                            }
                            Glacier.Tune.Export.GgufMerger.QuantizeRowQ4_K(rowTmp, pBlocks + b, 256);
                        }
                    }
                    else
                    {
                        for (int b = 0; b < blockCount; b++)
                        {
                            pBlocks[b].Delta = (Half)0.01f;
                            pBlocks[b].DeltaMin = (Half)0.0f;
                            for (int s = 0; s < 12; s++) pBlocks[b].Scales[s] = 1;
                            for (int q = 0; q < 128; q++) pBlocks[b].Qs[q] = 0x11;
                        }
                    }
                }
            }
            else
            {
                // Generic fill
                Array.Fill(buf, (byte)0x3D);
            }

            bw.Write(buf);

            // Align to 32 bytes
            ulong remPad = byteSize % 32;
            if (remPad != 0)
            {
                bw.Write(new byte[(int)(32 - remPad)]);
            }
        }

        bw.Flush();
    }

    private static void BuildSyntheticAdapter(
        string adapterBinPath,
        string dirPath,
        int layerCount,
        int hiddenDim,
        int ffnDim,
        int nHeadsQ,
        int nHeadsKv,
        int rank,
        float alpha)
    {
        int headDim = hiddenDim / nHeadsQ;
        int kvDim = nHeadsKv * headDim;

        // 1. Write adapter_config.json
        var config = new Dictionary<string, object>
        {
            ["base_model_name_or_path"] = "qwen2",
            ["r"] = rank,
            ["lora_alpha"] = alpha,
            ["lora_dropout"] = 0.0f,
            ["target_modules"] = new[] { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" },
            ["bias"] = "none",
            ["peft_type"] = "LORA",
            ["task_type"] = "CAUSAL_LM"
        };
        File.WriteAllText(Path.Combine(dirPath, "adapter_config.json"), JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        // 2. Write adapter_model.bin: Magic 'LORA' (0x41524F4C), Version 1
        using var fs = File.Create(adapterBinPath);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write((uint)0x41524F4C);
        bw.Write((uint)1);
        bw.Write(layerCount);
        bw.Write(rank);
        bw.Write(alpha);
        bw.Write(layerCount * 14); // 7 projections * 2 matrices

        void WriteAdapterMatrix(string name, int dim0, int dim1)
        {
            bw.Write(name);
            bw.Write(dim0);
            bw.Write(dim1);

            int count = dim0 * dim1;
            byte[] bytes = new byte[count * 4];
            unsafe
            {
                fixed (byte* pB = bytes)
                {
                    float* pF = (float*)pB;
                    for (int i = 0; i < count; i++)
                    {
                        pF[i] = 0.01f + (i % 7) * 0.002f;
                    }
                }
            }
            bw.Write(bytes);
        }

        for (int l = 0; l < layerCount; l++)
        {
            WriteAdapterMatrix($"blk.{l}.attn_q.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.attn_q.lora_b.weight", rank, hiddenDim);

            WriteAdapterMatrix($"blk.{l}.attn_k.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.attn_k.lora_b.weight", rank, kvDim);

            WriteAdapterMatrix($"blk.{l}.attn_v.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.attn_v.lora_b.weight", rank, kvDim);

            WriteAdapterMatrix($"blk.{l}.attn_output.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.attn_output.lora_b.weight", rank, hiddenDim);

            WriteAdapterMatrix($"blk.{l}.ffn_gate.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.ffn_gate.lora_b.weight", rank, ffnDim);

            WriteAdapterMatrix($"blk.{l}.ffn_up.lora_a.weight", hiddenDim, rank);
            WriteAdapterMatrix($"blk.{l}.ffn_up.lora_b.weight", rank, ffnDim);

            WriteAdapterMatrix($"blk.{l}.ffn_down.lora_a.weight", ffnDim, rank);
            WriteAdapterMatrix($"blk.{l}.ffn_down.lora_b.weight", rank, hiddenDim);
        }

        bw.Flush();
    }

    private static void WriteString(BinaryWriter bw, string str)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(str);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        try
        {
            if (Directory.Exists(DirectoryPath))
            {
                Directory.Delete(DirectoryPath, recursive: true);
            }
        }
        catch
        {
            // Best effort cleanup in temporary directory
        }
    }
}
