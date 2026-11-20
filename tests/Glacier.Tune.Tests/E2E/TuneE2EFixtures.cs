using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Tokenizer;
using Glacier.Tune.Data;

namespace Glacier.Tune.Tests.E2E;

/// <summary>
/// Isolated per-test temporary directory rooted under <see cref="Path.GetTempPath()"/>.
/// Guarantees cleanup on disposal.
/// </summary>
public sealed class TempDirectoryScope : IDisposable
{
    public string Path { get; }

    public TempDirectoryScope(string prefix = "glacier_tune_e2e_")
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"{prefix}{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path);
    }

    public string FilePath(string relativeName) => System.IO.Path.Combine(Path, relativeName);

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
        catch
        {
            // Best-effort cleanup if OS holds a transient handle.
        }
    }
}

/// <summary>
/// Configuration for synthetic GGUF v3 model generation (FX-5.1).
/// </summary>
public sealed class SyntheticModelSpec
{
    public string Architecture { get; init; } = "qwen35";
    public int BlockCount { get; init; } = 2;
    public int FullAttentionInterval { get; init; } = 2;
    public int HiddenDim { get; init; } = 32;
    public int FfnDim { get; init; } = 64;
    public int HeadCount { get; init; } = 2;
    public int HeadCountKv { get; init; } = 2;
    public int HeadDim { get; init; } = 16;
    public int RopeDim { get; init; } = 8;
    public float RopeFreqBase { get; init; } = 10000.0f;
    public int ContextLength { get; init; } = 128;
    public int ConvKernel { get; init; } = 4;
    public int StateSize { get; init; } = 8;
    public int GroupCount { get; init; } = 2;
    public int SsmHeads { get; init; } = 4;
    public bool HasQGate { get; init; } = true;
    public bool HasQkNorm { get; init; } = true;
    public bool IncludeSsmConv1d { get; init; } = true;
    public bool CorruptSsmConv1dShape { get; init; } = false;
    public bool FuseGdnTensors { get; init; } = false;
    public int Seed { get; init; } = 1337;
    public float WeightScale { get; init; } = 0.08f;
}

/// <summary>
/// Self-contained synthetic GGUF v3 binary builder, deterministic tokenizer/ChatML dataset factory,
/// and eval.jsonl generator for Glacier.Tune E2E tests (FX-5.1, FX-5.2, FX-5.3).
/// </summary>
public static class TuneE2EFixtures
{
    public const int EosTokenId = 256;
    public const int ImStartTokenId = 257;
    public const int ImEndTokenId = 258;
    public const int VocabSize = 264;

    private static readonly string[] ByteTokens = BuildGpt2ByteTokens();

    private static string[] BuildGpt2ByteTokens()
    {
        var map = new Dictionary<int, char>(256);
        for (int b = 33; b <= 126; b++) map[b] = (char)b;
        for (int b = 161; b <= 172; b++) map[b] = (char)b;
        for (int b = 174; b <= 255; b++) map[b] = (char)b;

        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!map.ContainsKey(b))
            {
                map[b] = (char)(256 + extra++);
            }
        }

        var tokens = new string[VocabSize];
        for (int b = 0; b < 256; b++)
        {
            tokens[b] = map[b].ToString();
        }

        tokens[256] = "<|endoftext|>";
        tokens[257] = "<|im_start|>";
        tokens[258] = "<|im_end|>";
        tokens[259] = "<|pad|>";
        tokens[260] = "<|extra_0|>";
        tokens[261] = "<|extra_1|>";
        tokens[262] = "<|extra_2|>";
        tokens[263] = "<|extra_3|>";
        return tokens;
    }

    public static void WriteSyntheticGguf(string path, SyntheticModelSpec? spec = null)
    {
        spec ??= new SyntheticModelSpec();
        var rng = new Random(spec.Seed);
        string arch = spec.Architecture;
        bool isHybrid = arch is "qwen35" or "qwen4exp" or "qwen38";

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, arch),
            ("general.name", GgufValueType.String, $"synthetic-{arch}"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ($"{arch}.block_count", GgufValueType.Uint32, (uint)spec.BlockCount),
            ($"{arch}.embedding_length", GgufValueType.Uint32, (uint)spec.HiddenDim),
            ($"{arch}.feed_forward_length", GgufValueType.Uint32, (uint)spec.FfnDim),
            ($"{arch}.context_length", GgufValueType.Uint32, (uint)spec.ContextLength),
            ($"{arch}.attention.head_count", GgufValueType.Uint32, (uint)spec.HeadCount),
            ($"{arch}.attention.head_count_kv", GgufValueType.Uint32, (uint)spec.HeadCountKv),
            ($"{arch}.attention.key_length", GgufValueType.Uint32, (uint)spec.HeadDim),
            ($"{arch}.attention.value_length", GgufValueType.Uint32, (uint)spec.HeadDim),
            ($"{arch}.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ($"{arch}.rope.dimension_count", GgufValueType.Uint32, (uint)spec.RopeDim),
            ($"{arch}.rope.freq_base", GgufValueType.Float32, spec.RopeFreqBase),
        };

        if (isHybrid)
        {
            int ssmInner = spec.SsmHeads * spec.StateSize;
            metadata.Add(($"{arch}.full_attention_interval", GgufValueType.Uint32, (uint)spec.FullAttentionInterval));
            metadata.Add(($"{arch}.ssm.conv_kernel", GgufValueType.Uint32, (uint)spec.ConvKernel));
            metadata.Add(($"{arch}.ssm.state_size", GgufValueType.Uint32, (uint)spec.StateSize));
            metadata.Add(($"{arch}.ssm.group_count", GgufValueType.Uint32, (uint)spec.GroupCount));
            metadata.Add(($"{arch}.ssm.time_step_rank", GgufValueType.Uint32, (uint)spec.SsmHeads));
            metadata.Add(($"{arch}.ssm.inner_size", GgufValueType.Uint32, (uint)ssmInner));
        }

        var tokenTypes = new int[VocabSize];
        for (int i = 0; i < 256; i++) tokenTypes[i] = 1;
        for (int i = 256; i < VocabSize; i++) tokenTypes[i] = 3;

        metadata.Add(("tokenizer.ggml.model", GgufValueType.String, "gpt2"));
        metadata.Add(("tokenizer.ggml.tokens", GgufValueType.Array, ByteTokens));
        metadata.Add(("tokenizer.ggml.token_type", GgufValueType.Array, tokenTypes));
        metadata.Add(("tokenizer.ggml.merges", GgufValueType.Array, Array.Empty<string>()));
        metadata.Add(("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)EosTokenId));
        metadata.Add(("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)EosTokenId));
        metadata.Add(("tokenizer.ggml.padding_token_id", GgufValueType.Uint32, (uint)EosTokenId));

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>();

        // Embeddings and output head: dims[0]=inDim (HiddenDim), dims[1]=outDim (VocabSize)
        tensors.Add(("token_embd.weight", [(ulong)spec.HiddenDim, (ulong)VocabSize], CreateRandomWeights(rng, spec.HiddenDim * VocabSize, spec.WeightScale)));
        tensors.Add(("output_norm.weight", [(ulong)spec.HiddenDim], CreateOnes(spec.HiddenDim)));
        tensors.Add(("output.weight", [(ulong)spec.HiddenDim, (ulong)VocabSize], CreateRandomWeights(rng, spec.HiddenDim * VocabSize, spec.WeightScale)));

        int qDim = spec.HeadCount * spec.HeadDim;
        int kvDim = spec.HeadCountKv * spec.HeadDim;
        int ssmInnerSize = spec.SsmHeads * spec.StateSize;
        int gdnConvChannels = (spec.GroupCount * 2 + spec.SsmHeads) * spec.StateSize;

        for (int l = 0; l < spec.BlockCount; l++)
        {
            bool isGdnLayer = isHybrid && ((l + 1) % spec.FullAttentionInterval != 0);
            tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)spec.HiddenDim], CreateOnes(spec.HiddenDim)));

            if (isGdnLayer)
            {
                if (spec.FuseGdnTensors)
                {
                    float[] qkvData = CreateRandomWeights(rng, spec.HiddenDim * gdnConvChannels, spec.WeightScale);
                    float[] gateData = CreateRandomWeights(rng, spec.HiddenDim * ssmInnerSize, spec.WeightScale);
                    float[] qkvGateData = new float[qkvData.Length + gateData.Length];
                    Array.Copy(qkvData, 0, qkvGateData, 0, qkvData.Length);
                    Array.Copy(gateData, 0, qkvGateData, qkvData.Length, gateData.Length);
                    tensors.Add(($"blk.{l}.attn_qkv_gate.weight", [(ulong)spec.HiddenDim, (ulong)(gdnConvChannels + ssmInnerSize)], qkvGateData));

                    float[] alphaData = CreateRandomWeights(rng, spec.HiddenDim * spec.SsmHeads, spec.WeightScale);
                    float[] betaData = CreateRandomWeights(rng, spec.HiddenDim * spec.SsmHeads, spec.WeightScale);
                    float[] betaAlphaData = new float[betaData.Length + alphaData.Length];
                    Array.Copy(betaData, 0, betaAlphaData, 0, betaData.Length);
                    Array.Copy(alphaData, 0, betaAlphaData, betaData.Length, alphaData.Length);
                    tensors.Add(($"blk.{l}.ssm_beta_alpha.weight", [(ulong)spec.HiddenDim, (ulong)(spec.SsmHeads * 2)], betaAlphaData));
                }
                else
                {
                    tensors.Add(($"blk.{l}.attn_qkv.weight", [(ulong)spec.HiddenDim, (ulong)gdnConvChannels], CreateRandomWeights(rng, spec.HiddenDim * gdnConvChannels, spec.WeightScale)));
                    tensors.Add(($"blk.{l}.attn_gate.weight", [(ulong)spec.HiddenDim, (ulong)ssmInnerSize], CreateRandomWeights(rng, spec.HiddenDim * ssmInnerSize, spec.WeightScale)));
                    tensors.Add(($"blk.{l}.ssm_alpha.weight", [(ulong)spec.HiddenDim, (ulong)spec.SsmHeads], CreateRandomWeights(rng, spec.HiddenDim * spec.SsmHeads, spec.WeightScale)));
                    tensors.Add(($"blk.{l}.ssm_beta.weight", [(ulong)spec.HiddenDim, (ulong)spec.SsmHeads], CreateRandomWeights(rng, spec.HiddenDim * spec.SsmHeads, spec.WeightScale)));
                }

                if (spec.IncludeSsmConv1d)
                {
                    int convCols = spec.CorruptSsmConv1dShape ? 1 : gdnConvChannels;
                    tensors.Add(($"blk.{l}.ssm_conv1d.weight", [(ulong)spec.ConvKernel, (ulong)convCols], CreateRandomWeights(rng, spec.ConvKernel * convCols, 0.2f)));
                }

                // ssm_a stores negative log-decay values per head
                var ssmA = new float[spec.SsmHeads];
                for (int h = 0; h < spec.SsmHeads; h++) ssmA[h] = -0.5f - 0.1f * h;
                tensors.Add(($"blk.{l}.ssm_a", [(ulong)spec.SsmHeads], ssmA));
                tensors.Add(($"blk.{l}.ssm_dt.bias", [(ulong)spec.SsmHeads], new float[spec.SsmHeads]));
                tensors.Add(($"blk.{l}.ssm_norm.weight", [(ulong)spec.StateSize], CreateOnes(spec.StateSize)));
                tensors.Add(($"blk.{l}.ssm_out.weight", [(ulong)ssmInnerSize, (ulong)spec.HiddenDim], CreateRandomWeights(rng, ssmInnerSize * spec.HiddenDim, spec.WeightScale)));
            }
            else
            {
                int qOutDim = spec.HasQGate ? 2 * qDim : qDim;
                tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)spec.HiddenDim, (ulong)qOutDim], CreateRandomWeights(rng, spec.HiddenDim * qOutDim, spec.WeightScale)));
                tensors.Add(($"blk.{l}.attn_k.weight", [(ulong)spec.HiddenDim, (ulong)kvDim], CreateRandomWeights(rng, spec.HiddenDim * kvDim, spec.WeightScale)));
                tensors.Add(($"blk.{l}.attn_v.weight", [(ulong)spec.HiddenDim, (ulong)kvDim], CreateRandomWeights(rng, spec.HiddenDim * kvDim, spec.WeightScale)));
                tensors.Add(($"blk.{l}.attn_output.weight", [(ulong)qDim, (ulong)spec.HiddenDim], CreateRandomWeights(rng, qDim * spec.HiddenDim, spec.WeightScale)));

                if (spec.HasQkNorm)
                {
                    tensors.Add(($"blk.{l}.attn_q_norm.weight", [(ulong)spec.HeadDim], CreateOnes(spec.HeadDim)));
                    tensors.Add(($"blk.{l}.attn_k_norm.weight", [(ulong)spec.HeadDim], CreateOnes(spec.HeadDim)));
                }
            }

            tensors.Add(($"blk.{l}.ffn_norm.weight", [(ulong)spec.HiddenDim], CreateOnes(spec.HiddenDim)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", [(ulong)spec.HiddenDim, (ulong)spec.FfnDim], CreateRandomWeights(rng, spec.HiddenDim * spec.FfnDim, spec.WeightScale)));
            tensors.Add(($"blk.{l}.ffn_up.weight", [(ulong)spec.HiddenDim, (ulong)spec.FfnDim], CreateRandomWeights(rng, spec.HiddenDim * spec.FfnDim, spec.WeightScale)));
            tensors.Add(($"blk.{l}.ffn_down.weight", [(ulong)spec.FfnDim, (ulong)spec.HiddenDim], CreateRandomWeights(rng, spec.FfnDim * spec.HiddenDim, spec.WeightScale)));
        }

        WriteGgufBinary(path, metadata, tensors, alignment: 32);
    }

    public static ChatMlDataset CreateChatMlDataset(
        BpeTokenizer tokenizer,
        IEnumerable<(string User, string Assistant)> pairs,
        int maxSeqLength = 128)
    {
        var sb = new StringBuilder();
        foreach (var (user, assistant) in pairs)
        {
            string u = System.Text.Json.JsonSerializer.Serialize(user);
            string a = System.Text.Json.JsonSerializer.Serialize(assistant);
            sb.Append("{\"messages\":[{\"role\":\"user\",\"content\":")
              .Append(u)
              .Append("},{\"role\":\"assistant\",\"content\":")
              .Append(a)
              .AppendLine("}]}");
        }

        using var scope = new TempDirectoryScope("glacier_ds_");
        string path = scope.FilePath("dataset.jsonl");
        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
        return ChatMlDataset.FromFile(path, tokenizer, maxSeqLength);
    }

    public static ChatMlDataset CreateGeneralizationTrainDataset(BpeTokenizer tokenizer)
    {
        return CreateChatMlDataset(tokenizer,
        [
            ("tag: alpha", "OK"),
            ("tag: beta", "OK"),
            ("tag: gamma", "OK"),
            ("tag: delta", "OK"),
            ("tag: alpha", "OK"),
            ("tag: beta", "OK"),
        ]);
    }

    public static ChatMlDataset CreateGeneralizationEvalDataset(BpeTokenizer tokenizer)
    {
        return CreateChatMlDataset(tokenizer,
        [
            ("tag: epsilon", "OK"),
            ("tag: zeta", "OK"),
        ]);
    }

    public static string WriteEvalJsonl(string path, IEnumerable<string> lines)
    {
        File.WriteAllLines(path, lines, Encoding.UTF8);
        return path;
    }

    private static float[] CreateRandomWeights(Random rng, int count, float scale)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
        {
            // Box-Muller approximate centered uniform -> clean deterministic bounded weights
            double u = rng.NextDouble() * 2.0 - 1.0;
            arr[i] = (float)(u * scale);
        }
        return arr;
    }

    private static float[] CreateOnes(int count)
    {
        var arr = new float[count];
        Array.Fill(arr, 1.0f);
        return arr;
    }

    private static void WriteGgufBinary(
        string path,
        List<(string Key, GgufValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, float[] Data)> tensors,
        uint alignment)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

        // Header: magic "GGUF" (0x46554747), version 3
        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, value) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            WriteGgufValue(bw, type, value);
        }

        ulong offset = 0;
        var offsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, dims, data) = tensors[i];
            ulong rem = offset % alignment;
            if (rem != 0) offset += alignment - rem;
            offsets[i] = offset;

            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++)
            {
                bw.Write(dims[d]);
            }
            bw.Write((uint)GgufType.F32);
            bw.Write(offset);

            offset += (ulong)data.Length * sizeof(float);
        }

        bw.Flush();
        long pos = fs.Position;
        long headerRem = pos % alignment;
        if (headerRem != 0)
        {
            fs.Write(new byte[alignment - headerRem]);
        }

        long dataStart = fs.Position;
        for (int i = 0; i < tensors.Count; i++)
        {
            long targetPos = dataStart + (long)offsets[i];
            long pad = targetPos - fs.Position;
            if (pad > 0) fs.Write(new byte[pad]);

            ReadOnlySpan<byte> raw = MemoryMarshal.AsBytes(tensors[i].Data.AsSpan());
            fs.Write(raw);
        }
        fs.Flush();
    }

    private static void WriteGgufString(BinaryWriter bw, string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static void WriteGgufValue(BinaryWriter bw, GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.Uint32:
                bw.Write((uint)value);
                break;
            case GgufValueType.Int32:
                bw.Write((int)value);
                break;
            case GgufValueType.Float32:
                bw.Write((float)value);
                break;
            case GgufValueType.String:
                WriteGgufString(bw, (string)value);
                break;
            case GgufValueType.Array:
                if (value is string[] strArr)
                {
                    bw.Write((uint)GgufValueType.String);
                    bw.Write((ulong)strArr.Length);
                    foreach (string s in strArr) WriteGgufString(bw, s);
                }
                else if (value is int[] intArr)
                {
                    bw.Write((uint)GgufValueType.Int32);
                    bw.Write((ulong)intArr.Length);
                    foreach (int v in intArr) bw.Write(v);
                }
                else
                {
                    throw new NotSupportedException($"Unsupported array element type: {value.GetType()}");
                }
                break;
            default:
                throw new NotSupportedException($"Unsupported GGUF metadata type: {type}");
        }
    }
}
