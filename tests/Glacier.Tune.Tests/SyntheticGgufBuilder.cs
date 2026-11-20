using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;

namespace Glacier.Tune.Tests;

internal static class SyntheticGgufBuilder
{
    private const uint GgufMagic = 0x46554747; // "GGUF"
    private const uint GgufVersion = 3;

    private enum ValueType : uint
    {
        Uint32 = 4,
        Float32 = 6,
        String = 8,
        Array = 9
    }

    public sealed record Qwen35Spec
    {
        public string Architecture { get; init; } = "qwen35";
        public int BlockCount { get; init; } = 2;
        public int ContextLength { get; init; } = 64;
        public int HiddenDim { get; init; } = 16;
        public int FfnDim { get; init; } = 24;
        public int HeadCount { get; init; } = 2;
        public int HeadCountKv { get; init; } = 1;
        public int HeadDim { get; init; } = 8;
        public int RopeDim { get; init; } = 4;
        public float RopeFreqBase { get; init; } = 10000f;
        public float RmsNormEps { get; init; } = 1e-6f;
        public int FullAttentionInterval { get; init; } = 2;
        public int SsmConvKernel { get; init; } = 4;
        public int SsmStateSize { get; init; } = 4;
        public int SsmGroupCount { get; init; } = 2;
        public int SsmTimeStepRank { get; init; } = 4;
        public bool GdnOnEvenLayers { get; init; } = true;
        public bool UseQGate { get; init; } = true;
        public bool UseQkNorm { get; init; } = true;
        public bool UseFusedGdnTensors { get; init; } = false;
        public bool UseFusedFullAttnQkv { get; init; } = false;
        public int Seed { get; init; } = 1234;
        public string[]? ExtraTokens { get; init; } = null;
    }

    public static string[] BuildGpt2ByteVocab(params string[] extraTokens)
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var byteToChar = new char[256];
        for (int i = 0; i < 256; i++)
        {
            byteToChar[bs[i]] = (char)cs[i];
        }

        var tokens = new List<string>(5 + 256 + (extraTokens?.Length ?? 0))
        {
            "<unk>",       // 0
            "<s>",         // 1
            "</s>",        // 2
            "<|im_start|>",// 3
            "<|im_end|>"   // 4
        };

        for (int b = 0; b < 256; b++)
        {
            tokens.Add(byteToChar[b].ToString());
        }

        if (extraTokens != null)
        {
            tokens.AddRange(extraTokens);
        }

        return tokens.ToArray();
    }

    public static void WriteQwen35HybridGguf(string path, Qwen35Spec spec)
    {
        var rng = new Random(spec.Seed);
        string[] vocab = BuildGpt2ByteVocab(spec.ExtraTokens ?? []);
        int vocabSize = vocab.Length;
        int hiddenDim = spec.HiddenDim;
        int ffnDim = spec.FfnDim;
        int nHeads = spec.HeadCount;
        int nHeadsKv = spec.HeadCountKv;
        int headDim = spec.HeadDim;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        int ssmInner = spec.SsmTimeStepRank * spec.SsmStateSize;
        int gdnConvChannels = (spec.SsmGroupCount * 2 + spec.SsmTimeStepRank) * spec.SsmStateSize;
        string arch = spec.Architecture;

        var metadata = new List<(string Key, ValueType Type, object Value)>
        {
            ("general.architecture", ValueType.String, arch),
            ("general.alignment", ValueType.Uint32, 32u),
            ($"{arch}.block_count", ValueType.Uint32, (uint)spec.BlockCount),
            ($"{arch}.context_length", ValueType.Uint32, (uint)spec.ContextLength),
            ($"{arch}.embedding_length", ValueType.Uint32, (uint)hiddenDim),
            ($"{arch}.feed_forward_length", ValueType.Uint32, (uint)ffnDim),
            ($"{arch}.attention.head_count", ValueType.Uint32, (uint)nHeads),
            ($"{arch}.attention.head_count_kv", ValueType.Uint32, (uint)nHeadsKv),
            ($"{arch}.attention.key_length", ValueType.Uint32, (uint)headDim),
            ($"{arch}.rope.freq_base", ValueType.Float32, spec.RopeFreqBase),
            ($"{arch}.rope.dimension_count", ValueType.Uint32, (uint)spec.RopeDim),
            ($"{arch}.attention.layer_norm_rms_epsilon", ValueType.Float32, spec.RmsNormEps),
            ($"{arch}.full_attention_interval", ValueType.Uint32, (uint)spec.FullAttentionInterval),
            ($"{arch}.ssm.conv_kernel", ValueType.Uint32, (uint)spec.SsmConvKernel),
            ($"{arch}.ssm.state_size", ValueType.Uint32, (uint)spec.SsmStateSize),
            ($"{arch}.ssm.group_count", ValueType.Uint32, (uint)spec.SsmGroupCount),
            ($"{arch}.ssm.time_step_rank", ValueType.Uint32, (uint)spec.SsmTimeStepRank),
            ($"{arch}.ssm.inner_size", ValueType.Uint32, (uint)ssmInner),
            ("tokenizer.ggml.model", ValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", ValueType.Array, vocab),
            ("tokenizer.ggml.merges", ValueType.Array, Array.Empty<string>()),
            ("tokenizer.ggml.bos_token_id", ValueType.Uint32, 1u),
            ("tokenizer.ggml.eos_token_id", ValueType.Uint32, 2u),
            ("tokenizer.ggml.padding_token_id", ValueType.Uint32, 1u)
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", [(ulong)hiddenDim, (ulong)vocabSize], RandomArray(rng, hiddenDim * vocabSize, 0.25f)),
            ("output_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)),
            ("output.weight", [(ulong)hiddenDim, (ulong)vocabSize], RandomArray(rng, hiddenDim * vocabSize, 0.25f))
        };

        for (int l = 0; l < spec.BlockCount; l++)
        {
            bool isGdn = spec.GdnOnEvenLayers ? (l % spec.FullAttentionInterval != (spec.FullAttentionInterval - 1)) : (l % spec.FullAttentionInterval == 0);
            if (spec.BlockCount == 1)
            {
                isGdn = spec.GdnOnEvenLayers;
            }

            tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)));

            if (isGdn)
            {
                if (spec.UseFusedGdnTensors)
                {
                    tensors.Add(($"blk.{l}.attn_qkv_gate.weight", [(ulong)hiddenDim, (ulong)(gdnConvChannels + ssmInner)], RandomArray(rng, hiddenDim * (gdnConvChannels + ssmInner), 0.2f)));
                    tensors.Add(($"blk.{l}.ssm_beta_alpha.weight", [(ulong)hiddenDim, (ulong)(2 * spec.SsmTimeStepRank)], RandomArray(rng, hiddenDim * 2 * spec.SsmTimeStepRank, 0.2f)));
                }
                else
                {
                    tensors.Add(($"blk.{l}.attn_qkv.weight", [(ulong)hiddenDim, (ulong)gdnConvChannels], RandomArray(rng, hiddenDim * gdnConvChannels, 0.2f)));
                    tensors.Add(($"blk.{l}.attn_gate.weight", [(ulong)hiddenDim, (ulong)ssmInner], RandomArray(rng, hiddenDim * ssmInner, 0.2f)));
                    tensors.Add(($"blk.{l}.ssm_alpha.weight", [(ulong)hiddenDim, (ulong)spec.SsmTimeStepRank], RandomArray(rng, hiddenDim * spec.SsmTimeStepRank, 0.2f)));
                    tensors.Add(($"blk.{l}.ssm_beta.weight", [(ulong)hiddenDim, (ulong)spec.SsmTimeStepRank], RandomArray(rng, hiddenDim * spec.SsmTimeStepRank, 0.2f)));
                }

                tensors.Add(($"blk.{l}.ssm_conv1d.weight", [(ulong)spec.SsmConvKernel, (ulong)gdnConvChannels], RandomArray(rng, spec.SsmConvKernel * gdnConvChannels, 0.25f)));
                tensors.Add(($"blk.{l}.ssm_a", [(ulong)spec.SsmTimeStepRank], NegativeArray(rng, spec.SsmTimeStepRank, 0.2f, 0.8f)));
                tensors.Add(($"blk.{l}.ssm_dt.bias", [(ulong)spec.SsmTimeStepRank], RandomArray(rng, spec.SsmTimeStepRank, 0.1f)));
                tensors.Add(($"blk.{l}.ssm_norm.weight", [(ulong)spec.SsmStateSize], OnesWithNoise(rng, spec.SsmStateSize, 0.05f)));
                tensors.Add(($"blk.{l}.ssm_out.weight", [(ulong)ssmInner, (ulong)hiddenDim], RandomArray(rng, ssmInner * hiddenDim, 0.2f)));
            }
            else
            {
                int qRows = spec.UseQGate ? 2 * qDim : qDim;
                if (spec.UseFusedFullAttnQkv)
                {
                    tensors.Add(($"blk.{l}.attn_qkv.weight", [(ulong)hiddenDim, (ulong)(qRows + 2 * kvDim)], RandomArray(rng, hiddenDim * (qRows + 2 * kvDim), 0.2f)));
                }
                else
                {
                    tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)hiddenDim, (ulong)qRows], RandomArray(rng, hiddenDim * qRows, 0.2f)));
                    tensors.Add(($"blk.{l}.attn_k.weight", [(ulong)hiddenDim, (ulong)kvDim], RandomArray(rng, hiddenDim * kvDim, 0.2f)));
                    tensors.Add(($"blk.{l}.attn_v.weight", [(ulong)hiddenDim, (ulong)kvDim], RandomArray(rng, hiddenDim * kvDim, 0.2f)));
                }

                if (spec.UseQkNorm)
                {
                    tensors.Add(($"blk.{l}.attn_q_norm.weight", [(ulong)headDim], OnesWithNoise(rng, headDim, 0.05f)));
                    tensors.Add(($"blk.{l}.attn_k_norm.weight", [(ulong)headDim], OnesWithNoise(rng, headDim, 0.05f)));
                }

                tensors.Add(($"blk.{l}.attn_output.weight", [(ulong)qDim, (ulong)hiddenDim], RandomArray(rng, qDim * hiddenDim, 0.2f)));
            }

            tensors.Add(($"blk.{l}.ffn_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", [(ulong)hiddenDim, (ulong)ffnDim], RandomArray(rng, hiddenDim * ffnDim, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_up.weight", [(ulong)hiddenDim, (ulong)ffnDim], RandomArray(rng, hiddenDim * ffnDim, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_down.weight", [(ulong)ffnDim, (ulong)hiddenDim], RandomArray(rng, ffnDim * hiddenDim, 0.2f)));
        }

        WriteGgufFile(path, metadata, tensors, alignment: 32);
    }

    public static void WriteStandardLlamaGguf(
        string path,
        int blockCount = 1,
        int hiddenDim = 16,
        int ffnDim = 24,
        int headCount = 2,
        int headCountKv = 2,
        int headDim = 8,
        int seed = 777,
        float[]? customOutputWeights = null,
        string[]? extraTokens = null)
    {
        var rng = new Random(seed);
        string[] vocab = BuildGpt2ByteVocab(extraTokens ?? []);
        int vocabSize = vocab.Length;
        int qDim = headCount * headDim;
        int kvDim = headCountKv * headDim;
        string arch = "llama";

        var metadata = new List<(string Key, ValueType Type, object Value)>
        {
            ("general.architecture", ValueType.String, arch),
            ("general.alignment", ValueType.Uint32, 32u),
            ($"{arch}.block_count", ValueType.Uint32, (uint)blockCount),
            ($"{arch}.context_length", ValueType.Uint32, 64u),
            ($"{arch}.embedding_length", ValueType.Uint32, (uint)hiddenDim),
            ($"{arch}.feed_forward_length", ValueType.Uint32, (uint)ffnDim),
            ($"{arch}.attention.head_count", ValueType.Uint32, (uint)headCount),
            ($"{arch}.attention.head_count_kv", ValueType.Uint32, (uint)headCountKv),
            ($"{arch}.attention.key_length", ValueType.Uint32, (uint)headDim),
            ($"{arch}.rope.freq_base", ValueType.Float32, 10000f),
            ($"{arch}.attention.layer_norm_rms_epsilon", ValueType.Float32, 1e-6f),
            ("tokenizer.ggml.model", ValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", ValueType.Array, vocab),
            ("tokenizer.ggml.merges", ValueType.Array, Array.Empty<string>()),
            ("tokenizer.ggml.bos_token_id", ValueType.Uint32, 1u),
            ("tokenizer.ggml.eos_token_id", ValueType.Uint32, 2u),
            ("tokenizer.ggml.padding_token_id", ValueType.Uint32, 1u)
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", [(ulong)hiddenDim, (ulong)vocabSize], RandomArray(rng, hiddenDim * vocabSize, 0.25f)),
            ("output_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)),
            ("output.weight", [(ulong)hiddenDim, (ulong)vocabSize], customOutputWeights ?? RandomArray(rng, hiddenDim * vocabSize, 0.25f))
        };

        for (int l = 0; l < blockCount; l++)
        {
            tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)));
            tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)hiddenDim, (ulong)qDim], RandomArray(rng, hiddenDim * qDim, 0.2f)));
            tensors.Add(($"blk.{l}.attn_k.weight", [(ulong)hiddenDim, (ulong)kvDim], RandomArray(rng, hiddenDim * kvDim, 0.2f)));
            tensors.Add(($"blk.{l}.attn_v.weight", [(ulong)hiddenDim, (ulong)kvDim], RandomArray(rng, hiddenDim * kvDim, 0.2f)));
            tensors.Add(($"blk.{l}.attn_output.weight", [(ulong)qDim, (ulong)hiddenDim], RandomArray(rng, qDim * hiddenDim, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", [(ulong)hiddenDim], OnesWithNoise(rng, hiddenDim, 0.05f)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", [(ulong)hiddenDim, (ulong)ffnDim], RandomArray(rng, hiddenDim * ffnDim, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_up.weight", [(ulong)hiddenDim, (ulong)ffnDim], RandomArray(rng, hiddenDim * ffnDim, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_down.weight", [(ulong)ffnDim, (ulong)hiddenDim], RandomArray(rng, ffnDim * hiddenDim, 0.2f)));
        }

        WriteGgufFile(path, metadata, tensors, alignment: 32);
    }

    private static void WriteGgufFile(
        string path,
        List<(string Key, ValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, float[] Data)> tensors,
        uint alignment)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(GgufMagic);
        bw.Write(GgufVersion);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, value) in metadata)
        {
            WriteString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case ValueType.Uint32:
                    bw.Write((uint)value);
                    break;
                case ValueType.Float32:
                    bw.Write((float)value);
                    break;
                case ValueType.String:
                    WriteString(bw, (string)value);
                    break;
                case ValueType.Array:
                    string[] arr = (string[])value;
                    bw.Write((uint)ValueType.String);
                    bw.Write((ulong)arr.Length);
                    foreach (string s in arr)
                    {
                        WriteString(bw, s);
                    }
                    break;
                default:
                    throw new NotSupportedException($"Unsupported test GGUF value type: {type}");
            }
        }

        var offsets = new ulong[tensors.Count];
        ulong runningOffset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            ulong rem = runningOffset % alignment;
            if (rem != 0) runningOffset += alignment - rem;
            offsets[i] = runningOffset;
            runningOffset += (ulong)tensors[i].Data.Length * sizeof(float);
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            WriteString(bw, t.Name);
            bw.Write((uint)t.Dims.Length);
            for (int d = 0; d < t.Dims.Length; d++)
            {
                bw.Write(t.Dims[d]);
            }
            bw.Write((uint)GgufType.F32);
            bw.Write(offsets[i]);
        }

        bw.Flush();
        long pos = fs.Position;
        long pad = pos % alignment == 0 ? 0 : alignment - (pos % alignment);
        if (pad > 0) bw.Write(new byte[pad]);

        long dataStart = fs.Position;
        for (int i = 0; i < tensors.Count; i++)
        {
            long target = dataStart + (long)offsets[i];
            long gap = target - fs.Position;
            if (gap > 0) bw.Write(new byte[gap]);
            bw.Write(MemoryMarshal.AsBytes(tensors[i].Data.AsSpan()));
        }
        bw.Flush();
    }

    private static void WriteString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static float[] RandomArray(Random rng, int len, float scale)
    {
        var arr = new float[len];
        for (int i = 0; i < len; i++)
        {
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        }
        return arr;
    }

    private static float[] OnesWithNoise(Random rng, int len, float noise)
    {
        var arr = new float[len];
        for (int i = 0; i < len; i++)
        {
            arr[i] = 1.0f + (float)(rng.NextDouble() * 2.0 - 1.0) * noise;
        }
        return arr;
    }

    private static float[] NegativeArray(Random rng, int len, float minAbs, float maxAbs)
    {
        var arr = new float[len];
        for (int i = 0; i < len; i++)
        {
            arr[i] = -(minAbs + (float)rng.NextDouble() * (maxAbs - minAbs));
        }
        return arr;
    }
}
