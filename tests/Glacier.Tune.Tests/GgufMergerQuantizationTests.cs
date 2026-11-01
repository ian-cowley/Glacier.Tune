namespace Glacier.Tune.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tune.Export;
using Xunit;

public unsafe class GgufMergerQuantizationTests
{
    [Fact]
    public void SetScaleMinK4_And_GetScaleMinK4_Roundtrip()
    {
        byte* scales = stackalloc byte[12];

        byte[] testScales = [0, 1, 7, 15, 31, 45, 62, 63];
        byte[] testMins = [0, 2, 8, 16, 32, 44, 61, 63];

        new Span<byte>(scales, 12).Clear();

        // Pack all 8 sub-blocks
        for (int j = 0; j < 8; j++)
        {
            GgufMerger.SetScaleMinK4(j, scales, testScales[j], testMins[j]);
        }

        // Unpack and verify bit-exact fidelity
        for (int j = 0; j < 8; j++)
        {
            GgufMerger.GetScaleMinK4(j, scales, out byte d, out byte m);
            Assert.Equal(testScales[j], d);
            Assert.Equal(testMins[j], m);
        }
    }

    [Fact]
    public void QuantizeRowQ4_0_Fidelity_And_Roundtrip()
    {
        int count = 64; // 2 blocks of 32
        float* src = stackalloc float[count];
        for (int i = 0; i < count; i++)
        {
            src[i] = MathF.Sin(i * 0.2f) * 4.5f;
        }

        BlockQ4_0* blocks = stackalloc BlockQ4_0[2];
        GgufMerger.QuantizeRowQ4_0(src, blocks, count);

        float* dequant = stackalloc float[count];
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        float dot = 0f, normSrc = 0f, normDeq = 0f;
        float maxErr = 0f;
        float sumSqErr = 0f;

        for (int i = 0; i < count; i++)
        {
            float orig = src[i];
            float recovered = dequant[i];
            float err = MathF.Abs(recovered - orig);
            if (err > maxErr) maxErr = err;
            sumSqErr += err * err;

            dot += orig * recovered;
            normSrc += orig * orig;
            normDeq += recovered * recovered;
        }

        float mse = sumSqErr / count;
        float cosSim = dot / (MathF.Sqrt(normSrc) * MathF.Sqrt(normDeq));

        Assert.True(cosSim > 0.99f, $"Expected cosine similarity > 0.99, got {cosSim:F4}");
        Assert.True(mse < 0.05f, $"Expected MSE < 0.05, got {mse:F4}");
        Assert.True(maxErr < 1.0f, $"Expected max error < 1.0, got {maxErr:F4}");
    }

    [Fact]
    public void QuantizeRowQ4_0_EdgeCases_Zeros_And_Constants()
    {
        int count = 32;
        float* src = stackalloc float[count];
        BlockQ4_0* blocks = stackalloc BlockQ4_0[1];
        float* dequant = stackalloc float[count];

        // 1. All Zeros
        new Span<float>(src, count).Clear();
        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        Assert.Equal(0f, (float)blocks[0].Delta);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(0f, dequant[i]);
        }

        // 2. Constant Positive Values
        for (int i = 0; i < count; i++) src[i] = 3.5f;
        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(dequant[i] - 3.5f) < 0.2f, $"Expected ~3.5f, got {dequant[i]}");
        }

        // 3. Constant Negative Values
        for (int i = 0; i < count; i++) src[i] = -6.0f;
        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(dequant[i] - (-6.0f)) < 0.2f, $"Expected ~-6.0f, got {dequant[i]}");
        }
    }

    [Fact]
    public void QuantizeRowQ4_K_Fidelity_And_Roundtrip()
    {
        int count = 512; // 2 super-blocks of 256
        float* src = stackalloc float[count];
        for (int i = 0; i < count; i++)
        {
            src[i] = MathF.Sin(i * 0.05f) * MathF.Cos(i * 0.013f) * 2.5f;
        }

        BlockQ4_K* blocks = stackalloc BlockQ4_K[2];
        GgufMerger.QuantizeRowQ4_K(src, blocks, count);

        float* dequant = stackalloc float[count];
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        float dot = 0f, normSrc = 0f, normDeq = 0f;
        float maxErr = 0f;
        float sumSqErr = 0f;

        for (int i = 0; i < count; i++)
        {
            float orig = src[i];
            float recovered = dequant[i];
            Assert.False(float.IsNaN(recovered), $"NaN detected at element {i}");
            Assert.False(float.IsInfinity(recovered), $"Infinity detected at element {i}");

            float err = MathF.Abs(recovered - orig);
            if (err > maxErr) maxErr = err;
            sumSqErr += err * err;

            dot += orig * recovered;
            normSrc += orig * orig;
            normDeq += recovered * recovered;
        }

        float mse = sumSqErr / count;
        float cosSim = dot / (MathF.Sqrt(normSrc) * MathF.Sqrt(normDeq));

        Assert.True(cosSim > 0.995f, $"Expected cosine similarity > 0.995, got {cosSim:F4}");
        Assert.True(mse < 0.03f, $"Expected MSE < 0.03, got {mse:F4}");
        Assert.True(maxErr < 0.5f, $"Expected max error < 0.5, got {maxErr:F4}");
    }

    [Fact]
    public void QuantizeRowQ4_K_EdgeCases_Zeros_And_Const()
    {
        int count = 256;
        float* src = stackalloc float[count];
        BlockQ4_K* blocks = stackalloc BlockQ4_K[1];
        float* dequant = stackalloc float[count];

        // 1. All Zeros
        new Span<float>(src, count).Clear();
        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        Assert.Equal(0f, (float)blocks[0].Delta);
        Assert.Equal(0f, (float)blocks[0].DeltaMin);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(0f, dequant[i]);
        }

        // 2. Constant positive
        for (int i = 0; i < count; i++) src[i] = 4.0f;
        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(dequant[i] - 4.0f) < 0.1f, $"Expected ~4.0f, got {dequant[i]}");
        }
    }

    [Fact]
    public void ResolveTargetType_And_RowByteSize_Validations()
    {
        // Dimension 5120 (multiple of 256)
        Assert.Equal(GgufType.Q4_K, GgufMerger.ResolveTargetType(TargetQuantization.Auto, GgufType.Q4_K, 5120));
        Assert.Equal(GgufType.Q4_K, GgufMerger.ResolveTargetType(TargetQuantization.Auto, GgufType.Q6_K, 5120));
        Assert.Equal(GgufType.Q4_0, GgufMerger.ResolveTargetType(TargetQuantization.Auto, GgufType.Q4_0, 5120));
        Assert.Equal(GgufType.Q8_0, GgufMerger.ResolveTargetType(TargetQuantization.Auto, GgufType.Q8_0, 5120));

        // Dimension 160 (multiple of 32, but NOT 256 -> falls back to Q4_0 for Q4_K)
        Assert.Equal(GgufType.Q4_0, GgufMerger.ResolveTargetType(TargetQuantization.Q4_K, GgufType.Q4_K, 160));

        // Row byte sizes for 14B model dimensions
        Assert.Equal(2880, GgufMerger.GetRowByteSize(GgufType.Q4_K, 5120));
        Assert.Equal(2880, GgufMerger.GetRowByteSize(GgufType.Q4_0, 5120));
        Assert.Equal(5440, GgufMerger.GetRowByteSize(GgufType.Q8_0, 5120));

        Assert.Equal(7776, GgufMerger.GetRowByteSize(GgufType.Q4_K, 13824));
        Assert.Equal(7776, GgufMerger.GetRowByteSize(GgufType.Q4_0, 13824));
        Assert.Equal(14688, GgufMerger.GetRowByteSize(GgufType.Q8_0, 13824));
    }

    [Fact]
    public void GgufMerger_Merge_Q4_K_Fusion_ValidatesDescriptors()
    {
        using var fixture = SyntheticGgufFixture.CreateSuperBlockQwenModel(hiddenDim: 256, ffnDim: 512);
        string outputPath = Path.Combine(fixture.DirectoryPath, "fused_q4k.gguf");

        var result = GgufMerger.Merge(
            fixture.GgufPath,
            fixture.AdapterPath,
            outputPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q4_K });

        Assert.Equal(outputPath, result.OutputPath);
        Assert.True(result.OutputSizeBytes > 0);
        Assert.Equal(7, result.AdaptedTensorCount); // 7 projections for 1 layer

        using var gguf = new GgufFile(outputPath);
        Assert.Equal("qwen2", gguf.Architecture);

        // Verify adapted projections are Q4_K
        string[] adaptedNames =
        [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight"
        ];

        foreach (var name in adaptedNames)
        {
            Assert.True(gguf.TryGetTensor(name, out var tensor) && tensor != null, $"Tensor {name} must exist");
            Assert.Equal(GgufType.Q4_K, tensor.Type);
        }

        // Verify non-adapted tensors preserved as F32
        Assert.True(gguf.TryGetTensor("token_embd.weight", out var embd) && embd != null);
        Assert.Equal(GgufType.F32, embd.Type);
        Assert.True(gguf.TryGetTensor("output.weight", out var outW) && outW != null);
        Assert.Equal(GgufType.F32, outW.Type);
    }

    [Fact]
    public void GgufMerger_Merge_Q4_0_Fusion_ValidatesDescriptors()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 64, ffnDim: 128);
        string outputPath = Path.Combine(fixture.DirectoryPath, "fused_q40.gguf");

        var result = GgufMerger.Merge(
            fixture.GgufPath,
            fixture.AdapterPath,
            outputPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q4_0 });

        Assert.True(File.Exists(outputPath));
        Assert.Equal(7, result.AdaptedTensorCount);

        using var gguf = new GgufFile(outputPath);
        Assert.True(gguf.TryGetTensor("blk.0.attn_q.weight", out var tensor) && tensor != null);
        Assert.Equal(GgufType.Q4_0, tensor.Type);
    }

    [Fact]
    public void GgufMerger_Merge_Q8_0_Fusion_ValidatesDescriptors()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 64, ffnDim: 128);
        string outputPath = Path.Combine(fixture.DirectoryPath, "fused_q80.gguf");

        var result = GgufMerger.Merge(
            fixture.GgufPath,
            fixture.AdapterPath,
            outputPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q8_0 });

        Assert.True(File.Exists(outputPath));
        Assert.Equal(7, result.AdaptedTensorCount);

        using var gguf = new GgufFile(outputPath);
        Assert.True(gguf.TryGetTensor("blk.0.attn_q.weight", out var tensor) && tensor != null);
        Assert.Equal(GgufType.Q8_0, tensor.Type);
    }
}
