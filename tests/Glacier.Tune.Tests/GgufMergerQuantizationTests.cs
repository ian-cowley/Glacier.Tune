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
    public void FusedModel_GlacierArchitectCoder14B_DescriptorVerification()
    {
        string modelPath = @"Glacier-Architect-Coder-14B-Q4_K_M.gguf";
        if (!File.Exists(modelPath)) return;

        using var gguf = new GgufFile(modelPath);
        Assert.Equal("qwen2", gguf.Architecture);
        Assert.Equal(579, (int)gguf.TensorCount);

        int q4KCount = 0;
        int q8Count = 0;
        int f32Count = 0;
        int q6KCount = 0;

        foreach (var t in gguf.TensorList)
        {
            switch (t.Type)
            {
                case GgufType.Q4_K:
                    q4KCount++;
                    break;
                case GgufType.Q8_0:
                    q8Count++;
                    break;
                case GgufType.F32:
                    f32Count++;
                    break;
                case GgufType.Q6_K:
                    q6KCount++;
                    break;
            }
        }

        // Must have 0 Q8_0 upcast tensors
        Assert.Equal(0, q8Count);
        // All 336 adapted projections + token_embd.weight = 337 Q4_K tensors
        Assert.Equal(337, q4KCount);
        // Preserved F32 tensors = 241
        Assert.Equal(241, f32Count);
        // Preserved output.weight = 1 Q6_K tensor
        Assert.Equal(1, q6KCount);

        // Verify file size is <= 8.5 GB budget (8,518,000,000 bytes)
        long length = new FileInfo(modelPath).Length;
        Assert.True(length <= 8_518_000_000L, $"File size {length} must be <= 8,518,000,000 bytes");
    }

    [Fact]
    public void FusedModel_GlacierGrandmaster4B_DescriptorVerification()
    {
        string[] candidatePaths = [
            @"Glacier-Grandmaster-4B-Q4_K_M.gguf",
            @"models/Glacier-Grandmaster-4B-Q4_K_M.gguf",
            @"C:\Users\spuri\source\repos\PolarsPlus\Glacier-Grandmaster-4B-Q4_K_M.gguf",
            @"C:\Users\spuri\source\repos\PolarsPlus\models\Glacier-Grandmaster-4B-Q4_K_M.gguf",
            Path.Combine(AppContext.BaseDirectory, "Glacier-Grandmaster-4B-Q4_K_M.gguf"),
            Path.Combine(AppContext.BaseDirectory, "models", "Glacier-Grandmaster-4B-Q4_K_M.gguf")
        ];

        string? modelPath = null;
        foreach (var p in candidatePaths)
        {
            if (File.Exists(p)) { modelPath = p; break; }
        }

        if (modelPath == null) return;

        using var gguf = new GgufFile(modelPath);
        Assert.Equal("qwen3", gguf.Architecture);
        Assert.Equal(36, gguf.BlockCount);
        Assert.Equal(262144, gguf.ContextLength);
        Assert.Equal(398, (int)gguf.TensorCount);

        int q4KCount = 0;
        int q8Count = 0;
        int f32Count = 0;
        int q6KCount = 0;

        foreach (var t in gguf.TensorList)
        {
            switch (t.Type)
            {
                case GgufType.Q4_K:
                    q4KCount++;
                    break;
                case GgufType.Q8_0:
                    q8Count++;
                    break;
                case GgufType.F32:
                    f32Count++;
                    break;
                case GgufType.Q6_K:
                    q6KCount++;
                    break;
            }
        }

        // Must have 0 Q8_0 upcast tensors
        Assert.Equal(0, q8Count);
        // All 252 adapted projections are Q4_K
        Assert.Equal(252, q4KCount);
        // Preserved F32 tensors = 145 (1 output_norm + 36 * 4 norm tensors)
        Assert.Equal(145, f32Count);
        // Preserved token_embd.weight = 1 Q6_K tensor
        Assert.Equal(1, q6KCount);

        // Verify file size is <= 2.5 GB (2,500,000,000 bytes) and acceptance criteria <= 2.6 GB
        long length = new FileInfo(modelPath).Length;
        Assert.True(length <= 2_500_000_000L, $"File size {length} must be <= 2,500,000,000 bytes (<= 2.5 GB)");
    }
}
