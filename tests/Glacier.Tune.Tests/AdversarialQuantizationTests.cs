namespace Glacier.Tune.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tune.Export;
using Xunit;

/// <summary>
/// Adversarial stress-testing suite for GgufMerger quantization and bit-packing kernels.
/// Authored by Challenger 1 (Milestone M1).
/// </summary>
public unsafe class AdversarialQuantizationTests
{
    [Fact]
    public void SetScaleMinK4_Exhaustive_All64x64_BitExactRoundtrip()
    {
        // 6-bit values range from 0 to 63
        byte* scales = stackalloc byte[12];

        for (int j = 0; j < 8; j++)
        {
            for (int sc = 0; sc < 64; sc++)
            {
                for (int m = 0; m < 64; m++)
                {
                    new Span<byte>(scales, 12).Clear();

                    GgufMerger.SetScaleMinK4(j, scales, (byte)sc, (byte)m);
                    GgufMerger.GetScaleMinK4(j, scales, out byte recSc, out byte recM);

                    Assert.Equal((byte)sc, recSc);
                    Assert.Equal((byte)m, recM);
                }
            }
        }
    }

    [Fact]
    public void SetScaleMinK4_MultiBlock_NoInterferenceAcrossAll8Subblocks()
    {
        // Test that setting scale and min on all 8 blocks simultaneously does not corrupt neighboring blocks
        byte* scales = stackalloc byte[12];
        new Span<byte>(scales, 12).Clear();

        byte[] expectedSc = [0, 15, 31, 45, 63, 7, 52, 1];
        byte[] expectedM = [63, 44, 16, 2, 0, 59, 10, 33];

        for (int j = 0; j < 8; j++)
        {
            GgufMerger.SetScaleMinK4(j, scales, expectedSc[j], expectedM[j]);
        }

        for (int j = 0; j < 8; j++)
        {
            GgufMerger.GetScaleMinK4(j, scales, out byte d, out byte m);
            Assert.Equal(expectedSc[j], d);
            Assert.Equal(expectedM[j], m);
        }
    }

    [Fact]
    public void QuantizeRowQ4_K_Adversarial_AllZerosAndSubnormals()
    {
        int count = 256;
        float* src = stackalloc float[count];
        BlockQ4_K* blocks = stackalloc BlockQ4_K[1];
        float* dequant = stackalloc float[count];

        // 1. Exact zero
        new Span<float>(src, count).Clear();
        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        Assert.Equal(0f, (float)blocks[0].Delta);
        Assert.Equal(0f, (float)blocks[0].DeltaMin);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(0f, dequant[i]);
            Assert.False(float.IsNaN(dequant[i]));
            Assert.False(float.IsInfinity(dequant[i]));
        }

        // 2. Denormals / Subnormals (1e-40f to 1e-45f)
        for (int i = 0; i < count; i++)
        {
            src[i] = (i % 2 == 0) ? 1.401298e-45f : -1.401298e-45f;
        }

        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.False(float.IsNaN(dequant[i]), $"NaN at element {i} for denormal input");
            Assert.False(float.IsInfinity(dequant[i]), $"Infinity at element {i} for denormal input");
        }
    }

    [Fact]
    public void QuantizeRowQ4_K_Adversarial_SingleImpulseAndExtremeConstants()
    {
        int count = 512;
        float* src = stackalloc float[count];
        BlockQ4_K* blocks = stackalloc BlockQ4_K[2];
        float* dequant = stackalloc float[count];

        // 1. Single non-zero impulse in 512 elements
        new Span<float>(src, count).Clear();
        src[42] = 100.0f;
        src[300] = -50.0f;

        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        Assert.False(float.IsNaN(dequant[42]));
        Assert.False(float.IsNaN(dequant[300]));
        Assert.True(dequant[42] > 80.0f, $"Expected impulse ~100.0f, got {dequant[42]}");
        Assert.True(dequant[300] < -40.0f, $"Expected negative impulse ~-50.0f, got {dequant[300]}");

        // 2. Extreme constant negative (-1000.0f)
        for (int i = 0; i < count; i++) src[i] = -1000.0f;
        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.True(MathF.Abs(dequant[i] - (-1000.0f)) < 50.0f, $"Expected ~-1000.0f, got {dequant[i]}");
        }
    }

    [Fact]
    public void QuantizeRowQ4_K_RealisticLoraWeightDistribution_FidelityAbove99Percent()
    {
        // LoRA weights typically follow normal distribution with zero mean and small variance (e.g. sigma=0.02)
        int count = 2048; // 8 super blocks
        float* src = stackalloc float[count];
        BlockQ4_K* blocks = stackalloc BlockQ4_K[8];
        float* dequant = stackalloc float[count];

        var rng = new Random(42);
        for (int i = 0; i < count; i++)
        {
            // Box-Muller transform for normal distribution
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            src[i] = (float)(randStdNormal * 0.05); // sigma = 0.05
        }

        GgufMerger.QuantizeRowQ4_K(src, blocks, count);
        GgufMerger.DequantizeRowQ4_K(blocks, dequant, count);

        float dot = 0f, normSrc = 0f, normDeq = 0f;
        for (int i = 0; i < count; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            dot += src[i] * dequant[i];
            normSrc += src[i] * src[i];
            normDeq += dequant[i] * dequant[i];
        }

        float cosSim = dot / (MathF.Sqrt(normSrc) * MathF.Sqrt(normDeq));
        Assert.True(cosSim > 0.99f, $"Expected cosine similarity > 0.99 for LoRA Gaussian distribution, got {cosSim:F4}");
    }

    [Fact]
    public void QuantizeRowQ4_0_Adversarial_AllZerosAndSubnormals()
    {
        int count = 64;
        float* src = stackalloc float[count];
        BlockQ4_0* blocks = stackalloc BlockQ4_0[2];
        float* dequant = stackalloc float[count];

        // 1. All Zeros
        new Span<float>(src, count).Clear();
        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        Assert.Equal(0f, (float)blocks[0].Delta);
        Assert.Equal(0f, (float)blocks[1].Delta);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(0f, dequant[i]);
            Assert.False(float.IsNaN(dequant[i]));
        }

        // 2. Denormals / Subnormals
        for (int i = 0; i < count; i++)
        {
            src[i] = (i % 2 == 0) ? 1.401298e-45f : -1.401298e-45f;
        }

        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.False(float.IsInfinity(dequant[i]));
        }
    }

    [Fact]
    public void QuantizeRowQ4_0_Adversarial_ExtremeDynamicRange()
    {
        int count = 64;
        float* src = stackalloc float[count];
        BlockQ4_0* blocks = stackalloc BlockQ4_0[2];
        float* dequant = stackalloc float[count];

        // Extreme values: 1e5 and -1e5
        for (int i = 0; i < count; i++)
        {
            src[i] = (i % 2 == 0) ? 100000.0f : -100000.0f;
        }

        GgufMerger.QuantizeRowQ4_0(src, blocks, count);
        GgufMerger.DequantizeRowQ4_0(blocks, dequant, count);

        for (int i = 0; i < count; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.False(float.IsInfinity(dequant[i]));
            float expected = (i % 2 == 0) ? 100000.0f : -100000.0f;
            Assert.True(MathF.Abs(dequant[i] - expected) / MathF.Abs(expected) <= 0.13f, $"Element {i}: expected {expected}, got {dequant[i]}");
        }
    }
}
