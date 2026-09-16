using System;
using System.IO;
using Glacier.Tune.Config;
using Glacier.Tune.Model;
using Xunit;

namespace Glacier.Tune.Tests;

public class LoraModelTests
{
    [Fact]
    public void GgufLoraModel_LoadsAndExtractsStats_WhenModelExists()
    {
        string ggufPath = @"C:\Users\spuri\source\repos\modelTrain\output\qwen2.5-coder-7b-enterprise-q8_0.gguf";
        if (!File.Exists(ggufPath)) return; // Skip if file not on current machine

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 16, Alpha = 32f });

        Assert.Equal(28, model.LayerCount);
        Assert.Equal(3584, model.HiddenDim);
        Assert.Equal(18944, model.FfnDim);

        var (trainable, total, pct) = model.GetParameterStats();
        Assert.True(trainable > 0);
        Assert.True(total > 0);
        Assert.True(pct < 5.0, "LoRA trainable parameters should be less than 5% of total base parameters");
    }

    [Fact]
    public void BaseGguf_InspectTensors()
    {
        string ggufPath = @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
        if (!File.Exists(ggufPath)) return;

        using var gguf = new Glacier.Inference.Gguf.GgufFile(ggufPath);
        Assert.True(gguf.TensorCount > 0);
        foreach (var t in gguf.TensorList)
        {
            ulong calcSize = t.GetByteSize();
            Assert.True(calcSize > 0, $"Tensor {t.Name} calculated size must be > 0");
        }
    }

    [Fact]
    public unsafe void GgufMerger_QuantizeRowQ8_0_HighFidelity()
    {
        int count = 64; // 2 blocks of 32
        float* src = stackalloc float[count];
        for (int i = 0; i < count; i++)
        {
            src[i] = MathF.Sin(i * 0.1f) * 5.0f;
        }

        Glacier.Inference.Quant.BlockQ8_0* blocks = stackalloc Glacier.Inference.Quant.BlockQ8_0[2];
        Glacier.Tune.Export.GgufMerger.QuantizeRowQ8_0(src, blocks, count);

        // Dequantize and verify < 0.5% max error
        for (int b = 0; b < 2; b++)
        {
            float scale = (float)blocks[b].Delta;
            Assert.True(scale > 0f);
            for (int i = 0; i < 32; i++)
            {
                float reconstructed = scale * blocks[b].Qs[i];
                float original = src[b * 32 + i];
                Assert.True(MathF.Abs(reconstructed - original) < 0.05f, $"Mismatch at {b * 32 + i}: got {reconstructed}, expected {original}");
            }
        }
    }
}
