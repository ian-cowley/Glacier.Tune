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
}
