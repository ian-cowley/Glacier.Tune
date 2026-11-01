using System;
using System.IO;
using Glacier.Tune.Config;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

public class LoraModelTests
{
    [Fact]
    public void GgufLoraModel_LoadsAndExtractsStats_WithSyntheticFixture()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 2, hiddenDim: 64, ffnDim: 128, rank: 4, alpha: 8f);
        using var model = fixture.CreateLoraModel();

        Assert.Equal(2, model.LayerCount);
        Assert.Equal(64, model.HiddenDim);
        Assert.Equal(128, model.FfnDim);

        var (trainable, total, pct) = model.GetParameterStats();
        Assert.True(trainable > 0, "Trainable parameters must be > 0");
        Assert.True(total > 0, "Total base parameters must be > 0");
        Assert.True(pct > 0.0 && pct < 50.0, $"LoRA trainable parameter ratio expected 0-50%, got {pct:F2}%");
    }

    [Fact]
    public void BaseGguf_InspectTensors_WithSyntheticFixture()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel();
        using var gguf = new Glacier.Inference.Gguf.GgufFile(fixture.GgufPath);

        Assert.True(gguf.TensorCount > 0, "Tensor count must be > 0");
        Assert.Equal("qwen2", gguf.Architecture);
        Assert.Equal(fixture.LayerCount, gguf.BlockCount);

        foreach (var t in gguf.TensorList)
        {
            ulong calcSize = t.GetByteSize();
            Assert.True(calcSize > 0, $"Tensor {t.Name} calculated size must be > 0");
        }
    }

    [Fact]
    public void GgufLoraModel_ForwardLossAndBackward_ComputesRealGradients()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 2, hiddenDim: 64, ffnDim: 128);
        using var model = fixture.CreateLoraModel();

        int[] inputTokens = [1, 5, 9, 12, 17, 21, 25, 30];
        int[] targetTokens = [5, 9, 12, 17, 21, 25, 30, -100];

        float loss = model.ForwardLossAndBackward(inputTokens, targetTokens);

        Assert.True(loss > 0f, $"Loss must be positive, got {loss}");
        Assert.False(float.IsNaN(loss), "Loss must not be NaN");
        Assert.False(float.IsInfinity(loss), "Loss must not be Infinity");

        // Verify gradients are accumulated on trainable adapter parameters
        bool hasNonZeroGradient = false;
        foreach (var p in model.TrainableParameters)
        {
            if (p.Grad != null)
            {
                var span = p.Grad.AsSpan();
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] != 0f)
                    {
                        hasNonZeroGradient = true;
                        break;
                    }
                }
            }
            if (hasNonZeroGradient) break;
        }

        Assert.True(hasNonZeroGradient, "Trainable parameters must receive non-zero gradients from backprop");
    }

    [Fact]
    public void LoraTrainer_Train_ExecutesRealOptimizationStep()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 64, ffnDim: 128);
        using var model = fixture.CreateLoraModel();
        var dataset = fixture.CreateSyntheticDataset(exampleCount: 2, seqLength: 6);

        var args = new TrainingArguments
        {
            LearningRate = 1e-4f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            MaxGradNorm = 1.0f,
            OutputDir = Path.Combine(fixture.DirectoryPath, "trained_adapter")
        };
        using var trainer = new LoraTrainer(model, args);
        int stepsObserved = 0;
        float lastLoss = 0f;

        trainer.Train(dataset, result =>
        {
            stepsObserved++;
            lastLoss = result.Loss;
            Assert.True(result.Step > 0);
            Assert.True(result.StepDurationMs >= 0.0);
            Assert.True(result.Loss > 0f, $"Step {result.Step}: Expected positive loss but got {result.Loss}");
        });

        Assert.True(stepsObserved >= 1, "At least 1 training step must execute");
        Assert.True(lastLoss > 0f, "Loss must be recorded");
        Assert.True(File.Exists(Path.Combine(args.OutputDir, "adapter_model.bin")), "Adapter must be exported");
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
