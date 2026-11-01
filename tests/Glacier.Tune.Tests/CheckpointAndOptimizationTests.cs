namespace Glacier.Tune.Tests;

using System;
using System.IO;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

public class CheckpointAndOptimizationTests
{
    // =========================================================================
    // 1. Cosine Decay Scheduling Tests
    // =========================================================================

    [Fact]
    public void CosineScheduler_WarmupLinearGrowth_MatchesAnalyticalValues()
    {
        const int totalSteps = 1000;
        const float baseLr = 2e-4f;
        const float warmupRatio = 0.05f; // 50 steps

        // Step 0: 0.0
        float lr0 = LoraTrainer.ComputeCosineLr(0, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lr0);

        // Step 25: exactly 50% of warmup -> 0.5 * baseLr
        float lr25 = LoraTrainer.ComputeCosineLr(25, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr * 0.5f, lr25, precision: 6);

        // Step 50: end of warmup -> exactly baseLr
        float lr50 = LoraTrainer.ComputeCosineLr(50, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr, lr50, precision: 6);
    }

    [Fact]
    public void CosineScheduler_CosineDecay_MatchesAnalyticalValues()
    {
        const int totalSteps = 1000;
        const float baseLr = 1e-4f;
        const float warmupRatio = 0.0f; // No warmup: pure cosine decay

        // Step 0: progress 0.0 -> baseLr * 0.5 * (1 + cos(0)) = baseLr
        float lr0 = LoraTrainer.ComputeCosineLr(0, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr, lr0, precision: 6);

        // Step 500: progress 0.5 -> baseLr * 0.5 * (1 + cos(pi/2)) = 0.5 * baseLr
        float lr500 = LoraTrainer.ComputeCosineLr(500, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr * 0.5f, lr500, precision: 5);

        // Step 1000: progress 1.0 -> baseLr * 0.5 * (1 + cos(pi)) = 0.0
        float lr1000 = LoraTrainer.ComputeCosineLr(1000, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lr1000, precision: 5);

        // Step 1200: clamped beyond totalSteps -> 0.0
        float lrBeyond = LoraTrainer.ComputeCosineLr(1200, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lrBeyond, precision: 5);
    }

    [Fact]
    public void CosineScheduler_AnnealedLearningRate_DecaysProportionately()
    {
        const int totalSteps = 200;
        const float baseLr = 2e-4f;
        const float annealedLr = baseLr * 0.5f; // Round 1 annealed LR: 1e-4f

        float lrInitial = LoraTrainer.ComputeCosineLr(0, totalSteps, annealedLr, 0.0f);
        Assert.Equal(annealedLr, lrInitial, precision: 6);

        float lrMid = LoraTrainer.ComputeCosineLr(100, totalSteps, annealedLr, 0.0f);
        Assert.Equal(annealedLr * 0.5f, lrMid, precision: 6);

        float lrFinal = LoraTrainer.ComputeCosineLr(200, totalSteps, annealedLr, 0.0f);
        Assert.Equal(0.0f, lrFinal, precision: 6);
    }

    // =========================================================================
    // 2. SIMD Gradient Norm Clipping Tests
    // =========================================================================

    [Fact]
    public void GradientNormClipping_BelowMaxNorm_PreservesGradients()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel();
        using var dummyModel = fixture.CreateLoraModel();
        var trainer = new LoraTrainer(dummyModel, new TrainingArguments { MaxGradNorm = 1.0f });

        // Set parameter gradients to values with total norm < 1.0
        var param = dummyModel.TrainableParameters[0];
        param.Grad!.Fill(0.01f);

        var spanBefore = param.Grad.AsSpan().ToArray();
        float norm = trainer.ClipGradients(1.0f);

        Assert.True(norm < 1.0f, $"Norm should be below 1.0, got {norm}");
        var spanAfter = param.Grad.AsSpan().ToArray();

        for (int i = 0; i < spanBefore.Length; i++)
        {
            Assert.Equal(spanBefore[i], spanAfter[i]);
        }
    }

    [Fact]
    public void GradientNormClipping_AboveMaxNorm_ScalesToUnitNormWithSimdVector256()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel();
        using var dummyModel = fixture.CreateLoraModel();
        var trainer = new LoraTrainer(dummyModel, new TrainingArguments { MaxGradNorm = 1.0f });

        // Fill all gradients with large values so total norm is well above 1.0
        foreach (var p in dummyModel.TrainableParameters)
        {
            p.Grad!.Fill(2.0f);
        }

        float initialNorm = trainer.ClipGradients(1.0f);
        Assert.True(initialNorm > 1.0f, $"Initial norm should exceed 1.0, got {initialNorm}");

        // Now re-calculate post-clipping norm: should be exactly ~1.0
        double postNormSq = 0.0;
        foreach (var p in dummyModel.TrainableParameters)
        {
            var span = p.Grad!.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                postNormSq += (double)span[i] * span[i];
            }
        }
        float postNorm = MathF.Sqrt((float)postNormSq);
        Assert.True(MathF.Abs(postNorm - 1.0f) < 1e-4f, $"Post-clip norm should be 1.0, got {postNorm}");
    }

    // =========================================================================
    // 3. Round-Trip SaveAdapter -> LoadAdapter Mathematical Exactness
    // =========================================================================

    [Fact]
    public void SaveAdapter_LoadAdapter_All7Projections_MathematicalExactness_RoundTrip()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel();
        string tempDir = Path.Combine(fixture.DirectoryPath, "SaveLoadRoundtrip");
        Directory.CreateDirectory(tempDir);

        using var model = fixture.CreateLoraModel();
        Assert.True(model.Blocks.Count >= 1);

        var block = model.Blocks[0];

        // 1. Fill all 7 projection matrices with known values
        var rng = new Random(1337);
        FillTensor(block.QProj.AdapterA, rng);
        FillTensor(block.QProj.AdapterB, rng);
        FillTensor(block.KProj.AdapterA, rng);
        FillTensor(block.KProj.AdapterB, rng);
        FillTensor(block.VProj.AdapterA, rng);
        FillTensor(block.VProj.AdapterB, rng);
        FillTensor(block.OProj.AdapterA, rng);
        FillTensor(block.OProj.AdapterB, rng);
        FillTensor(block.GateProj.AdapterA, rng);
        FillTensor(block.GateProj.AdapterB, rng);
        FillTensor(block.UpProj.AdapterA, rng);
        FillTensor(block.UpProj.AdapterB, rng);
        FillTensor(block.DownProj.AdapterA, rng);
        FillTensor(block.DownProj.AdapterB, rng);

        // Snapshot expected values
        float[] expectedQA = block.QProj.AdapterA.AsSpan().ToArray();
        float[] expectedQB = block.QProj.AdapterB.AsSpan().ToArray();
        float[] expectedKA = block.KProj.AdapterA.AsSpan().ToArray();
        float[] expectedKB = block.KProj.AdapterB.AsSpan().ToArray();

        // 2. Save adapter
        model.SaveAdapter(tempDir);
        string binFile = Path.Combine(tempDir, "adapter_model.bin");
        Assert.True(File.Exists(binFile), "adapter_model.bin must exist on disk");

        // 3. Clear matrices in memory
        block.QProj.AdapterA.Fill(0f);
        block.QProj.AdapterB.Fill(0f);
        block.KProj.AdapterA.Fill(0f);
        block.KProj.AdapterB.Fill(0f);

        // 4. Reload adapter and verify exact mathematical equivalence
        model.LoadAdapter(tempDir);

        AssertTensorsEqual(expectedQA, block.QProj.AdapterA);
        AssertTensorsEqual(expectedQB, block.QProj.AdapterB);
        AssertTensorsEqual(expectedKA, block.KProj.AdapterA);
        AssertTensorsEqual(expectedKB, block.KProj.AdapterB);

        // 5. Direct binary file path loading
        block.QProj.AdapterA.Fill(-999f);
        model.LoadAdapter(binFile);
        AssertTensorsEqual(expectedQA, block.QProj.AdapterA);
    }

    // =========================================================================
    // 4. LoraTrainer Checkpoint Resumption Tests
    // =========================================================================

    [Fact]
    public void LoraTrainer_ResumeFromCheckpoint_SetsAnnealedLrAndCustomStartingStep()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel();
        string tempDir = Path.Combine(fixture.DirectoryPath, "ResumeCheckpoint");
        Directory.CreateDirectory(tempDir);

        using var model = fixture.CreateLoraModel();
        model.SaveAdapter(tempDir);

        var args = new TrainingArguments
        {
            LearningRate = 2e-4f,
            WarmupRatio = 0.05f
        };

        using var trainer = new LoraTrainer(model, args);
        Assert.Equal(0, trainer.StartingStep);

        // Resume with annealed learning rate and custom starting step
        const float annealedLr = 1e-4f;
        const int startingStep = 450;
        trainer.ResumeFromCheckpoint(tempDir, annealedLearningRate: annealedLr, startingStep: startingStep);

        Assert.Equal(startingStep, trainer.StartingStep);
        Assert.Equal(annealedLr, trainer.ResumedLearningRate);
        Assert.Equal(annealedLr, trainer.Optimizer.LearningRate);
    }

    private static void FillTensor(Tensor<float> tensor, Random rng)
    {
        var span = tensor.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
        }
    }

    private static void AssertTensorsEqual(float[] expected, Tensor<float> actualTensor)
    {
        var actual = actualTensor.AsSpan();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }
}
