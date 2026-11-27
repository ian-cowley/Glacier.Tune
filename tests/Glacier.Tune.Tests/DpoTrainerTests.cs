using System;
using System.IO;
using System.Linq;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

public class DpoTrainerTests
{
    [Theory]
    [InlineData(-2.5, -3.1, -2.6, -2.9, 0.1, 0.0)]
    [InlineData(-1.0, -4.0, -2.0, -2.0, 0.25, 0.0)]
    [InlineData(-5.0, -1.5, -2.5, -2.0, 0.5, 0.1)]
    public void DpoLoss_GradientsMatchFiniteDifferences(
        double piW, double piL, double refW, double refL, double beta, double smoothing)
    {
        const double eps = 1e-6;
        var res = DpoLoss.Compute(piW, piL, refW, refL, beta, smoothing);

        double fPlusW = DpoLoss.Compute(piW + eps, piL, refW, refL, beta, smoothing).Loss;
        double fMinusW = DpoLoss.Compute(piW - eps, piL, refW, refL, beta, smoothing).Loss;
        double numGradW = (fPlusW - fMinusW) / (2.0 * eps);

        double fPlusL = DpoLoss.Compute(piW, piL + eps, refW, refL, beta, smoothing).Loss;
        double fMinusL = DpoLoss.Compute(piW, piL - eps, refW, refL, beta, smoothing).Loss;
        double numGradL = (fPlusL - fMinusL) / (2.0 * eps);

        Assert.True(Math.Abs(res.GradPolicyChosen - numGradW) < 1e-6, $"Chosen grad mismatch: {res.GradPolicyChosen} vs {numGradW}");
        Assert.True(Math.Abs(res.GradPolicyRejected - numGradL) < 1e-6, $"Rejected grad mismatch: {res.GradPolicyRejected} vs {numGradL}");
    }

    [Theory]
    [InlineData(500.0, -500.0)]
    [InlineData(-500.0, 500.0)]
    public void DpoLoss_ExtremeMargins_RemainFiniteWithoutNanOrInfinity(double piW, double piL)
    {
        var res = DpoLoss.Compute(piW, piL, 0.0, 0.0, beta: 0.5);
        Assert.True(double.IsFinite(res.Loss));
        Assert.True(double.IsFinite(res.GradPolicyChosen));
        Assert.True(double.IsFinite(res.GradPolicyRejected));
        Assert.InRange(Math.Abs(res.GradPolicyChosen), 0.0, 0.5 + 1e-9);
    }

    [Fact]
    public void DpoDataset_LoadJsonl_ParsesValidPairsAndSkipsInvalidLines()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"dpo_{Guid.NewGuid():N}.jsonl");
        try
        {
            File.WriteAllLines(tempFile,
            [
                "{\"prompt\":\"2+2?\",\"chosen\":\"4\",\"rejected\":\"5\"}",
                "{invalid json}",
                "{\"prompt\":\"sky?\",\"chosen\":\"blue\",\"rejected\":\"green\"}"
            ]);

            var pairs = DpoDataset.LoadJsonl(tempFile);
            Assert.Equal(2, pairs.Count);
            Assert.Equal("4", pairs[0].Chosen);
            Assert.Equal("green", pairs[1].Rejected);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void DpoTrainer_MultiStepOptimization_ReducesLossAndIncreasesRewardMargin()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 2, hiddenDim: 64, ffnDim: 128, rank: 4, alpha: 8f);
        using var model = fixture.CreateLoraModel();

        // Construct contrasting token sequences where chosen != rejected on completion tokens
        int[] chosenIds = [1, 5, 9, 12, 17, 21, 25, 30];
        int[] chosenLabels = [-100, -100, -100, 12, 17, 21, 25, 30];
        int[] rejectedIds = [1, 5, 9, 2, 3, 4, 6, 7];
        int[] rejectedLabels = [-100, -100, -100, 2, 3, 4, 6, 7];

        var dataset = new[]
        {
            new TokenizedPreferencePair(chosenIds, chosenLabels, rejectedIds, rejectedLabels)
        };

        var args = new DpoTrainingArguments
        {
            Beta = 0.25,
            LearningRate = 1e-2f,
            WarmupRatio = 0.0f,
            Epochs = 6
        };

        using var trainer = new DpoTrainer(model, args);
        var history = trainer.Train(dataset);

        Assert.Equal(6, history.Count);
        Assert.True(history[^1].Loss < history[0].Loss, $"Expected DPO loss to drop: {history[0].Loss:F4} -> {history[^1].Loss:F4}");
        Assert.True(history[^1].RewardMargin > history[0].RewardMargin, $"Expected reward margin to increase: {history[0].RewardMargin:F4} -> {history[^1].RewardMargin:F4}");
    }
}
