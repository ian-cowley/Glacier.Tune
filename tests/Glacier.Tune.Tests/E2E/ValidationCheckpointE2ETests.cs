using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests.E2E;

[Collection("DiagnosticsLogger")]
[Trait("Suite", "UpgradesE2E")]
public sealed class ValidationCheckpointE2ETests
{
    private static SyntheticModelSpec CreateTinyBaseSpec() => new()
    {
        Architecture = "llama",
        BlockCount = 2,
        HiddenDim = 32,
        FfnDim = 64,
        HeadCount = 2,
        HeadCountKv = 2,
        HeadDim = 16,
        RopeDim = 16,
        HasQGate = false,
        HasQkNorm = false,
        Seed = 101
    };

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F11")]
    public void T1_F11_01_EvalSteps_DefaultsToZero_AndTrainOverloadWorksWithAndWithoutEvalDataset()
    {
        var defaultArgs = new TrainingArguments();
        Assert.Equal(0, defaultArgs.EvalSteps);

        using var scope = new TempDirectoryScope("f11_01_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var modelA = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(modelA.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(modelA.Tokenizer);

        // 1. Legacy call without evalDataset
        var argsA = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            MaxSteps = 2,
            EvalSteps = 0,
            OutputDir = scope.FilePath("out_no_eval")
        };
        using var trainerA = new LoraTrainer(modelA, argsA);
        trainerA.Train(trainDs);
        Assert.Null(trainerA.BestValidationLoss);
        Assert.Null(trainerA.LastValidationLoss);
        Assert.True(File.Exists(Path.Combine(argsA.OutputDir, "adapter_model.bin")));

        // 2. New overload with evalDataset but EvalSteps = 0
        using var modelB = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        var argsB = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            MaxSteps = 2,
            OutputDir = scope.FilePath("out_with_eval_zero")
        };
        using var trainerB = new LoraTrainer(modelB, argsB);
        trainerB.Train(trainDs, evalDataset: evalDs);
        Assert.Null(trainerB.BestValidationLoss);
        Assert.Null(trainerB.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(argsB.OutputDir, "best_adapter.bin")));
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F11")]
    public void T1_F11_02_ValidationRunsStrictlyAtMultiplesOfEvalSteps()
    {
        using var scope = new TempDirectoryScope("f11_02_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        var evalStepsLogged = new List<int>();
        var stepNumbers = new List<int>();
        var priorLogger = GlacierDiagnostics.Logger;
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, _) =>
        {
            if (msg.StartsWith("[EVAL] Step ", StringComparison.Ordinal))
            {
                int colonIdx = msg.IndexOf(':', 12);
                if (colonIdx > 12 && int.TryParse(msg.AsSpan(12, colonIdx - 12), out int step))
                {
                    evalStepsLogged.Add(step);
                }
            }
        }, LogLevel.Information);

        try
        {
            string outDir = scope.FilePath("out");
            string bestPath = Path.Combine(outDir, "best_adapter.bin");
            var args = new TrainingArguments
            {
                LearningRate = 2e-3f,
                Epochs = 1,
                GradientAccumulationSteps = 1,
                MaxSteps = 6,
                EvalSteps = 2,
                LoggingSteps = 1,
                OutputDir = outDir
            };

            float? lastValAtStep3 = null;
            float? lastValAtStep4 = null;
            float? lastValAtStep5 = null;
            float? lastValAtStep6 = null;

            using var trainer = new LoraTrainer(model, args);
            trainer.Train(trainDs, onStep: r =>
            {
                stepNumbers.Add(r.Step);
                if (r.Step is 1 or 2)
                {
                    Assert.Null(trainer.LastValidationLoss);
                    Assert.Null(trainer.BestValidationLoss);
                    Assert.False(File.Exists(bestPath));
                }
                else if (r.Step == 3)
                {
                    Assert.NotNull(trainer.LastValidationLoss);
                    Assert.NotNull(trainer.BestValidationLoss);
                    Assert.True(File.Exists(bestPath));
                    lastValAtStep3 = trainer.LastValidationLoss;
                }
                else if (r.Step == 4)
                {
                    lastValAtStep4 = trainer.LastValidationLoss;
                }
                else if (r.Step == 5)
                {
                    lastValAtStep5 = trainer.LastValidationLoss;
                }
                else if (r.Step == 6)
                {
                    lastValAtStep6 = trainer.LastValidationLoss;
                }
            }, evalDataset: evalDs);

            Assert.Equal([1, 2, 3, 4, 5, 6], stepNumbers);
            Assert.Equal([2, 4, 6], evalStepsLogged);
            Assert.NotNull(lastValAtStep3);
            Assert.Equal(lastValAtStep3, lastValAtStep4);
            Assert.NotNull(lastValAtStep5);
            Assert.Equal(lastValAtStep5, lastValAtStep6);
            Assert.NotNull(trainer.LastValidationLoss);
            Assert.True(float.IsFinite(trainer.LastValidationLoss!.Value));
            Assert.NotNull(trainer.BestValidationLoss);
            Assert.True(float.IsFinite(trainer.BestValidationLoss!.Value));
            Assert.True(trainer.BestValidationLoss!.Value <= trainer.LastValidationLoss!.Value);

            float finalEvalLoss = trainer.Evaluate(evalDs);
            Assert.InRange(MathF.Abs(finalEvalLoss - trainer.LastValidationLoss!.Value), 0f, 1e-4f);
        }
        finally
        {
            GlacierDiagnostics.Logger = priorLogger;
        }
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F11")]
    public void T1_F11_03_BestAdapterBinWrittenAndLoadableOnNewMinimumValLoss()
    {
        using var scope = new TempDirectoryScope("f11_03_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 3e-3f,
            Epochs = 2,
            GradientAccumulationSteps = 2,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        };

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(trainDs, evalDataset: evalDs);

        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        Assert.True(File.Exists(bestPath));
        Assert.True(new FileInfo(bestPath).Length > 0);
        Assert.NotNull(trainer.BestValidationLoss);

        // Verify best_adapter.bin loads cleanly into a fresh model and reproduces BestValidationLoss
        using var reloadedModel = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        reloadedModel.LoadAdapter(bestPath);
        using var evalTrainer = new LoraTrainer(reloadedModel, args);
        float reloadedLoss = evalTrainer.Evaluate(evalDs);
        Assert.True(float.IsFinite(reloadedLoss));
        Assert.InRange(MathF.Abs(reloadedLoss - trainer.BestValidationLoss!.Value), 0f, 1e-4f);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F11")]
    public void T1_F11_04_BestAdapterBinNotOverwrittenWhenValidationLossWorsens()
    {
        using var scope = new TempDirectoryScope("f11_04_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");

        // Steps 1-2 teach "ALPHA..." (matches evalDs), Steps 3-6 aggressively overwrite with conflicting target at high LR
        var trainDs = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("signal: eval", "ALPHA ALPHA ALPHA ALPHA ALPHA ALPHA"),
            ("signal: eval", "ALPHA ALPHA ALPHA ALPHA ALPHA ALPHA"),
            ("signal: eval", "ZZZZZ ZZZZZ ZZZZZ ZZZZZ ZZZZZ ZZZZZ"),
            ("signal: eval", "XXXXX XXXXX XXXXX XXXXX XXXXX XXXXX"),
            ("signal: eval", "QQQQQ QQQQQ QQQQQ QQQQQ QQQQQ QQQQQ"),
            ("signal: eval", "WWWWW WWWWW WWWWW WWWWW WWWWW WWWWW")
        ]);
        var evalDs = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("signal: eval", "ALPHA ALPHA ALPHA ALPHA ALPHA ALPHA")
        ]);

        string outDir = scope.FilePath("out");
        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        var args = new TrainingArguments
        {
            LearningRate = 0.08f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        };

        byte[]? bestHashAfterStep2 = null;
        float? bestLossAfterStep2 = null;

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(trainDs, onStep: r =>
        {
            if (r.Step == 3 && File.Exists(bestPath))
            {
                bestHashAfterStep2 = SHA256.HashData(File.ReadAllBytes(bestPath));
                bestLossAfterStep2 = trainer.BestValidationLoss;
            }
        }, evalDataset: evalDs);

        Assert.NotNull(bestHashAfterStep2);
        Assert.NotNull(bestLossAfterStep2);
        Assert.NotNull(trainer.BestValidationLoss);
        Assert.NotNull(trainer.LastValidationLoss);
        Assert.True(trainer.LastValidationLoss!.Value > trainer.BestValidationLoss!.Value,
            $"Expected final valLoss ({trainer.LastValidationLoss}) > bestValLoss ({trainer.BestValidationLoss}) due to conflicting steps.");

        byte[] finalBestHash = SHA256.HashData(File.ReadAllBytes(bestPath));
        Assert.Equal(bestHashAfterStep2, finalBestHash);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F11")]
    public void T1_F11_05_BestAdapterBinImprovesValidationLossOverUntrainedBaseModel()
    {
        using var scope = new TempDirectoryScope("f11_05_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var baseModel = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(baseModel.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(baseModel.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 5e-3f,
            Epochs = 3,
            GradientAccumulationSteps = 2,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        };

        using var baseTrainer = new LoraTrainer(baseModel, args);
        float initialValLoss = baseTrainer.Evaluate(evalDs);

        baseTrainer.Train(trainDs, evalDataset: evalDs);
        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        Assert.True(File.Exists(bestPath));

        using var bestModel = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        bestModel.LoadAdapter(bestPath);
        using var bestTrainer = new LoraTrainer(bestModel, args);
        float bestValLoss = bestTrainer.Evaluate(evalDs);

        Assert.True(bestValLoss < initialValLoss,
            $"Expected best_adapter.bin valLoss ({bestValLoss:F4}) < initial untrained valLoss ({initialValLoss:F4}).");
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_01_EvalStepsZeroWithNonNullEvalDatasetRunsNoValidation()
    {
        using var scope = new TempDirectoryScope("f11_t2_01_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            EvalSteps = 0,
            OutputDir = outDir
        };

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(trainDs, evalDataset: evalDs);

        Assert.Null(trainer.BestValidationLoss);
        Assert.Null(trainer.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(outDir, "best_adapter.bin")));
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_02_EvalStepsPositiveWithNullEvalDatasetIsGracefulNoOp()
    {
        using var scope = new TempDirectoryScope("f11_t2_02_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            EvalSteps = 2,
            OutputDir = outDir
        };

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(trainDs, evalDataset: null);

        Assert.Null(trainer.BestValidationLoss);
        Assert.Null(trainer.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(outDir, "best_adapter.bin")));
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_03_EvalStepsOneEveryStepWithExactTieDoesNotOverwriteBestAdapter()
    {
        using var scope = new TempDirectoryScope("f11_t2_03_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        var evalStepsLogged = new List<int>();
        var priorLogger = GlacierDiagnostics.Logger;
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, _) =>
        {
            if (msg.StartsWith("[EVAL] Step ", StringComparison.Ordinal))
            {
                int colonIdx = msg.IndexOf(':', 12);
                if (colonIdx > 12 && int.TryParse(msg.AsSpan(12, colonIdx - 12), out int step))
                {
                    evalStepsLogged.Add(step);
                }
            }
        }, LogLevel.Information);

        try
        {
            // LearningRate = 0.0f guarantees weights stay identical across steps 1, 2, 3 -> exact valLoss tie!
            var args = new TrainingArguments
            {
                LearningRate = 0.0f,
                WeightDecay = 0.0f,
                Epochs = 1,
                GradientAccumulationSteps = 1,
                MaxSteps = 3,
                EvalSteps = 1,
                OutputDir = outDir
            };

            DateTime? step1WriteTime = null;
            byte[]? step1Hash = null;

            using var trainer = new LoraTrainer(model, args);
            trainer.Train(trainDs, onStep: r =>
            {
                if (r.Step == 2)
                {
                    Assert.True(File.Exists(bestPath));
                    step1WriteTime = File.GetLastWriteTimeUtc(bestPath);
                    step1Hash = SHA256.HashData(File.ReadAllBytes(bestPath));
                    System.Threading.Thread.Sleep(25); // Ensure any spurious rewrite would alter timestamp
                }
            }, evalDataset: evalDs);

            Assert.Equal([1, 2, 3], evalStepsLogged);
            Assert.NotNull(step1WriteTime);
            Assert.Equal(step1WriteTime!.Value, File.GetLastWriteTimeUtc(bestPath));
            Assert.Equal(step1Hash, SHA256.HashData(File.ReadAllBytes(bestPath)));
            Assert.Equal(trainer.BestValidationLoss, trainer.LastValidationLoss);
        }
        finally
        {
            GlacierDiagnostics.Logger = priorLogger;
        }
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_04_EvalStepsGreaterThanTotalStepsCompletesWithoutError()
    {
        using var scope = new TempDirectoryScope("f11_t2_04_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            MaxSteps = 5,
            EvalSteps = 100,
            OutputDir = outDir
        };

        int stepsRan = 0;
        using var trainer = new LoraTrainer(model, args);
        trainer.Train(trainDs, onStep: _ => stepsRan++, evalDataset: evalDs);

        Assert.Equal(5, stepsRan);
        Assert.Null(trainer.BestValidationLoss);
        Assert.Null(trainer.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(outDir, "best_adapter.bin")));
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_05_EmptyEvalDatasetOrNegativeEvalStepsSafelySkipsWithoutNan()
    {
        using var scope = new TempDirectoryScope("f11_t2_05_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var emptyEvalDs = new ChatMlDataset();
        var nonEmptyEvalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        // 1. Empty evalDataset with positive EvalSteps
        var argsEmpty = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            MaxSteps = 2,
            EvalSteps = 1,
            OutputDir = scope.FilePath("out_empty")
        };
        using var trainerEmpty = new LoraTrainer(model, argsEmpty);
        Assert.True(float.IsNaN(trainerEmpty.Evaluate(emptyEvalDs)));
        trainerEmpty.Train(trainDs, evalDataset: emptyEvalDs);
        Assert.Null(trainerEmpty.BestValidationLoss);
        Assert.Null(trainerEmpty.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(argsEmpty.OutputDir, "best_adapter.bin")));

        // 2. Negative EvalSteps = -1 with non-empty evalDataset
        var argsNeg = new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            MaxSteps = 2,
            EvalSteps = -1,
            OutputDir = scope.FilePath("out_neg")
        };
        using var trainerNeg = new LoraTrainer(model, argsNeg);
        trainerNeg.Train(trainDs, evalDataset: nonEmptyEvalDs);
        Assert.Null(trainerNeg.BestValidationLoss);
        Assert.Null(trainerNeg.LastValidationLoss);
        Assert.False(File.Exists(Path.Combine(argsNeg.OutputDir, "best_adapter.bin")));
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_06_ResumeTrainingWithEvalDatasetHonorsAdditionalStepsAndUpdatesBestAdapter()
    {
        using var scope = new TempDirectoryScope("f11_t2_06_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var initModel = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateChatMlDataset(initModel.Tokenizer,
        [
            ("tag: alpha", "OK"),
            ("tag: beta", "OK")
        ]);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(initModel.Tokenizer);

        string initOut = scope.FilePath("init_ckpt");
        using (var initTrainer = new LoraTrainer(initModel, new TrainingArguments
        {
            LearningRate = 3e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            MaxSteps = 2,
            WarmupRatio = 0f,
            OutputDir = initOut
        }))
        {
            initTrainer.Train(trainDs);
        }
        string ckptPath = Path.Combine(initOut, "adapter_model.bin");
        Assert.True(File.Exists(ckptPath));

        void RunResumeCase(int? configuredMaxSteps, string outSubdir)
        {
            using var model = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
            string outDir = scope.FilePath(outSubdir);
            var args = new TrainingArguments
            {
                LearningRate = 3e-3f,
                Epochs = 5,
                GradientAccumulationSteps = 1,
                MaxSteps = configuredMaxSteps,
                EvalSteps = 2,
                WarmupRatio = 0f,
                OutputDir = outDir
            };

            var steps = new List<int>();
            var evalStepsLogged = new List<int>();
            var priorLogger = GlacierDiagnostics.Logger;
            GlacierDiagnostics.Logger = new DelegateGlacierLogger((_, msg, _) =>
            {
                if (msg.StartsWith("[EVAL] Step ", StringComparison.Ordinal))
                {
                    int colonIdx = msg.IndexOf(':', 12);
                    if (colonIdx > 12 && int.TryParse(msg.AsSpan(12, colonIdx - 12), out int step))
                    {
                        evalStepsLogged.Add(step);
                    }
                }
            }, LogLevel.Information);

            try
            {
                using var trainer = new LoraTrainer(model, args);
                trainer.ResumeTraining(
                    ckptPath,
                    trainDs,
                    startingStep: 2,
                    additionalSteps: 3,
                    onStep: r => steps.Add(r.Step),
                    evalDataset: evalDs);

                Assert.Equal([3, 4, 5], steps);
                Assert.Equal([4], evalStepsLogged);
                Assert.NotNull(trainer.BestValidationLoss);
                Assert.NotNull(trainer.LastValidationLoss);
                string bestPath = Path.Combine(outDir, "best_adapter.bin");
                Assert.True(File.Exists(bestPath));

                using var checkModel = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");
                checkModel.LoadAdapter(bestPath);
                using var checkTrainer = new LoraTrainer(checkModel, args);
                Assert.InRange(MathF.Abs(checkTrainer.Evaluate(evalDs) - trainer.BestValidationLoss!.Value), 0f, 1e-4f);
            }
            finally
            {
                GlacierDiagnostics.Logger = priorLogger;
            }
        }

        RunResumeCase(configuredMaxSteps: null, "resume_null_maxsteps");
        RunResumeCase(configuredMaxSteps: 2, "resume_maxsteps_2");
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F11")]
    public void T2_F11_07_TrailingShortExampleFlushesPendingAccumulatedGradients()
    {
        using var scope = new TempDirectoryScope("f11_t2_07_");
        string ggufPath = scope.FilePath("base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateTinyBaseSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var ds = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("a", "OK"),
            ("b", "OK"),
            ("c", "OK")
        ]);

        // Append a trailing <2-token example after 3 valid examples with GradientAccumulationSteps = 2
        var examplesField = typeof(ChatMlDataset).GetField("_examples", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var list = (List<TrainingExample>)examplesField.GetValue(ds)!;
        list.Add(new TrainingExample([65], [ChatMlDataset.IgnoreIndex]));
        Assert.Equal(4, ds.Count);

        var steps = new List<int>();
        var args = new TrainingArguments
        {
            LearningRate = 3e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 2,
            MaxSteps = null,
            WarmupRatio = 0f,
            OutputDir = scope.FilePath("out")
        };

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(ds, onStep: r => steps.Add(r.Step));

        // Examples 0+1 -> step 1; example 2's pending gradient must be flushed at end of epoch -> step 2
        Assert.Equal([1, 2], steps);
    }
}
