using System;
using System.IO;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Tune.Config;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Export;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests.E2E;

[Trait("Suite", "UpgradesE2E")]
public sealed class CrossFeatureTuneE2ETests
{
    private static SyntheticModelSpec CreateQwen35HybridSpec(int seed = 2026, bool fuseGdnTensors = false) => new()
    {
        Architecture = "qwen35",
        BlockCount = 2,
        FullAttentionInterval = 2,
        HiddenDim = 32,
        FfnDim = 64,
        HeadCount = 2,
        HeadCountKv = 2,
        HeadDim = 16,
        RopeDim = 8,
        ConvKernel = 4,
        StateSize = 8,
        GroupCount = 2,
        SsmHeads = 4,
        HasQGate = true,
        HasQkNorm = true,
        FuseGdnTensors = fuseGdnTensors,
        Seed = seed
    };

    [Fact]
    [Trait("Tier", "Tier3")]
    [Trait("Feature", "F11+F12")]
    public void T3_11_Qwen35GdnTrainingWithEvalStepsSavesLowestLossBestAdapter()
    {
        using var scope = new TempDirectoryScope("t3_11_");
        string ggufPath = scope.FilePath("qwen35.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        var args = new TrainingArguments
        {
            LearningRate = 5e-3f,
            Epochs = 2,
            GradientAccumulationSteps = 2,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        };

        using var trainer = new LoraTrainer(model, args);
        float baseValLoss = trainer.Evaluate(evalDs);
        trainer.Train(trainDs, evalDataset: evalDs);

        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        Assert.True(File.Exists(bestPath));
        Assert.NotNull(trainer.BestValidationLoss);
        Assert.True(trainer.BestValidationLoss!.Value < baseValLoss,
            $"Expected hybrid GDN best valLoss ({trainer.BestValidationLoss:F4}) < base ({baseValLoss:F4}).");

        using var reloaded = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        reloaded.LoadAdapter(bestPath);
        using var evalTrainer = new LoraTrainer(reloaded, args);
        Assert.InRange(MathF.Abs(evalTrainer.Evaluate(evalDs) - trainer.BestValidationLoss.Value), 0f, 1e-4f);
    }

    [Fact]
    [Trait("Tier", "Tier3")]
    [Trait("Feature", "F12+F13")]
    public void T3_12_Qwen35GdnLoraFeedsPostTuneEvaluatorWithMetricsAndTokenReduction()
    {
        using var scope = new TempDirectoryScope("t3_12_");
        string ggufPath = scope.FilePath("qwen35.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec(seed: 303));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var baseModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        using var tunedModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");

        var trainDs = TuneE2EFixtures.CreateChatMlDataset(tunedModel.Tokenizer,
        [
            ("grade", "OK"),
            ("grade", "OK")
        ]);

        using (var trainer = new LoraTrainer(tunedModel, new TrainingArguments
        {
            LearningRate = 0.04f,
            Epochs = 15,
            GradientAccumulationSteps = 1,
            MaxSteps = 20,
            WarmupRatio = 0f,
            OutputDir = scope.FilePath("out")
        }))
        {
            trainer.Train(trainDs);
        }

        // Base prompt = 20 ASCII chars (20 tokens), Tuned prompt = "grade" inside ChatML or plain 5-char prompt
        string evalPath = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            """{"messages":[{"role":"system","content":"12345678901234567890"},{"role":"user","content":"grade"},{"role":"assistant","content":"OK"}]}"""
        ]);

        var report = PostTuneEvaluator.Evaluate(baseModel, tunedModel, evalPath,
            new EvalOptions { MaxNewTokens = 4, StripSystemTurnForFineTuned = true });

        Assert.Equal(1, report.TotalExamples);
        Assert.True(report.TunedExactMatch >= report.BaseExactMatch);
        Assert.True(report.TunedF1 >= report.BaseF1);
        Assert.Equal(1.0, report.TunedExactMatch);
        Assert.True(report.TotalBasePromptTokens > report.TotalTunedPromptTokens);
        double expectedReduction = (double)(report.TotalBasePromptTokens - report.TotalTunedPromptTokens) / report.TotalBasePromptTokens * 100.0;
        Assert.InRange(Math.Abs(report.PromptTokenReductionPercent - expectedReduction), 0.0, 1e-6);
    }

    [Fact]
    [Trait("Tier", "Tier3")]
    [Trait("Feature", "F11+F13")]
    public void T3_13_BestAdapterSelectedByLoraTrainerOutperformsOverfittedFinalInPostTuneEvaluator()
    {
        using var scope = new TempDirectoryScope("t3_13_");
        string ggufPath = scope.FilePath("qwen35.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec(seed: 404));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var trainModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");

        // Phase 1: 14 examples of ("task", "OK") -> learns "OK" cleanly (EM=1.0) and saves best_adapter.bin
        // Phase 2: 18 conflicting examples of ("task", "ZZ") -> overwrites final adapter_model.bin while LR is active
        var mixedPairs = new System.Collections.Generic.List<(string, string)>();
        for (int i = 0; i < 14; i++) mixedPairs.Add(("task", "OK"));
        for (int i = 0; i < 18; i++) mixedPairs.Add(("task", "ZZ"));

        var trainDs = TuneE2EFixtures.CreateChatMlDataset(trainModel.Tokenizer, mixedPairs);
        var evalDs = TuneE2EFixtures.CreateChatMlDataset(trainModel.Tokenizer, [("task", "OK")]);

        string outDir = scope.FilePath("out");
        using (var trainer = new LoraTrainer(trainModel, new TrainingArguments
        {
            LearningRate = 0.05f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        }))
        {
            trainer.Train(trainDs, evalDataset: evalDs);
        }

        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        string finalPath = Path.Combine(outDir, "adapter_model.bin");
        Assert.True(File.Exists(bestPath));
        Assert.True(File.Exists(finalPath));

        string evalJsonl = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            """{"messages":[{"role":"user","content":"task"},{"role":"assistant","content":"OK"}]}"""
        ]);

        using var bestModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        bestModel.LoadAdapter(bestPath);
        using var finalModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        finalModel.LoadAdapter(finalPath);

        // Compare finalModel (as baseline) vs bestModel (as tuned) in PostTuneEvaluator
        var report = PostTuneEvaluator.Evaluate(finalModel, bestModel, evalJsonl, new EvalOptions { MaxNewTokens = 4 });
        Assert.True(report.TunedExactMatch > report.BaseExactMatch,
            $"Expected best_adapter.bin EM ({report.TunedExactMatch:F2}) > overfitted final EM ({report.BaseExactMatch:F2}).");
        Assert.True(report.TunedF1 > report.BaseF1,
            $"Expected best_adapter.bin F1 ({report.TunedF1:F2}) > overfitted final F1 ({report.BaseF1:F2}).");
    }

    [Fact]
    [Trait("Tier", "Tier3")]
    [Trait("Feature", "F9+F12")]
    public void T3_14_MergedQwen35GdnGgufPassesInferenceKlDivergenceVerifier()
    {
        using var scope = new TempDirectoryScope("t3_14_");
        static void VerifyMergeAndKl(string baseGgufPath, string adapterBinPath, string mergedGgufPath, TargetQuantization quant, int[] tok, ReadOnlySpan<float> refLogits, double threshold, string label)
        {
            var mergeResult = GgufMerger.Merge(baseGgufPath, adapterBinPath, mergedGgufPath,
                new GgufMerger.MergeOptions { TargetQuantization = quant });
            Assert.True(mergeResult.AdaptedTensorCount > 0, $"{label}: expected AdaptedTensorCount > 0");

            using var mergedSession = new InferenceSession(mergedGgufPath, maxSeqLen: 128, device: "cpu");
            float[] mergedLogits = new float[tok.Length * mergedSession.VocabSize];
            mergedSession.VerifyBatchLogits(tok, 0, mergedLogits);

            var klResult = KlDivergenceVerifier.EvaluateLogits(
                refLogits,
                mergedLogits,
                mergedSession.VocabSize,
                new KlDivergenceOptions { Threshold = threshold, MinTop1Agreement = 1.0 });

            Assert.True(klResult.Passed, $"{label} KL verification failed: {klResult.FailureReason}");
            Assert.True(klResult.MeanKlDivergence < threshold, $"{label} MeanKL={klResult.MeanKlDivergence}");
            Assert.Equal(1.0, klResult.Top1AgreementRate);
        }

        // 1. Unfused GDN tensors: F32 + Q8_0 merge & InferenceSession KL verification
        string baseGguf = scope.FilePath("qwen35_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(baseGguf, CreateQwen35HybridSpec(seed: 505));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var loraModel = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");
        var ds = TuneE2EFixtures.CreateChatMlDataset(loraModel.Tokenizer, [("hello", "world")]);

        string outDir = scope.FilePath("out");
        using (var trainer = new LoraTrainer(loraModel, new TrainingArguments
        {
            LearningRate = 1e-2f,
            Epochs = 4,
            GradientAccumulationSteps = 1,
            MaxSteps = 4,
            WarmupRatio = 0f,
            OutputDir = outDir
        }))
        {
            trainer.Train(ds);
        }

        string adapterBin = Path.Combine(outDir, "adapter_model.bin");
        int[] tokens = loraModel.Tokenizer.Encode("<|im_start|>user\nhello<|im_end|>\n");
        using var unmergedLogits = loraModel.ForwardLogits(tokens);

        VerifyMergeAndKl(baseGguf, adapterBin, scope.FilePath("qwen35_merged_f32.gguf"),
            TargetQuantization.F32, tokens, unmergedLogits.AsSpan(), 1e-4, "Unfused-F32");
        VerifyMergeAndKl(baseGguf, adapterBin, scope.FilePath("qwen35_merged_q8_0.gguf"),
            TargetQuantization.Q8_0, tokens, unmergedLogits.AsSpan(), 1e-3, "Unfused-Q8_0");

        // 2. Fused GDN tensors (attn_qkv_gate + ssm_beta_alpha): F32 + Q8_0 merge & InferenceSession KL verification
        string fusedBaseGguf = scope.FilePath("qwen35_fused_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(fusedBaseGguf, CreateQwen35HybridSpec(seed: 1505, fuseGdnTensors: true));

        using var fusedModel = GgufLoraModel.Load(fusedBaseGguf, cfg, device: "cpu");
        var fusedDs = TuneE2EFixtures.CreateChatMlDataset(fusedModel.Tokenizer, [("hello", "world")]);
        string fusedOutDir = scope.FilePath("fused_out");
        using (var fusedTrainer = new LoraTrainer(fusedModel, new TrainingArguments
        {
            LearningRate = 1e-2f,
            Epochs = 4,
            GradientAccumulationSteps = 1,
            MaxSteps = 4,
            WarmupRatio = 0f,
            OutputDir = fusedOutDir
        }))
        {
            fusedTrainer.Train(fusedDs);
        }

        string fusedAdapterBin = Path.Combine(fusedOutDir, "adapter_model.bin");
        using var fusedUnmergedLogits = fusedModel.ForwardLogits(tokens);

        VerifyMergeAndKl(fusedBaseGguf, fusedAdapterBin, scope.FilePath("qwen35_fused_merged_f32.gguf"),
            TargetQuantization.F32, tokens, fusedUnmergedLogits.AsSpan(), 1e-4, "FusedGDN-F32");
        VerifyMergeAndKl(fusedBaseGguf, fusedAdapterBin, scope.FilePath("qwen35_fused_merged_q8_0.gguf"),
            TargetQuantization.Q8_0, tokens, fusedUnmergedLogits.AsSpan(), 1e-3, "FusedGDN-Q8_0");
    }

    [Fact]
    [Trait("Tier", "Tier4")]
    [Trait("Feature", "F11+F12+F13")]
    public void T4_03_EndToEndQwen35GdnFineTuneBestCheckpointAndPostTuneEval()
    {
        using var scope = new TempDirectoryScope("t4_03_");
        string baseGguf = scope.FilePath("qwen35_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(baseGguf, CreateQwen35HybridSpec(seed: 606));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");

        var trainDs = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("sentiment: great", "Positive"),
            ("sentiment: awesome", "Positive"),
            ("sentiment: solid", "Positive"),
            ("sentiment: happy", "Positive")
        ]);
        var evalDs = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("sentiment: nice", "Positive")
        ]);

        string outDir = scope.FilePath("run_out");
        var args = new TrainingArguments
        {
            LearningRate = 0.03f,
            Epochs = 6,
            GradientAccumulationSteps = 1,
            EvalSteps = 4,
            WarmupRatio = 0f,
            OutputDir = outDir
        };

        using var trainer = new LoraTrainer(model, args);
        float initValLoss = trainer.Evaluate(evalDs);
        trainer.Train(trainDs, evalDataset: evalDs);

        string bestPath = Path.Combine(outDir, "best_adapter.bin");
        Assert.True(File.Exists(bestPath));
        Assert.NotNull(trainer.BestValidationLoss);
        Assert.True(trainer.BestValidationLoss!.Value < initValLoss);

        string evalJsonl = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            """{"messages":[{"role":"system","content":"Please read carefully and classify sentiment precisely."},{"role":"user","content":"sentiment: nice"},{"role":"assistant","content":"Positive"}]}"""
        ]);

        using var baseModel = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");
        using var bestModel = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");
        bestModel.LoadAdapter(bestPath);

        var report = PostTuneEvaluator.Evaluate(baseModel, bestModel, evalJsonl,
            new EvalOptions { MaxNewTokens = 12, StripSystemTurnForFineTuned = true });

        Assert.Equal(1, report.TotalExamples);
        Assert.True(report.TunedExactMatch >= report.BaseExactMatch);
        Assert.True(report.TunedF1 >= report.BaseF1);
        Assert.True(report.PromptTokenReductionPercent > 0.0);
        Assert.Contains("Fine-Tuned Exact Match", report.FormatSummary());
    }

    [Fact]
    [Trait("Tier", "Tier4")]
    [Trait("Feature", "F12+F11+F9")]
    public void T4_09_Qwen35GdnTrainWithBestCheckpointMergeAndKlRegressionGate()
    {
        using var scope = new TempDirectoryScope("t4_09_");
        string baseGguf = scope.FilePath("qwen35_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(baseGguf, CreateQwen35HybridSpec(seed: 707));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");
        var trainDs = TuneE2EFixtures.CreateGeneralizationTrainDataset(model.Tokenizer);
        var evalDs = TuneE2EFixtures.CreateGeneralizationEvalDataset(model.Tokenizer);

        string outDir = scope.FilePath("out");
        using (var trainer = new LoraTrainer(model, new TrainingArguments
        {
            LearningRate = 1e-2f,
            Epochs = 2,
            GradientAccumulationSteps = 2,
            EvalSteps = 2,
            WarmupRatio = 0f,
            OutputDir = outDir
        }))
        {
            trainer.Train(trainDs, evalDataset: evalDs);
        }

        string bestAdapter = Path.Combine(outDir, "best_adapter.bin");
        Assert.True(File.Exists(bestAdapter));

        string mergedGguf = scope.FilePath("qwen35_best_merged.gguf");
        GgufMerger.Merge(baseGguf, bestAdapter, mergedGguf,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });

        using var bestUnmerged = GgufLoraModel.Load(baseGguf, cfg, device: "cpu");
        bestUnmerged.LoadAdapter(bestAdapter);

        int[] tokens = bestUnmerged.Tokenizer.Encode("<|im_start|>user\ntag: epsilon<|im_end|>\n");
        using var unmergedLogits = bestUnmerged.ForwardLogits(tokens);

        using var mergedSession = new InferenceSession(mergedGguf, maxSeqLen: 128, device: "cpu");
        float[] mergedLogits = new float[tokens.Length * mergedSession.VocabSize];
        mergedSession.VerifyBatchLogits(tokens, 0, mergedLogits);

        var klResult = KlDivergenceVerifier.EvaluateLogits(
            unmergedLogits.AsSpan(),
            mergedLogits,
            mergedSession.VocabSize,
            new KlDivergenceOptions { Threshold = 1e-4, MinTop1Agreement = 1.0 });

        Assert.True(klResult.Passed, $"Merged GGUF KL gate failed: {klResult.FailureReason}");
        Assert.True(klResult.MeanKlDivergence < 1e-4);
        Assert.Equal(1.0, klResult.Top1AgreementRate);
    }
}
