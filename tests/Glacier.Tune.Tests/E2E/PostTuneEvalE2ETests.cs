using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Glacier.Tune.Config;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests.E2E;

[Trait("Suite", "UpgradesE2E")]
public sealed class PostTuneEvalE2ETests
{
    private static SyntheticModelSpec CreateEvalTestSpec() => new()
    {
        Architecture = "llama",
        BlockCount = 1,
        HiddenDim = 32,
        FfnDim = 64,
        HeadCount = 2,
        HeadCountKv = 2,
        HeadDim = 16,
        RopeDim = 16,
        HasQGate = false,
        HasQkNorm = false,
        Seed = 777
    };

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F13")]
    public void T1_F13_01_LoadExamplesParsesPromptMessagesAndBasePromptSchemas()
    {
        using var scope = new TempDirectoryScope("f13_01_");
        string evalPath = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            """{"prompt":"Classify: great","expected":"Positive","base_prompt":"System instructions: Classify sentiment carefully: great"}""",
            """{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}""",
            """{"messages":[{"role":"user","content":"2+2?"}],"expected":"4"}""",
            """{"input":"Summarize","reference":"Done"}"""
        ]);

        var examples = PostTuneEvaluator.LoadExamples(evalPath, new EvalOptions { StripSystemTurnForFineTuned = true });

        Assert.Equal(4, examples.Count);
        Assert.Equal("Positive", examples[0].Expected);
        Assert.Contains("System instructions:", examples[0].BasePrompt);
        Assert.Equal("Classify: great", examples[0].FineTunedPrompt);

        Assert.Equal("Hi", examples[1].Expected);
        Assert.Contains("Be concise.", examples[1].BasePrompt);
        Assert.DoesNotContain("Be concise.", examples[1].FineTunedPrompt);

        Assert.Equal("4", examples[2].Expected);
        Assert.Equal("Done", examples[3].Expected);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F13")]
    public void T1_F13_02_ExactMatchMatchesAnalyticalMean()
    {
        (string Pred, string Exp)[] pairs =
        [
            ("Positive", "Positive"),
            ("the quick brown fox", "the quick red fox"),
            ("alpha beta", "alpha beta gamma delta"),
            ("unrelated", "target")
        ];

        double meanEm = pairs.Average(p => PostTuneEvaluator.ComputeExactMatch(p.Pred, p.Exp));
        Assert.InRange(Math.Abs(meanEm - 0.25), 0.0, 1e-6);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F13")]
    public void T1_F13_03_TokenF1MatchesAnalyticalMean()
    {
        (string Pred, string Exp)[] pairs =
        [
            ("Positive", "Positive"),
            ("the quick brown fox", "the quick red fox"),
            ("alpha beta", "alpha beta gamma delta"),
            ("unrelated", "target")
        ];

        double[] f1s = pairs.Select(p => PostTuneEvaluator.ComputeTokenF1(p.Pred, p.Exp)).ToArray();
        Assert.InRange(Math.Abs(f1s[0] - 1.0), 0.0, 1e-6);
        Assert.InRange(Math.Abs(f1s[1] - 0.75), 0.0, 1e-6);
        Assert.InRange(Math.Abs(f1s[2] - (2.0 / 3.0)), 0.0, 1e-6);
        Assert.InRange(Math.Abs(f1s[3] - 0.0), 0.0, 1e-6);

        double meanF1 = f1s.Average();
        Assert.InRange(Math.Abs(meanF1 - 0.6041666667), 0.0, 1e-4);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F13")]
    public void T1_F13_04_PromptTokenReductionPercentMatchesAnalyticalFormula()
    {
        // Analytical formula: (20 - 5) / 20 * 100 = 75.0%
        double reduction = PostTuneEvaluator.ComputePromptTokenReductionPercent(20, 5);
        Assert.InRange(Math.Abs(reduction - 75.0), 0.0, 1e-4);
        Assert.Equal(0.0, PostTuneEvaluator.ComputePromptTokenReductionPercent(20, 20));

        // End-to-end tokenizer-backed prompt token reduction in PostTuneEvaluator.Evaluate
        using var scope = new TempDirectoryScope("f13_04_");
        string ggufPath = scope.FilePath("model.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateEvalTestSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");

        // Use exact 20-byte ASCII string and 5-byte ASCII string (GPT-2 byte tokenizer encodes 1 ASCII byte = 1 token)
        string basePrompt20 = "12345678901234567890";
        string tunedPrompt5 = "12345";
        Assert.Equal(20, model.Tokenizer.Encode(basePrompt20).Length);
        Assert.Equal(5, model.Tokenizer.Encode(tunedPrompt5).Length);

        string evalPath = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            $$"""{"prompt":"{{tunedPrompt5}}","base_prompt":"{{basePrompt20}}","expected":"OK"}"""
        ]);

        var report = PostTuneEvaluator.Evaluate(model, model, evalPath, new EvalOptions { MaxNewTokens = 2 });
        Assert.Equal(20, report.TotalBasePromptTokens);
        Assert.Equal(5, report.TotalTunedPromptTokens);
        Assert.InRange(Math.Abs(report.PromptTokenReductionPercent - 75.0), 0.0, 1e-4);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F13")]
    public void T1_F13_05_EvaluatorOutputsComparativeBaseVsTunedDeltasAndJson()
    {
        using var scope = new TempDirectoryScope("f13_05_");
        string ggufPath = scope.FilePath("model.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateEvalTestSpec());

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var baseModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
        using var tunedModel = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");

        var trainDs = TuneE2EFixtures.CreateChatMlDataset(tunedModel.Tokenizer,
        [
            ("classify", "OK"),
            ("classify", "OK")
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

        string evalPath = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("eval.jsonl"),
        [
            $$"""{"messages":[{"role":"system","content":"Follow all formatting rules carefully."},{"role":"user","content":"classify"},{"role":"assistant","content":"OK"}]}"""
        ]);

        var report = PostTuneEvaluator.Evaluate(baseModel, tunedModel, evalPath,
            new EvalOptions { MaxNewTokens = 4, StripSystemTurnForFineTuned = true });

        Assert.Equal(1, report.TotalExamples);
        Assert.True(report.TunedExactMatch >= report.BaseExactMatch);
        Assert.True(report.TunedF1 >= report.BaseF1);
        Assert.True(report.TunedExactMatch == 1.0, $"Expected tuned model to learn 'OK', got '{report.Samples[0].TunedPrediction}'.");
        Assert.True(report.PromptTokenReductionPercent > 0.0);

        // Verify JSON & summary formatting are AOT-clean and accurate
        string json = report.ToJson();
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(1, doc.RootElement.GetProperty("total_examples").GetInt32());
        Assert.Equal(report.TunedExactMatch, doc.RootElement.GetProperty("tuned_exact_match").GetDouble(), precision: 6);
        Assert.Equal(report.TunedF1, doc.RootElement.GetProperty("tuned_f1").GetDouble(), precision: 6);
        Assert.Contains("Prompt Token Reduction", report.FormatSummary());
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F13")]
    public void T2_F13_01_EmptyEvalJsonlReturnsZeroSampleReportWithoutNan()
    {
        using var scope = new TempDirectoryScope("f13_t2_01_");
        string ggufPath = scope.FilePath("model.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateEvalTestSpec());

        string emptyPath = scope.FilePath("empty.jsonl");
        File.WriteAllText(emptyPath, "\n   \n");

        var examples = PostTuneEvaluator.LoadExamples(emptyPath);
        Assert.Empty(examples);

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
        var report = PostTuneEvaluator.Evaluate(model, model, emptyPath);

        Assert.Equal(0, report.TotalExamples);
        Assert.Equal(0.0, report.BaseExactMatch);
        Assert.Equal(0.0, report.TunedExactMatch);
        Assert.Equal(0.0, report.BaseF1);
        Assert.Equal(0.0, report.TunedF1);
        Assert.Equal(0.0, report.PromptTokenReductionPercent);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F13")]
    public void T2_F13_02_MalformedJsonLineThrowsInvalidDataExceptionWithLineInfo()
    {
        using var scope = new TempDirectoryScope("f13_t2_02_");
        string badPath = TuneE2EFixtures.WriteEvalJsonl(scope.FilePath("malformed.jsonl"),
        [
            """{"prompt":"ok","expected":"yes"}""",
            """{"prompt":"broken", """
        ]);

        var ex = Assert.Throws<InvalidDataException>(() => PostTuneEvaluator.LoadExamples(badPath));
        Assert.Contains("line 2", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F13")]
    public void T2_F13_03_EmptyExpectedAndPredictionHandleZeroLengthWithoutNan()
    {
        Assert.Equal(1.0, PostTuneEvaluator.ComputeExactMatch("", ""));
        Assert.Equal(1.0, PostTuneEvaluator.ComputeTokenF1("", ""));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch("", "nonempty"));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeTokenF1("", "nonempty"));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch("nonempty", ""));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeTokenF1("nonempty", ""));
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F13")]
    public void T2_F13_04_NormalizationOptionsAndZeroBaseTokensAvoidDivisionByZero()
    {
        Assert.Equal(1.0, PostTuneEvaluator.ComputeExactMatch("  POSITIVE\t\n", "positive", ignoreCase: true, normalizeWhitespace: true));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch("  POSITIVE  ", "positive", ignoreCase: false, normalizeWhitespace: true));
        Assert.Equal(1.0, PostTuneEvaluator.ComputeTokenF1("Alpha   BETA", "alpha beta", ignoreCase: true, normalizeWhitespace: true));

        double zeroBase = PostTuneEvaluator.ComputePromptTokenReductionPercent(0, 5);
        Assert.False(double.IsNaN(zeroBase));
        Assert.Equal(0.0, zeroBase);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F13")]
    public void T2_F13_05_MissingEvalJsonlFileThrowsFileNotFoundException()
    {
        using var scope = new TempDirectoryScope("f13_t2_05_");
        string missing = scope.FilePath("does_not_exist.jsonl");
        Assert.Throws<FileNotFoundException>(() => PostTuneEvaluator.LoadExamples(missing));
    }
}
