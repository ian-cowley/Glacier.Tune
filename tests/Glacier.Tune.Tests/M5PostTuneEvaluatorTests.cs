using System;
using System.IO;
using System.Text.Json;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

public sealed class M5PostTuneEvaluatorTests
{
    private static string CreateTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "glacier_m5_eval_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void ParseExamples_SupportsAllSchemas_AndComputesNormalizedMetrics()
    {
        string jsonl =
            "{\"messages\":[{\"role\":\"system\",\"content\":\"Be concise.\"},{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello!\"}]}\n" +
            "{\"prompt\":\"Q: 2+2\",\"base_prompt\":\"You are a math tutor. Q: 2+2\",\"expected\":\"4\"}\n" +
            "{\"base_messages\":[{\"role\":\"system\",\"content\":\"Verbose rules\"},{\"role\":\"user\",\"content\":\"Go\"}],\"messages\":[{\"role\":\"user\",\"content\":\"Go\"},{\"role\":\"assistant\",\"content\":\"Done\"}]}\n";

        using var sr = new StringReader(jsonl);
        var examples = PostTuneEvaluator.ParseExamples(sr, new EvalOptions { StripSystemTurnForFineTuned = true });

        Assert.Equal(3, examples.Count);
        Assert.Contains("<|im_start|>system", examples[0].BasePrompt);
        Assert.DoesNotContain("<|im_start|>system", examples[0].FineTunedPrompt);
        Assert.Equal("Hello!", examples[0].Expected);

        Assert.Equal("You are a math tutor. Q: 2+2", examples[1].BasePrompt);
        Assert.Equal("Q: 2+2", examples[1].FineTunedPrompt);
        Assert.Equal("4", examples[1].Expected);

        Assert.Contains("Verbose rules", examples[2].BasePrompt);
        Assert.DoesNotContain("Verbose rules", examples[2].FineTunedPrompt);
        Assert.Equal("Done", examples[2].Expected);

        Assert.Equal(1.0, PostTuneEvaluator.ComputeExactMatch("  hello   WORLD ", "Hello world"));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch("hello", "world"));

        // Precision=2/3, Recall=2/2=1.0 => F1=0.8
        double f1 = PostTuneEvaluator.ComputeTokenF1("the cat sat", "cat sat");
        Assert.Equal(0.8, f1, precision: 5);

        Assert.Equal(50.0, PostTuneEvaluator.ComputePromptTokenReductionPercent(100, 50), precision: 5);
    }

    [Fact]
    public void Evaluate_EndToEnd_MeasuresBaseVsFineTunedExactMatchF1AndTokenReduction()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "eval_model.gguf");
            string outDir = Path.Combine(tempDir, "out");
            string trainJsonl = Path.Combine(tempDir, "train.jsonl");
            string evalJsonl = Path.Combine(tempDir, "eval.jsonl");

            // Build a deterministic GGUF with non-zero LM head rows so LoRA can steer hidden states
            // toward "ok<|im_end|>", while biasing the untuned base model toward emitting 'x'.
            var vocab = SyntheticGgufBuilder.BuildGpt2ByteVocab();
            int hiddenDim = 16;
            int vocabSize = vocab.Length;
            var lmRng = new Random(9001);
            float[] outWeights = new float[hiddenDim * vocabSize];
            for (int i = 0; i < outWeights.Length; i++)
            {
                outWeights[i] = (float)(lmRng.NextDouble() * 2.0 - 1.0) * 0.4f;
            }
            int tokX = Array.IndexOf(vocab, "x");
            Assert.True(tokX >= 0);
            for (int d = 0; d < hiddenDim; d++)
            {
                outWeights[tokX * hiddenDim + d] = 1.5f;
            }

            SyntheticGgufBuilder.WriteStandardLlamaGguf(
                ggufPath,
                blockCount: 1,
                hiddenDim: hiddenDim,
                ffnDim: 24,
                headCount: 2,
                headCountKv: 2,
                headDim: 8,
                seed: 5150,
                customOutputWeights: outWeights);

            // Train adapter on "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n" -> "ok<|im_end|>"
            File.WriteAllText(
                trainJsonl,
                "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}\n");

            var loraCfg = new LoraConfig { Rank = 8, Alpha = 32f, Seed = 42 };
            using (var model = GgufLoraModel.Load(ggufPath, loraCfg, device: "cpu"))
            {
                var trainDs = ChatMlDataset.FromFile(trainJsonl, model.Tokenizer, maxSeqLength: 64);
                var trainArgs = new TrainingArguments
                {
                    LearningRate = 2e-2f,
                    Epochs = 40,
                    GradientAccumulationSteps = 1,
                    WarmupRatio = 0f,
                    LoggingSteps = 40,
                    SaveSteps = 0,
                    OutputDir = outDir,
                    Device = "cpu"
                };
                using var trainer = new LoraTrainer(model, trainArgs);
                trainer.Train(trainDs);
            }

            string adapterPath = Path.Combine(outDir, "adapter_model.bin");
            Assert.True(File.Exists(adapterPath));

            // Eval JSONL includes a verbose base_messages system prompt vs compact fine-tuned messages
            File.WriteAllText(
                evalJsonl,
                "{\"base_messages\":[{\"role\":\"system\",\"content\":\"Follow all instructions carefully and concisely.\"},{\"role\":\"user\",\"content\":\"q\"}],\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}\n");

            var report = PostTuneEvaluator.Evaluate(
                ggufPath,
                adapterPath,
                evalJsonl,
                new EvalOptions { MaxNewTokens = 6, MaxSeqLength = 64 },
                device: "cpu");

            Assert.Equal(1, report.TotalExamples);
            Assert.Equal(0.0, report.BaseExactMatch);
            Assert.Equal(1.0, report.TunedExactMatch);
            Assert.Equal(1.0, report.TunedF1, precision: 5);
            Assert.True(report.TunedExactMatch > report.BaseExactMatch);
            Assert.True(report.TunedF1 > report.BaseF1);
            Assert.True(report.PromptTokenReductionPercent > 0.0);
            Assert.True(report.TotalBasePromptTokens > report.TotalTunedPromptTokens);

            string summary = report.FormatSummary();
            Assert.Contains("Fine-Tuned Exact Match:  100.00%", summary);

            string json = report.ToJson();
            using var doc = JsonDocument.Parse(json);
            Assert.Equal(1.0, doc.RootElement.GetProperty("tuned_exact_match").GetDouble(), precision: 5);
            Assert.True(doc.RootElement.GetProperty("prompt_token_reduction_percent").GetDouble() > 0.0);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
