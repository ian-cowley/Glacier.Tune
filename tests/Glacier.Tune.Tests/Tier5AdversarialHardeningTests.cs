using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Export;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

[Trait("Tier", "Tier5")]
public sealed class Tier5AdversarialHardeningTests
{
    private static string CreateTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "glacier_t5_tune_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void Tier5_ClipGradients_HugeFiniteGradientDoesNotOverflowFloatNormSqToZero()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);
            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
            using var trainer = new LoraTrainer(model, new TrainingArguments { OutputDir = Path.Combine(tempDir, "out") });

            model.TrainableParameters[0].Grad!.AsSpan().Fill(1e20f);
            float norm = trainer.ClipGradients(1.0f);

            Assert.True(float.IsFinite(norm) || norm > 1e19f);
            var grad = model.TrainableParameters[0].Grad;
            Assert.NotNull(grad);
            Assert.True(grad.AsSpan()[0] > 0f,
                "Huge finite gradient was zeroed out due to (float)totalNormSq overflow.");
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_ClipGradients_NanGradientDoesNotPoisonOptimizerWeights()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);
            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
            using var trainer = new LoraTrainer(model, new TrainingArguments { OutputDir = Path.Combine(tempDir, "out") });

            model.TrainableParameters[0].Grad!.AsSpan().Fill(float.NaN);
            trainer.ClipGradients(1.0f);
            trainer.Optimizer.Step();

            Assert.True(float.IsFinite(model.TrainableParameters[0].AsSpan()[0]),
                "NaN gradient bypassed ClipGradients and poisoned parameter weights.");
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_LoraTrainer_DisposedTrainerThrowsObjectDisposedExceptionOnTrain()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);
            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
            var trainer = new LoraTrainer(model, new TrainingArguments { OutputDir = Path.Combine(tempDir, "out") });
            trainer.Dispose();

            Assert.Throws<ObjectDisposedException>(() => trainer.Train(new ChatMlDataset()));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_ResumeTraining_SameOutputDir_DoesNotOverwriteBetterBestAdapterWithWorseLoss()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            string outDir = Path.Combine(tempDir, "out");
            string bestPath = Path.Combine(outDir, "best_adapter.bin");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);

            var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
            using var model = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");

            string t1 = Path.Combine(tempDir, "t1.jsonl");
            string e1 = Path.Combine(tempDir, "e1.jsonl");
            File.WriteAllText(t1, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yes\"}]}\n");
            File.WriteAllText(e1, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yes\"}]}\n");
            var ds1 = ChatMlDataset.FromFile(t1, model.Tokenizer, 64);
            var ev1 = ChatMlDataset.FromFile(e1, model.Tokenizer, 64);

            using var trainer1 = new LoraTrainer(model, new TrainingArguments
            {
                LearningRate = 2e-2f, Epochs = 4, GradientAccumulationSteps = 1, EvalSteps = 1, WarmupRatio = 0f, OutputDir = outDir
            });
            trainer1.Train(ds1, evalDataset: ev1);
            byte[] phase1BestHash = SHA256.HashData(File.ReadAllBytes(bestPath));

            string t2 = Path.Combine(tempDir, "t2.jsonl");
            File.WriteAllText(t2,
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"zzzzzzzzzz\"}]}\n" +
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"zzzzzzzzzz\"}]}\n");
            var ds2 = ChatMlDataset.FromFile(t2, model.Tokenizer, 64);

            // Remove Phase 1 optimizer momentum so Phase 2 immediately steps toward the contradicting ds2 ("zzzzzzzzzz")
            // distribution rather than riding Phase 1's residual "yes" momentum to a lower ev1 loss on step 5.
            string optStatePath = Path.Combine(outDir, "optimizer_state.bin");
            if (File.Exists(optStatePath)) File.Delete(optStatePath);

            using var trainer2 = new LoraTrainer(model, new TrainingArguments
            {
                LearningRate = 5e-1f, Epochs = 1, GradientAccumulationSteps = 1, EvalSteps = 1, WarmupRatio = 0f, OutputDir = outDir
            });
            trainer2.ResumeTraining(Path.Combine(outDir, "adapter_model.bin"), ds2, annealedLearningRate: 5e-1f, startingStep: 4, additionalSteps: 2, evalDataset: ev1);

            byte[] phase2BestHash = SHA256.HashData(File.ReadAllBytes(bestPath));
            Assert.Equal(phase1BestHash, phase2BestHash);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_PostTuneEvaluator_GenerateGreedy_PromptAtMaxSeqLengthStillGeneratesOrTruncatesRoomForCompletion()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);
            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");

            string gen = PostTuneEvaluator.GenerateGreedy(model, "hello world!", maxNewTokens: 4, maxSeqLength: 4);
            Assert.NotEmpty(gen);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_PostTuneEvaluator_NormalizeText_PreserveWhitespaceWhenNormalizeWhitespaceFalse()
    {
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch("hello ", "hello", ignoreCase: true, normalizeWhitespace: false));
        Assert.Equal(0.0, PostTuneEvaluator.ComputeExactMatch(" hello", "hello", ignoreCase: true, normalizeWhitespace: false));
        Assert.Equal(1.0, PostTuneEvaluator.ComputeExactMatch("HELLO ", "hello ", ignoreCase: true, normalizeWhitespace: false));
    }

    [Fact]
    public void Tier5_LoadAdapter_ValidatesRankMismatchAndPropagatesSavedAlphaScaling()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            string outDir = Path.Combine(tempDir, "out");
            SyntheticGgufBuilder.WriteStandardLlamaGguf(ggufPath, blockCount: 1, hiddenDim: 16, ffnDim: 24, headCount: 2, headCountKv: 2, headDim: 8);

            using (var saveModel = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 32f, Seed = 42 }, device: "cpu"))
            {
                saveModel.Blocks[0].QProj.AdapterB.Fill(0.25f);
                saveModel.SaveAdapter(outDir);
            }

            string adapterBin = Path.Combine(outDir, "adapter_model.bin");

            // Rank mismatch (4 -> 8) must throw InvalidDataException instead of silently loading 0 tensors
            using (var wrongRankModel = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 8, Alpha = 32f }, device: "cpu"))
            {
                Assert.Throws<InvalidDataException>(() => wrongRankModel.LoadAdapter(adapterBin));
            }

            // Alpha mismatch (32 -> 8) must propagate saved Alpha=32 so Scaling == 32 / 4 == 8
            using var diffAlphaModel = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
            diffAlphaModel.LoadAdapter(adapterBin);
            Assert.Equal(32f, diffAlphaModel.Config.Alpha);
            Assert.Equal(8f, diffAlphaModel.Blocks[0].QProj.Scaling, precision: 5);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_TransformerBlock_MissingRequiredBaseWeights_ThrowsInvalidDataExceptionAtLoad()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "missing_ffn_gate.gguf");
            WriteLlamaGgufMissingFfnGate(ggufPath);

            Assert.Throws<InvalidDataException>(() =>
                GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu"));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_LoraTrainer_EmptyDatasetAndZeroRankConfig_CompletesWithoutNanOrCrash()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "model.gguf");
            string outDir = Path.Combine(tempDir, "out_zero");
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2, HiddenDim = 16, FfnDim = 24, HeadCount = 2, HeadCountKv = 1, HeadDim = 8, RopeDim = 4,
                FullAttentionInterval = 2, SsmConvKernel = 4, SsmStateSize = 4, SsmGroupCount = 2, SsmTimeStepRank = 4
            });

            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 0, Alpha = 8f }, device: "cpu");
            Assert.Empty(model.TrainableParameters);

            using var trainer = new LoraTrainer(model, new TrainingArguments
            {
                Epochs = 2, EvalSteps = 1, GradientAccumulationSteps = 1, OutputDir = outDir
            });
            trainer.Train(new ChatMlDataset(), evalDataset: new ChatMlDataset());

            Assert.Null(trainer.BestValidationLoss);
            Assert.True(File.Exists(Path.Combine(outDir, "adapter_model.bin")));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_GdnMerger_Q4_0QuantizationAndRelativeOutputFilename_Succeeds()
    {
        string tempDir = CreateTempDir();
        try
        {
            string baseGguf = Path.Combine(tempDir, "qwen35.gguf");
            string outDir = Path.Combine(tempDir, "ckpt");
            string mergedPath = Path.Combine(tempDir, "qwen35_q4_0.gguf");
            SyntheticGgufBuilder.WriteQwen35HybridGguf(baseGguf, new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2, HiddenDim = 32, FfnDim = 64, HeadCount = 2, HeadCountKv = 1, HeadDim = 16, RopeDim = 8,
                FullAttentionInterval = 2, SsmConvKernel = 4, SsmStateSize = 8, SsmGroupCount = 2, SsmTimeStepRank = 4
            });

            using var model = GgufLoraModel.Load(baseGguf, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
            string trainPath = Path.Combine(tempDir, "train.jsonl");
            File.WriteAllText(trainPath, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}\n");
            var ds = ChatMlDataset.FromFile(trainPath, model.Tokenizer, 64);

            using (var trainer = new LoraTrainer(model, new TrainingArguments { LearningRate = 1e-2f, Epochs = 2, GradientAccumulationSteps = 1, OutputDir = outDir }))
            {
                trainer.Train(ds);
            }

            var res = GgufMerger.Merge(baseGguf, Path.Combine(outDir, "adapter_model.bin"), mergedPath,
                new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q4_0 });
            Assert.True(res.AdaptedTensorCount > 0);
            Assert.True(res.OutputSizeBytes > 0);

            using var mergedModel = GgufLoraModel.Load(mergedPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");
            using var logits = mergedModel.ForwardLogits([5, 6, 7]);
            foreach (float v in logits.AsSpan()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_PostTuneEvaluator_F1MultisetDuplicates_And_NegativeReduction_And_ToJson()
    {
        Assert.Equal(0.5, PostTuneEvaluator.ComputeTokenF1("a a a", "a"), precision: 6);
        Assert.Equal(1.0, PostTuneEvaluator.ComputeTokenF1("   \t\n", ""));
        Assert.Equal(1.0, PostTuneEvaluator.ComputeExactMatch("   \t\n", ""));
        Assert.Equal(-100.0, PostTuneEvaluator.ComputePromptTokenReductionPercent(10, 20), precision: 6);
        Assert.Equal(0.0, PostTuneEvaluator.ComputePromptTokenReductionPercent(0, 10), precision: 6);
    }

    private static void WriteLlamaGgufMissingFfnGate(string path)
    {
        const int dim = 16, ffnDim = 24, nHeads = 2, nHeadsKv = 2, headDim = 8, vocab = 256;
        var tensors = new System.Collections.Generic.List<(string Name, GgufType Type, ulong[] Dims, byte[] Data)>();

        void AddF32(string name, params ulong[] dims)
        {
            ulong elems = 1;
            foreach (var d in dims) elems *= d;
            float[] f = new float[elems];
            Array.Fill(f, 0.01f);
            byte[] data = new byte[elems * 4];
            Buffer.BlockCopy(f, 0, data, 0, data.Length);
            tensors.Add((name, GgufType.F32, dims, data));
        }

        AddF32("token_embd.weight", (ulong)dim, (ulong)vocab);
        AddF32("output_norm.weight", (ulong)dim);
        AddF32("output.weight", (ulong)dim, (ulong)vocab);
        AddF32("blk.0.attn_norm.weight", (ulong)dim);
        AddF32("blk.0.attn_q.weight", (ulong)dim, (ulong)(nHeads * headDim));
        AddF32("blk.0.attn_k.weight", (ulong)dim, (ulong)(nHeadsKv * headDim));
        AddF32("blk.0.attn_v.weight", (ulong)dim, (ulong)(nHeadsKv * headDim));
        AddF32("blk.0.attn_output.weight", (ulong)(nHeads * headDim), (ulong)dim);
        AddF32("blk.0.ffn_norm.weight", (ulong)dim);
        // Intentionally omit blk.0.ffn_gate.weight
        AddF32("blk.0.ffn_up.weight", (ulong)dim, (ulong)ffnDim);
        AddF32("blk.0.ffn_down.weight", (ulong)ffnDim, (ulong)dim);

        var vocabTokens = Enumerable.Range(0, vocab).Select(i => i == 0 ? "<|endoftext|>" : $"t{i}").ToArray();
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
        void WriteStr(string s) { byte[] b = Encoding.UTF8.GetBytes(s); bw.Write((ulong)b.Length); bw.Write(b); }
        void Align32() { long m = fs.Position % 32; if (m != 0) bw.Write(new byte[32 - m]); }

        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write(10UL);

        void WriteKvStr(string k, string v) { WriteStr(k); bw.Write((uint)GgufValueType.String); WriteStr(v); }
        void WriteKvU32(string k, uint v) { WriteStr(k); bw.Write((uint)GgufValueType.Uint32); bw.Write(v); }
        void WriteKvStrArr(string k, string[] v)
        {
            WriteStr(k);
            bw.Write((uint)GgufValueType.Array);
            bw.Write((uint)GgufValueType.String);
            bw.Write((ulong)v.Length);
            foreach (var s in v) WriteStr(s);
        }

        WriteKvStr("general.architecture", "llama");
        WriteKvU32("llama.block_count", 1u);
        WriteKvU32("llama.embedding_length", (uint)dim);
        WriteKvU32("llama.feed_forward_length", (uint)ffnDim);
        WriteKvU32("llama.attention.head_count", (uint)nHeads);
        WriteKvU32("llama.attention.head_count_kv", (uint)nHeadsKv);
        WriteKvU32("llama.attention.key_length", (uint)headDim);
        WriteKvU32("llama.context_length", 64u);
        WriteKvStrArr("tokenizer.ggml.tokens", vocabTokens);
        WriteKvStrArr("tokenizer.ggml.merges", Array.Empty<string>());

        ulong off = 0;
        foreach (var t in tensors)
        {
            WriteStr(t.Name);
            bw.Write((uint)t.Dims.Length);
            foreach (var d in t.Dims) bw.Write(d);
            bw.Write((uint)t.Type);
            bw.Write(off);
            ulong sz = (ulong)t.Data.Length;
            off += sz;
            if (off % 32 != 0) off += 32 - (off % 32);
        }
        Align32();
        foreach (var t in tensors) { bw.Write(t.Data); Align32(); }
    }
}
