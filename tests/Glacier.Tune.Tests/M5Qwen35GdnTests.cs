using System;
using System.IO;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Export;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

public sealed class M5Qwen35GdnTests
{
    private static string CreateTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "glacier_m5_gdn_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public unsafe void Qwen35Hybrid_ForwardLogits_MatchesInferenceQwen3HybridModel_AtZeroLora()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "qwen35_hybrid.gguf");
            var spec = new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2,
                HiddenDim = 16,
                FfnDim = 24,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 8,
                RopeDim = 4,
                FullAttentionInterval = 2,
                SsmConvKernel = 4,
                SsmStateSize = 4,
                SsmGroupCount = 2,
                SsmTimeStepRank = 4,
                GdnOnEvenLayers = true,
                UseQGate = true,
                UseQkNorm = true,
                Seed = 2026
            };
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, spec);

            int[] tokens = [5, 8, 12, 19, 25, 33];

            using var tuneModel = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "auto");
            Assert.True(tuneModel.IsHybrid);
            Assert.Single(tuneModel.GdnBlocks);
            Assert.Single(tuneModel.Blocks);
            Assert.False(tuneModel.IsGpuAccelerated);

            using var tuneLogits = tuneModel.ForwardLogits(tokens);
            using var tuneLastLogits = tuneModel.ForwardLastTokenLogits(tokens);

            using var infModel = new Qwen3HybridModel(tuneModel.Weights, maxSeqLen: 64);
            using var kvCache = new KVCache(tuneModel.Weights.BlockCount, tuneModel.Weights.HeadCountKv, tuneModel.Weights.HeadDim, maxSeqLen: 64, precision: KvCachePrecision.Fp32);
            int vocabSize = tuneModel.VocabSize;
            float[] infLogits = new float[tokens.Length * vocabSize];
            infModel.VerifyBatchLogits(tokens, 0, infLogits, vocabSize, kvCache);

            float* pTune = (float*)tuneLogits.DataPointer;
            float maxDiff = 0f;
            for (int i = 0; i < infLogits.Length; i++)
            {
                float diff = MathF.Abs(pTune[i] - infLogits[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            Assert.True(maxDiff < 1e-4f, $"Max logit mismatch vs Qwen3HybridModel: {maxDiff}");

            // Also verify ForwardLastTokenLogits matches final row of ForwardLogits
            float* pLast = (float*)tuneLastLogits.DataPointer;
            float* pLastFromFull = pTune + (long)(tokens.Length - 1) * vocabSize;
            for (int v = 0; v < vocabSize; v++)
            {
                Assert.True(MathF.Abs(pLast[v] - pLastFromFull[v]) < 1e-5f);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    public unsafe void GdnBlock_Backward_MatchesFiniteDifferenceGradients_AcrossAllAdapters(string architecture)
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, $"gdn_{architecture}.gguf");
            var spec = new SyntheticGgufBuilder.Qwen35Spec
            {
                Architecture = architecture,
                BlockCount = 1,
                HiddenDim = 8,
                FfnDim = 12,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 4,
                RopeDim = 2,
                SsmConvKernel = 4,
                SsmStateSize = 4,
                SsmGroupCount = 2,
                SsmTimeStepRank = 4,
                GdnOnEvenLayers = true,
                Seed = 31415
            };
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, spec);

            var config = new LoraConfig { Rank = 2, Alpha = 4f, Seed = 99 };
            using var model = GgufLoraModel.Load(ggufPath, config, device: "cpu");
            Assert.Single(model.GdnBlocks);

            // Perturb AdapterB non-zero so gradients flow through both A and B (and downstream ops)
            var rng = new Random(123);
            foreach (var p in model.TrainableParameters)
            {
                var sp = p.AsSpan();
                for (int i = 0; i < sp.Length; i++)
                {
                    sp[i] += (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
                }
            }

            int[] inputTokens = [5, 7, 11, 15, 21];
            int[] targets = [-100, 7, 11, 15, 2];

            // Zero grads and compute analytical gradients
            foreach (var p in model.TrainableParameters)
            {
                p.Grad!.AsSpan().Clear();
            }
            float baseLoss = model.ForwardLossAndBackward(inputTokens, targets);
            Assert.True(float.IsFinite(baseLoss));

            var gdn = model.GdnBlocks[0];
            LoraAdapter[] adapters =
            [
                gdn.QkvProj, gdn.AttnGateProj, gdn.AlphaProj, gdn.BetaProj,
                gdn.SsmOutProj, gdn.GateProj, gdn.UpProj, gdn.DownProj
            ];

            const float eps = 1e-3f;
            foreach (var adapter in adapters)
            {
                Assert.True(adapter.IsEnabled);
                VerifyFiniteDifference(model, inputTokens, targets, adapter.AdapterA, eps);
                VerifyFiniteDifference(model, inputTokens, targets, adapter.AdapterB, eps);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void FullAttention_QGate_QkNorm_PartialRoPE_Backward_MatchesFiniteDifferenceGradients()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "qgate_attn.gguf");
            var spec = new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 1,
                HiddenDim = 8,
                FfnDim = 12,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 8,
                RopeDim = 4, // Partial RoPE (4 out of 8)
                GdnOnEvenLayers = false,
                UseQGate = true,
                UseQkNorm = true,
                Seed = 4242
            };
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, spec);

            using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 2, Alpha = 4f, Seed = 55 }, device: "cpu");
            Assert.Single(model.Blocks);
            Assert.Empty(model.GdnBlocks);

            var rng = new Random(456);
            foreach (var p in model.TrainableParameters)
            {
                var sp = p.AsSpan();
                for (int i = 0; i < sp.Length; i++)
                {
                    sp[i] += (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
                }
            }

            int[] inputTokens = [5, 9, 14, 20];
            int[] targets = [-100, 9, 14, 2];

            foreach (var p in model.TrainableParameters) p.Grad!.AsSpan().Clear();
            model.ForwardLossAndBackward(inputTokens, targets);

            var attn = model.Blocks[0];
            LoraAdapter[] adapters =
            [
                attn.QProj, attn.KProj, attn.VProj, attn.OProj,
                attn.GateProj, attn.UpProj, attn.DownProj
            ];

            const float eps = 1e-3f;
            foreach (var adapter in adapters)
            {
                Assert.True(adapter.IsEnabled);
                VerifyFiniteDifference(model, inputTokens, targets, adapter.AdapterA, eps);
                VerifyFiniteDifference(model, inputTokens, targets, adapter.AdapterB, eps);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void Qwen35Hybrid_TrainSaveLoadAndMerge_ProducesLogitParity(bool useFusedGdnTensors)
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "qwen35_base.gguf");
            string mergedGgufPath = Path.Combine(tempDir, "qwen35_merged.gguf");
            string outDir = Path.Combine(tempDir, "out");
            string trainPath = Path.Combine(tempDir, "train.jsonl");

            var spec = new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2,
                HiddenDim = 16,
                FfnDim = 24,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 8,
                RopeDim = 4,
                FullAttentionInterval = 2,
                SsmConvKernel = 4,
                SsmStateSize = 4,
                SsmGroupCount = 2,
                SsmTimeStepRank = 4,
                GdnOnEvenLayers = true,
                UseQGate = true,
                UseQkNorm = true,
                UseFusedGdnTensors = useFusedGdnTensors,
                Seed = 8888
            };
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, spec);

            File.WriteAllText(
                trainPath,
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":\"world\"}]}\n" +
                "{\"messages\":[{\"role\":\"user\",\"content\":\"cat\"},{\"role\":\"assistant\",\"content\":\"dog\"}]}\n");

            var loraCfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 77 };
            using var model = GgufLoraModel.Load(ggufPath, loraCfg, device: "cpu");
            var dataset = ChatMlDataset.FromFile(trainPath, model.Tokenizer, maxSeqLength: 64);

            var trainArgs = new TrainingArguments
            {
                LearningRate = 5e-3f,
                Epochs = 6,
                GradientAccumulationSteps = 1,
                LoggingSteps = 20,
                SaveSteps = 0,
                WarmupRatio = 0f,
                OutputDir = outDir,
                Device = "cpu"
            };

            float initialEvalLoss;
            float finalEvalLoss;
            using (var trainer = new LoraTrainer(model, trainArgs))
            {
                initialEvalLoss = trainer.Evaluate(dataset);
                trainer.Train(dataset);
                finalEvalLoss = trainer.Evaluate(dataset);
            }

            Assert.True(finalEvalLoss < initialEvalLoss, $"Expected dataset loss to decrease on Qwen 3.5 GDN hybrid, got initial={initialEvalLoss}, final={finalEvalLoss}");
            Assert.False(model.GdnBlocks[0].QkvProj.IsZeroDelta);
            Assert.False(model.GdnBlocks[0].SsmOutProj.IsZeroDelta);

            string adapterBin = Path.Combine(outDir, "adapter_model.bin");
            Assert.True(File.Exists(adapterBin));

            int[] probeTokens = [5, 9, 13, 17, 21];
            using var tunedLogits = model.ForwardLogits(probeTokens);

            // Reload into fresh model and check exact logit parity
            using var reloadedModel = GgufLoraModel.Load(ggufPath, loraCfg, device: "cpu");
            reloadedModel.LoadAdapter(adapterBin);
            using var reloadedLogits = reloadedModel.ForwardLogits(probeTokens);

            float* pTuned = (float*)tunedLogits.DataPointer;
            float* pReloaded = (float*)reloadedLogits.DataPointer;
            for (int i = 0; i < (int)tunedLogits.ElementCount; i++)
            {
                Assert.Equal(pTuned[i], pReloaded[i], precision: 5);
            }

            // Merge adapter into GGUF and verify merged base GGUF matches tuned LoRA logits
            var mergeRes = GgufMerger.Merge(ggufPath, adapterBin, mergedGgufPath, new GgufMerger.MergeOptions
            {
                TargetQuantization = TargetQuantization.F32
            });
            Assert.True(mergeRes.AdaptedTensorCount > 0);
            Assert.True(File.Exists(mergedGgufPath));

            using var mergedModel = GgufLoraModel.Load(mergedGgufPath, loraCfg, device: "cpu");
            using var mergedLogits = mergedModel.ForwardLogits(probeTokens);
            float* pMerged = (float*)mergedLogits.DataPointer;
            float maxDiff = 0f;
            for (int i = 0; i < (int)tunedLogits.ElementCount; i++)
            {
                float diff = MathF.Abs(pTuned[i] - pMerged[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            Assert.True(maxDiff < 1e-4f, $"Max logit diff after GgufMerger.Merge: {maxDiff}");
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public unsafe void Qwen35Hybrid_CheckpointActivations_MatchesUncheckpointed_AndFusedFullAttnQkvMergesCleanly()
    {
        string tempDir = CreateTempDir();
        try
        {
            string ggufPath = Path.Combine(tempDir, "qwen35_fused_qkv.gguf");
            string mergedGgufPath = Path.Combine(tempDir, "qwen35_fused_qkv_merged.gguf");
            string outDir = Path.Combine(tempDir, "out");

            var spec = new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2,
                HiddenDim = 16,
                FfnDim = 24,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 8,
                RopeDim = 4,
                FullAttentionInterval = 2,
                SsmConvKernel = 4,
                SsmStateSize = 4,
                SsmGroupCount = 2,
                SsmTimeStepRank = 4,
                GdnOnEvenLayers = true,
                UseQGate = true,
                UseQkNorm = true,
                UseFusedFullAttnQkv = true,
                Seed = 9999
            };
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, spec);

            int[] inputTokens = [5, 7, 11, 15];
            int[] targets = [-100, 7, 11, 2];

            var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 33 };
            using var modelNoCkpt = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");
            using var modelWithCkpt = GgufLoraModel.Load(ggufPath, cfg, device: "cpu");

            // Perturb AdapterB identically on both models so all adapter gradients are active
            for (int p = 0; p < modelNoCkpt.TrainableParameters.Count; p++)
            {
                var s1 = modelNoCkpt.TrainableParameters[p].AsSpan();
                var s2 = modelWithCkpt.TrainableParameters[p].AsSpan();
                for (int i = 0; i < s1.Length; i++)
                {
                    float v = 0.05f * ((i % 5) - 2);
                    s1[i] += v;
                    s2[i] += v;
                }
            }

            float lossNoCkpt = modelNoCkpt.ForwardLossAndBackward(inputTokens, targets, checkpointActivations: false);
            float lossWithCkpt = modelWithCkpt.ForwardLossAndBackward(inputTokens, targets, checkpointActivations: true);
            Assert.Equal(lossNoCkpt, lossWithCkpt, precision: 5);

            for (int p = 0; p < modelNoCkpt.TrainableParameters.Count; p++)
            {
                var g1 = modelNoCkpt.TrainableParameters[p].Grad!.AsSpan();
                var g2 = modelWithCkpt.TrainableParameters[p].Grad!.AsSpan();
                for (int i = 0; i < g1.Length; i++)
                {
                    Assert.Equal(g1[i], g2[i], precision: 4);
                }
            }

            // Step optimizer on modelWithCkpt, save adapter, merge fused attn_qkv.weight, and check logit parity
            using var trainer = new LoraTrainer(modelWithCkpt, new TrainingArguments { LearningRate = 1e-2f, OutputDir = outDir, Device = "cpu" });
            trainer.Optimizer.Step();
            modelWithCkpt.SaveAdapter(outDir);

            string adapterBin = Path.Combine(outDir, "adapter_model.bin");
            var mergeRes = GgufMerger.Merge(ggufPath, adapterBin, mergedGgufPath, new GgufMerger.MergeOptions
            {
                TargetQuantization = TargetQuantization.F32
            });
            Assert.True(mergeRes.AdaptedTensorCount > 0);

            using var tunedLogits = modelWithCkpt.ForwardLogits(inputTokens);
            using var mergedModel = GgufLoraModel.Load(mergedGgufPath, cfg, device: "cpu");
            using var mergedLogits = mergedModel.ForwardLogits(inputTokens);
            float* pTuned = (float*)tunedLogits.DataPointer;
            float* pMerged = (float*)mergedLogits.DataPointer;
            for (int i = 0; i < (int)tunedLogits.ElementCount; i++)
            {
                Assert.True(MathF.Abs(pTuned[i] - pMerged[i]) < 1e-4f);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    private static void VerifyFiniteDifference(
        GgufLoraModel model,
        int[] inputTokens,
        int[] targets,
        Tensor<float> param,
        float eps)
    {
        var data = param.AsSpan();
        var grad = param.Grad!.AsSpan();
        int step = Math.Max(1, data.Length / 4); // Sample representative indices across rows/cols for fast rigorous checks

        for (int idx = 0; idx < data.Length; idx += step)
        {
            float orig = data[idx];

            data[idx] = orig + eps;
            float lossPlus = model.EvaluateLoss(inputTokens, targets);

            data[idx] = orig - eps;
            float lossMinus = model.EvaluateLoss(inputTokens, targets);

            data[idx] = orig;

            float numerical = (lossPlus - lossMinus) / (2.0f * eps);
            float analytical = grad[idx];
            float absErr = MathF.Abs(numerical - analytical);
            float tol = 5e-3f + 2e-2f * MathF.Max(MathF.Abs(numerical), MathF.Abs(analytical));
            Assert.True(absErr <= tol, $"Grad mismatch at idx {idx}: numerical={numerical:F6}, analytical={analytical:F6}, absErr={absErr:F6}");
        }
    }
}
