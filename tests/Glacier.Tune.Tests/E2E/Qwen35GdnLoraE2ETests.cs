using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Export;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests.E2E;

[Trait("Suite", "UpgradesE2E")]
public sealed class Qwen35GdnLoraE2ETests
{
    private static SyntheticModelSpec CreateQwen35HybridSpec(
        bool hasQGate = true,
        bool hasQkNorm = true,
        int ropeDim = 8,
        bool includeConv1d = true,
        bool corruptConv1d = false,
        bool fuseGdnTensors = false,
        int seed = 1337) => new()
    {
        Architecture = "qwen35",
        BlockCount = 2,
        FullAttentionInterval = 2,
        HiddenDim = 32,
        FfnDim = 64,
        HeadCount = 2,
        HeadCountKv = 2,
        HeadDim = 16,
        RopeDim = ropeDim,
        ConvKernel = 4,
        StateSize = 8,
        GroupCount = 2,
        SsmHeads = 4,
        HasQGate = hasQGate,
        HasQkNorm = hasQkNorm,
        IncludeSsmConv1d = includeConv1d,
        CorruptSsmConv1dShape = corruptConv1d,
        FuseGdnTensors = fuseGdnTensors,
        Seed = seed
    };

    private static float MaxAbsGrad(Tensor<float>? grad)
    {
        if (grad == null) return 0f;
        var span = grad.AsSpan();
        float max = 0f;
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            Assert.True(float.IsFinite(v), $"Non-finite gradient detected at index {i}: {v}");
            float abs = MathF.Abs(v);
            if (abs > max) max = abs;
        }
        return max;
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F12")]
    public unsafe void T1_F12_01_GgufLoraModelLoadsQwen35GdnHybridWithQGateQkNormPartialRope()
    {
        using var scope = new TempDirectoryScope("f12_01_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var model = GgufLoraModel.Load(ggufPath, loraConfig, device: "cpu");

        Assert.True(model.IsHybrid);
        Assert.Equal(2, model.LayerCount);
        Assert.Single(model.GdnBlocks);
        Assert.Single(model.Blocks);
        Assert.Equal(0, model.GdnBlocks[0].LayerIndex);
        Assert.Equal(1, model.Blocks[0].LayerIndex);

        Assert.True(model.Weights.IsHybridSsm);
        Assert.True(model.Weights.Layers[0].IsGdn);
        Assert.False(model.Weights.Layers[1].IsGdn);
        Assert.True(model.Weights.Layers[1].HasQGate);
        Assert.True(model.Weights.Layers[1].AttnQNormWeight != null);
        Assert.True(model.Weights.Layers[1].AttnKNormWeight != null);
        Assert.Equal(8, model.Weights.RopeDimensionCount);

        // Layer 1 Q projection with HasQGate doubles outFeatures (2 * qDim = 64)
        Assert.Equal(64, model.Blocks[0].QProj.OutFeatures);
        Assert.True(model.GdnBlocks[0].QkvProj.IsEnabled);
        Assert.True(model.GdnBlocks[0].SsmOutProj.IsEnabled);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F12")]
    public void T1_F12_02_PureCSharpGdnForwardPassProducesFiniteLogits()
    {
        using var scope = new TempDirectoryScope("f12_02_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        int[] tokens = model.Tokenizer.Encode("<|im_start|>user\nhello gdn<|im_end|>\n");

        using var logits = model.ForwardLogits(tokens);
        Assert.Equal(2, logits.Rank);
        Assert.Equal(tokens.Length, logits.Shape[0]);
        Assert.Equal(model.VocabSize, logits.Shape[1]);

        var span = logits.AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < span.Length; i++)
        {
            Assert.True(float.IsFinite(span[i]), $"Non-finite logit at index {i}: {span[i]}");
            if (span[i] != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "Expected non-trivial logits from Qwen 3.5 GDN forward pass.");
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F12")]
    public void T1_F12_03_PureCSharpBpttComputesNonZeroFiniteGradientsOnGdnAndQGateLora()
    {
        using var scope = new TempDirectoryScope("f12_03_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        var ds = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer, [("hi", "ok")]);
        var example = ds.Examples[0];

        // Step 1 with MaxSteps = 2 ensures cosine LR > 0 on step 1, updating AdapterB away from zero so subsequent backward produces non-zero dA and dB.
        using (var trainer = new LoraTrainer(model, new TrainingArguments
        {
            LearningRate = 1e-2f,
            Epochs = 2,
            GradientAccumulationSteps = 1,
            MaxSteps = 2,
            WarmupRatio = 0f,
            OutputDir = scope.FilePath("step1")
        }))
        {
            trainer.Train(ds);
        }

        // Zero grads and run backward on Step 2 to inspect exact gradients on both Layer 0 (GDN) and Layer 1 (QGate)
        foreach (var p in model.TrainableParameters)
        {
            p.Grad?.AsSpan().Clear();
        }

        float loss = model.ForwardLossAndBackward(example.InputIds, example.Labels);
        Assert.True(float.IsFinite(loss) && loss > 0f);

        var gdn = model.GdnBlocks[0];
        (string Name, LoraAdapter Adapter)[] gdnAdapters =
        [
            ("attn_qkv", gdn.QkvProj),
            ("attn_gate", gdn.AttnGateProj),
            ("ssm_alpha", gdn.AlphaProj),
            ("ssm_beta", gdn.BetaProj),
            ("ssm_out", gdn.SsmOutProj),
            ("ffn_gate", gdn.GateProj),
            ("ffn_up", gdn.UpProj),
            ("ffn_down", gdn.DownProj)
        ];
        foreach (var (name, adapter) in gdnAdapters)
        {
            Assert.True(MaxAbsGrad(adapter.AdapterA.Grad) > 0f, $"GDN {name} A grad should be non-zero.");
            Assert.True(MaxAbsGrad(adapter.AdapterB.Grad) > 0f, $"GDN {name} B grad should be non-zero.");
        }

        var attn = model.Blocks[0];
        (string Name, LoraAdapter Adapter)[] attnAdapters =
        [
            ("attn_q", attn.QProj),
            ("attn_k", attn.KProj),
            ("attn_v", attn.VProj),
            ("attn_output", attn.OProj),
            ("ffn_gate", attn.GateProj),
            ("ffn_up", attn.UpProj),
            ("ffn_down", attn.DownProj)
        ];
        foreach (var (name, adapter) in attnAdapters)
        {
            Assert.True(MaxAbsGrad(adapter.AdapterA.Grad) > 0f, $"HasQGate {name} A grad should be non-zero.");
            Assert.True(MaxAbsGrad(adapter.AdapterB.Grad) > 0f, $"HasQGate {name} B grad should be non-zero.");
        }

        // Lightweight central finite-difference gradient parity check on GdnBlock + HasQGate TransformerBlock
        const int blockSeqLen = 5;
        using var xBlock = TensorFloatExtensions.RandomNormal([blockSeqLen, model.HiddenDim], 0f, 0.25f, seed: 101);
        using var dOutBlock = TensorFloatExtensions.RandomNormal([blockSeqLen, model.HiddenDim], 0f, 0.25f, seed: 102);

        static float DotProduct(Tensor<float> a, Tensor<float> b)
        {
            var sa = a.AsSpan();
            var sb = b.AsSpan();
            double sum = 0.0;
            for (int i = 0; i < sa.Length; i++) sum += (double)sa[i] * sb[i];
            return (float)sum;
        }

        static void AssertCentralFiniteDiffParity(string label, Tensor<float> param, Func<float> evalScalarLoss)
        {
            const float eps = 1e-3f;
            var gradSpan = param.Grad!.AsSpan();
            int bestIdx = 0;
            float bestAbs = 0f;
            for (int i = 0; i < gradSpan.Length; i++)
            {
                float abs = MathF.Abs(gradSpan[i]);
                if (abs > bestAbs)
                {
                    bestAbs = abs;
                    bestIdx = i;
                }
            }
            Assert.True(bestAbs > 1e-5f, $"{label}: expected non-trivial analytical gradient, got max={bestAbs}");

            float analytic = gradSpan[bestIdx];
            var pSpan = param.AsSpan();
            float orig = pSpan[bestIdx];
            pSpan[bestIdx] = orig + eps;
            float lossPlus = evalScalarLoss();
            pSpan[bestIdx] = orig - eps;
            float lossMinus = evalScalarLoss();
            pSpan[bestIdx] = orig;

            float numeric = (lossPlus - lossMinus) / (2f * eps);
            float absErr = MathF.Abs(analytic - numeric);
            float denom = MathF.Max(1e-6f, MathF.Max(MathF.Abs(analytic), MathF.Abs(numeric)));
            float relErr = absErr / denom;
            Assert.True(absErr < 1e-4f || relErr < 2e-2f,
                $"{label} finite-diff mismatch: analytic={analytic:E4}, numeric={numeric:E4}, absErr={absErr:E3}, relErr={relErr:E3}");
            if (MathF.Abs(analytic) > 1e-2f)
            {
                Assert.True(relErr < 3e-2f,
                    $"{label} relative finite-diff mismatch: analytic={analytic:E4}, numeric={numeric:E4}, relErr={relErr:E3}");
            }
        }

        // GdnBlock BPTT finite-difference check across all 8 GDN adapters
        foreach (var p in model.TrainableParameters) p.Grad?.AsSpan().Clear();
        using (var yGdn = gdn.Forward(xBlock, blockSeqLen, saveActivations: true, out var gdnState))
        using (var dXGdn = gdn.Backward(dOutBlock, gdnState!, xBlock, blockSeqLen))
        {
            gdnState!.Dispose();
            Assert.True(MaxAbsGrad(dXGdn) > 0f);
        }

        float EvalGdnScalar()
        {
            using var y = gdn.Forward(xBlock, blockSeqLen, saveActivations: false, out _);
            return DotProduct(y, dOutBlock);
        }

        foreach (var (name, adapter) in gdnAdapters)
        {
            AssertCentralFiniteDiffParity($"GDN.{name}.A", adapter.AdapterA, EvalGdnScalar);
            AssertCentralFiniteDiffParity($"GDN.{name}.B", adapter.AdapterB, EvalGdnScalar);
        }

        // HasQGate + QK-RMSNorm + partial-RoPE TransformerBlock finite-difference check across all 7 adapters
        foreach (var p in model.TrainableParameters) p.Grad?.AsSpan().Clear();
        using (var yAttn = attn.Forward(xBlock, blockSeqLen, saveActivations: true, out var attnState))
        using (var dXAttn = attn.Backward(dOutBlock, attnState!, xBlock, blockSeqLen))
        {
            attnState!.Dispose();
            Assert.True(MaxAbsGrad(dXAttn) > 0f);
        }

        float EvalAttnScalar()
        {
            using var y = attn.Forward(xBlock, blockSeqLen, saveActivations: false, out _);
            return DotProduct(y, dOutBlock);
        }

        foreach (var (name, adapter) in attnAdapters)
        {
            AssertCentralFiniteDiffParity($"HasQGate.{name}.A", adapter.AdapterA, EvalAttnScalar);
            AssertCentralFiniteDiffParity($"HasQGate.{name}.B", adapter.AdapterB, EvalAttnScalar);
        }
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F12")]
    public void T1_F12_04_EndToEndGdnLoraTrainingReducesLossAndFallsBackFromGpuEngine()
    {
        using var scope = new TempDirectoryScope("f12_04_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        // device: "auto" exercises SupportsGpuEngine(weights) fallback guard for hybrid GDN
        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "auto");
        Assert.False(model.IsGpuAccelerated, "Hybrid GDN/QGate/QK-Norm/partial-RoPE model must fall back cleanly to pure-C# CPU mode.");

        var ds = TuneE2EFixtures.CreateChatMlDataset(model.Tokenizer,
        [
            ("q1", "YES"),
            ("q2", "YES")
        ]);

        var losses = new List<float>();
        var args = new TrainingArguments
        {
            LearningRate = 1e-2f,
            Epochs = 6,
            GradientAccumulationSteps = 2,
            MaxSteps = 10,
            WarmupRatio = 0f,
            OutputDir = scope.FilePath("out")
        };

        using var trainer = new LoraTrainer(model, args);
        trainer.Train(ds, onStep: r => losses.Add(r.Loss));

        Assert.True(losses.Count >= 6);
        Assert.All(losses, l => Assert.True(float.IsFinite(l)));
        Assert.True(losses[^1] < losses[0],
            $"Expected final training loss ({losses[^1]:F4}) < initial loss ({losses[0]:F4}).");
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F12")]
    public void T1_F12_05_GdnAdapterSaveReloadAndGgufMergerPreserveLogits()
    {
        using var scope = new TempDirectoryScope("f12_05_");
        string baseGgufPath = scope.FilePath("qwen35_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(baseGgufPath, CreateQwen35HybridSpec());

        var loraConfig = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var trainedModel = GgufLoraModel.Load(baseGgufPath, loraConfig, device: "cpu");
        var ds = TuneE2EFixtures.CreateChatMlDataset(trainedModel.Tokenizer, [("p", "answer")]);

        string outDir = scope.FilePath("adapter_out");
        using (var trainer = new LoraTrainer(trainedModel, new TrainingArguments
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

        int[] probeTokens = trainedModel.Tokenizer.Encode("<|im_start|>user\np<|im_end|>\n");
        using var preSaveLogits = trainedModel.ForwardLogits(probeTokens);

        // Reload adapter into fresh model
        string adapterBin = Path.Combine(outDir, "adapter_model.bin");
        using var reloadedModel = GgufLoraModel.Load(baseGgufPath, loraConfig, device: "cpu");
        reloadedModel.LoadAdapter(adapterBin);
        using var reloadedLogits = reloadedModel.ForwardLogits(probeTokens);

        // Merge adapter into standalone GGUF (lossless F32) and run forward without LoRA
        string mergedGgufPath = scope.FilePath("qwen35_merged.gguf");
        var mergeResult = GgufMerger.Merge(baseGgufPath, adapterBin, mergedGgufPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });
        Assert.True(mergeResult.AdaptedTensorCount > 0);
        Assert.True(File.Exists(mergedGgufPath));

        using var mergedModel = GgufLoraModel.Load(mergedGgufPath, loraConfig, device: "cpu");
        using var mergedLogits = mergedModel.ForwardLogits(probeTokens);

        var preSpan = preSaveLogits.AsSpan();
        var relSpan = reloadedLogits.AsSpan();
        var merSpan = mergedLogits.AsSpan();
        float maxReloadDiff = 0f;
        float maxMergeDiff = 0f;
        for (int i = 0; i < preSpan.Length; i++)
        {
            maxReloadDiff = MathF.Max(maxReloadDiff, MathF.Abs(preSpan[i] - relSpan[i]));
            maxMergeDiff = MathF.Max(maxMergeDiff, MathF.Abs(preSpan[i] - merSpan[i]));
        }

        Assert.True(maxReloadDiff < 1e-5f, $"Reloaded adapter logit maxDiff={maxReloadDiff} >= 1e-5");
        Assert.True(maxMergeDiff < 1e-4f, $"Merged GGUF logit maxDiff={maxMergeDiff} >= 1e-4");

        // Verify F32 and Q8_0 merged GGUFs via InferenceSession + KlDivergenceVerifier
        static void AssertMergedInferenceKl(string ggufFile, int[] tok, ReadOnlySpan<float> refLogits, double threshold, string label)
        {
            using var session = new InferenceSession(ggufFile, maxSeqLen: 128, device: "cpu");
            float[] actualLogits = new float[tok.Length * session.VocabSize];
            session.VerifyBatchLogits(tok, 0, actualLogits);
            var kl = KlDivergenceVerifier.EvaluateLogits(
                refLogits,
                actualLogits,
                session.VocabSize,
                new KlDivergenceOptions { Threshold = threshold, MinTop1Agreement = 1.0 });
            Assert.True(kl.Passed, $"{label} KL verification failed: {kl.FailureReason}");
            Assert.True(kl.MeanKlDivergence < threshold, $"{label} MeanKL={kl.MeanKlDivergence} >= {threshold}");
            Assert.Equal(1.0, kl.Top1AgreementRate);
        }

        AssertMergedInferenceKl(mergedGgufPath, probeTokens, preSpan, 1e-4, "Unfused-F32");

        string mergedQ8Path = scope.FilePath("qwen35_merged_q8_0.gguf");
        var mergeQ8Result = GgufMerger.Merge(baseGgufPath, adapterBin, mergedQ8Path,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q8_0 });
        Assert.True(mergeQ8Result.AdaptedTensorCount > 0);
        AssertMergedInferenceKl(mergedQ8Path, probeTokens, preSpan, 1e-3, "Unfused-Q8_0");

        // Fused GDN tensors (attn_qkv_gate + ssm_beta_alpha) training + F32 & Q8_0 merge + InferenceSession KL verification
        string fusedBasePath = scope.FilePath("qwen35_fused_base.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(fusedBasePath, CreateQwen35HybridSpec(fuseGdnTensors: true, seed: 2468));
        using var fusedModel = GgufLoraModel.Load(fusedBasePath, loraConfig, device: "cpu");
        var fusedDs = TuneE2EFixtures.CreateChatMlDataset(fusedModel.Tokenizer, [("p", "answer")]);
        string fusedOutDir = scope.FilePath("fused_adapter_out");
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

        using var fusedPreLogits = fusedModel.ForwardLogits(probeTokens);
        string fusedAdapterBin = Path.Combine(fusedOutDir, "adapter_model.bin");

        string fusedMergedF32 = scope.FilePath("qwen35_fused_merged_f32.gguf");
        var fusedMergeF32 = GgufMerger.Merge(fusedBasePath, fusedAdapterBin, fusedMergedF32,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });
        Assert.True(fusedMergeF32.AdaptedTensorCount > 0);
        AssertMergedInferenceKl(fusedMergedF32, probeTokens, fusedPreLogits.AsSpan(), 1e-4, "FusedGDN-F32");

        string fusedMergedQ8 = scope.FilePath("qwen35_fused_merged_q8_0.gguf");
        var fusedMergeQ8 = GgufMerger.Merge(fusedBasePath, fusedAdapterBin, fusedMergedQ8,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q8_0 });
        Assert.True(fusedMergeQ8.AdaptedTensorCount > 0);
        AssertMergedInferenceKl(fusedMergedQ8, probeTokens, fusedPreLogits.AsSpan(), 1e-3, "FusedGDN-Q8_0");
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F12")]
    public void T2_F12_01_SequenceLengthsShorterThanConvKernelHandleLeftPaddingAndBptt()
    {
        using var scope = new TempDirectoryScope("f12_t2_01_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");

        // seqLen = 1 (< convKernel = 4): ForwardLogits + GdnBlock.Forward/Backward
        using (var logits1 = model.ForwardLogits([10]))
        {
            Assert.Equal(1, logits1.Shape[0]);
            foreach (float v in logits1.AsSpan()) Assert.True(float.IsFinite(v));
        }

        var gdn = model.GdnBlocks[0];
        using (var x1 = TensorFloatExtensions.RandomNormal([1, model.HiddenDim], 0f, 0.2f, seed: 7))
        using (var out1 = gdn.Forward(x1, seqLen: 1, saveActivations: true, out var st1))
        using (var dOut1 = TensorFloatExtensions.RandomNormal([1, model.HiddenDim], 0f, 0.2f, seed: 8))
        using (var dX1 = gdn.Backward(dOut1, st1!, x1, seqLen: 1))
        {
            st1!.Dispose();
            Assert.True(MaxAbsGrad(dX1) > 0f);
            Assert.True(MaxAbsGrad(gdn.SsmOutProj.AdapterB.Grad) > 0f);
        }

        // seqLen = 2 (< convKernel = 4): full model ForwardLossAndBackward
        float loss2 = model.ForwardLossAndBackward([10, 11], [11, TuneE2EFixtures.EosTokenId]);
        Assert.True(float.IsFinite(loss2) && loss2 > 0f);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(16)]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F12")]
    public void T2_F12_02_ConvKernelAndChunkBoundarySequenceLengthsExecuteCleanly(int seqLen)
    {
        using var scope = new TempDirectoryScope($"f12_t2_02_{seqLen}_");
        string ggufPath = scope.FilePath("qwen35_hybrid.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(ggufPath, CreateQwen35HybridSpec());

        using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        int[] inputs = new int[seqLen];
        int[] targets = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            inputs[i] = 65 + (i % 26);
            targets[i] = 65 + ((i + 1) % 26);
        }

        using var logits = model.ForwardLogits(inputs);
        Assert.Equal(seqLen, logits.Shape[0]);
        foreach (float v in logits.AsSpan()) Assert.True(float.IsFinite(v));

        float loss = model.ForwardLossAndBackward(inputs, targets);
        Assert.True(float.IsFinite(loss) && loss > 0f);
        Assert.True(MaxAbsGrad(model.GdnBlocks[0].SsmOutProj.AdapterB.Grad) > 0f);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F12")]
    public void T2_F12_03_HasQGateFalseVsTrueDispatchWithMatchingShapesAndFiniteGrads()
    {
        using var scope = new TempDirectoryScope("f12_t2_03_");
        string noGatePath = scope.FilePath("no_qgate.gguf");
        string withGatePath = scope.FilePath("with_qgate.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(noGatePath, CreateQwen35HybridSpec(hasQGate: false));
        TuneE2EFixtures.WriteSyntheticGguf(withGatePath, CreateQwen35HybridSpec(hasQGate: true));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var mNoGate = GgufLoraModel.Load(noGatePath, cfg, device: "cpu");
        using var mWithGate = GgufLoraModel.Load(withGatePath, cfg, device: "cpu");

        Assert.False(mNoGate.Weights.Layers[1].HasQGate);
        Assert.True(mWithGate.Weights.Layers[1].HasQGate);
        Assert.Equal(32, mNoGate.Blocks[0].QProj.OutFeatures);
        Assert.Equal(64, mWithGate.Blocks[0].QProj.OutFeatures);

        int[] inputs = [65, 66, 67, 68, 69];
        int[] targets = [66, 67, 68, 69, TuneE2EFixtures.EosTokenId];

        float lNo = mNoGate.ForwardLossAndBackward(inputs, targets);
        float lWith = mWithGate.ForwardLossAndBackward(inputs, targets);
        Assert.True(float.IsFinite(lNo) && lNo > 0f);
        Assert.True(float.IsFinite(lWith) && lWith > 0f);
        Assert.True(MaxAbsGrad(mNoGate.Blocks[0].QProj.AdapterB.Grad) > 0f);
        Assert.True(MaxAbsGrad(mWithGate.Blocks[0].QProj.AdapterB.Grad) > 0f);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F12")]
    public unsafe void T2_F12_04_PartialRopeLeavesUnrotatedTailUntouchedAndProducesFiniteGrads()
    {
        // 1. Direct kernel check: unrotated tail [ropeDim..headDim) must remain bit-identical after RoPE
        const int seqLen = 3, nHeadsQ = 2, nHeadsKv = 2, headDim = 16, ropeDim = 8;
        float[] q = new float[seqLen * nHeadsQ * headDim];
        float[] k = new float[seqLen * nHeadsKv * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = 0.1f * ((i % headDim) + 1);
        for (int i = 0; i < k.Length; i++) k[i] = -0.05f * ((i % headDim) + 1);
        float[] qCopy = (float[])q.Clone();
        float[] kCopy = (float[])k.Clone();

        fixed (float* pQ = q, pK = k)
        {
            RoPEKernel.ForwardSequence(pQ, pK, seqLen, nHeadsQ, nHeadsKv, headDim, 10000f, ropeDim: ropeDim);
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHeadsQ; h++)
            {
                int baseIdx = (t * nHeadsQ + h) * headDim;
                for (int d = ropeDim; d < headDim; d++)
                {
                    Assert.Equal(qCopy[baseIdx + d], q[baseIdx + d]);
                    Assert.Equal(kCopy[baseIdx + d], k[baseIdx + d]);
                }
            }
        }

        // 2. End-to-end model check for partial ropeDim=8 vs full ropeDim=16
        using var scope = new TempDirectoryScope("f12_t2_04_");
        string partialPath = scope.FilePath("partial_rope.gguf");
        string fullPath = scope.FilePath("full_rope.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(partialPath, CreateQwen35HybridSpec(ropeDim: 8));
        TuneE2EFixtures.WriteSyntheticGguf(fullPath, CreateQwen35HybridSpec(ropeDim: 16));

        var cfg = new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 };
        using var mPartial = GgufLoraModel.Load(partialPath, cfg, device: "cpu");
        using var mFull = GgufLoraModel.Load(fullPath, cfg, device: "cpu");

        int[] inputs = [65, 66, 67, 68];
        int[] targets = [66, 67, 68, TuneE2EFixtures.EosTokenId];
        float lPartial = mPartial.ForwardLossAndBackward(inputs, targets);
        float lFull = mFull.ForwardLossAndBackward(inputs, targets);
        Assert.True(float.IsFinite(lPartial));
        Assert.True(float.IsFinite(lFull));
        Assert.True(MaxAbsGrad(mPartial.Blocks[0].QProj.AdapterB.Grad) > 0f);
        Assert.True(MaxAbsGrad(mFull.Blocks[0].QProj.AdapterB.Grad) > 0f);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F12")]
    public void T2_F12_05_MissingOrCorruptSsmConv1dThrowsCleanManagedException()
    {
        using var scope = new TempDirectoryScope("f12_t2_05_");
        string missingConvPath = scope.FilePath("missing_conv1d.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(missingConvPath, CreateQwen35HybridSpec(includeConv1d: false));

        var exMissing = Assert.ThrowsAny<Exception>(() =>
            GgufLoraModel.Load(missingConvPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu"));
        Assert.True(exMissing is InvalidDataException or KeyNotFoundException,
            $"Unexpected exception type for missing ssm_conv1d: {exMissing.GetType()}");

        // Truncated GGUF payload must throw clean managed exception without native crash
        string truncatedPath = scope.FilePath("truncated.gguf");
        byte[] validBytes = File.ReadAllBytes(missingConvPath);
        File.WriteAllBytes(truncatedPath, validBytes[..(validBytes.Length / 2)]);
        var exTruncated = Assert.ThrowsAny<Exception>(() =>
            GgufLoraModel.Load(truncatedPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu"));
        Assert.True(exTruncated is InvalidDataException or EndOfStreamException or ArgumentException,
            $"Unexpected exception type for truncated GGUF: {exTruncated.GetType()}");

        // Wrong-shape ssm_conv1d.weight ([4, 1] instead of [4, 64]) must throw a clean managed exception on load
        string corruptConvPath = scope.FilePath("corrupt_conv1d.gguf");
        TuneE2EFixtures.WriteSyntheticGguf(corruptConvPath, CreateQwen35HybridSpec(corruptConv1d: true));
        var exCorrupt = Assert.ThrowsAny<Exception>(() =>
            GgufLoraModel.Load(corruptConvPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu"));
        Assert.True(exCorrupt is InvalidDataException or ArgumentException,
            $"Unexpected exception type for corrupt ssm_conv1d shape: {exCorrupt.GetType()}");
    }
}
