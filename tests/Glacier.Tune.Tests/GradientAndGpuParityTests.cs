namespace Glacier.Tune.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Export;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

public unsafe class GradientAndGpuParityTests
{
    // =========================================================================
    // 1. Multi-Layer Central Finite-Difference Gradient Oracles (All 7 Projections)
    // =========================================================================

    [Fact]
    public void MultiLayer_All7Projections_ColdStartZeroB_ReceivesNonZeroGradientsMatchingFiniteDifferences()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 4, alpha: 8f, weightType: GgufType.F32, zeroMeanWeights: true);
        using var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 123 }, device: "cpu");

        int[] inputTokens = [1, 3, 7, 11, 15];
        int[] targetTokens = [-100, 7, 11, 15, 2];

        float loss = model.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false);
        Assert.True(loss > 0f && float.IsFinite(loss));

        // At cold start (AdapterB == 0), dA is analytically 0, while EVERY AdapterB across ALL 7 projections
        // in BOTH Layer 0 and Layer 1 MUST receive exact non-zero gradients propagated through frozen base weights!
        for (int l = 0; l < model.LayerCount; l++)
        {
            var b = model.Blocks[l];
            (string Name, LoraAdapter Proj)[] projections =
            [
                ("q_proj", b.QProj),
                ("k_proj", b.KProj),
                ("v_proj", b.VProj),
                ("o_proj", b.OProj),
                ("gate_proj", b.GateProj),
                ("up_proj", b.UpProj),
                ("down_proj", b.DownProj)
            ];

            foreach (var (name, proj) in projections)
            {
                var gradB = proj.AdapterB.Grad!.AsSpan();
                float maxAbsGrad = 0f;
                for (int i = 0; i < gradB.Length; i++)
                {
                    float a = MathF.Abs(gradB[i]);
                    if (a > maxAbsGrad) maxAbsGrad = a;
                }

                Assert.True(maxAbsGrad > 1e-6f, $"Layer {l} {name}.AdapterB received zero gradient at cold start (maxAbs={maxAbsGrad})");
                VerifyTensorGradAgainstFiniteDifferences(model, proj.AdapterB, inputTokens, targetTokens, sampleCount: 6, $"L{l}.{name}.B");
            }
        }
    }

    [Fact]
    public void MultiLayer_All7Projections_TrainedNonZeroB_MatchesCentralFiniteDifferencesForBothAAndB()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 4, alpha: 8f, weightType: GgufType.F32, zeroMeanWeights: true);
        using var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 321 }, device: "cpu");

        PopulateDeterministicNonZeroAdapters(model, seed: 9001, scale: 0.08f);

        int[] inputTokens = [2, 5, 8, 13, 21];
        int[] targetTokens = [-100, 8, 13, 21, 4];

        float loss = model.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false);
        Assert.True(loss > 0f && float.IsFinite(loss));

        for (int l = 0; l < model.LayerCount; l++)
        {
            var b = model.Blocks[l];
            (string Name, LoraAdapter Proj)[] projections =
            [
                ("q_proj", b.QProj),
                ("k_proj", b.KProj),
                ("v_proj", b.VProj),
                ("o_proj", b.OProj),
                ("gate_proj", b.GateProj),
                ("up_proj", b.UpProj),
                ("down_proj", b.DownProj)
            ];

            foreach (var (name, proj) in projections)
            {
                VerifyTensorGradAgainstFiniteDifferences(model, proj.AdapterA, inputTokens, targetTokens, sampleCount: 4, $"L{l}.{name}.A");
                VerifyTensorGradAgainstFiniteDifferences(model, proj.AdapterB, inputTokens, targetTokens, sampleCount: 4, $"L{l}.{name}.B");
            }
        }
    }

    [Theory]
    [InlineData(GgufType.Q8_0, 64, 128)]
    [InlineData(GgufType.Q4_0, 64, 128)]
    [InlineData(GgufType.Q4_K, 256, 512)]
    public void QuantizedBaseWeights_MultiLayerBackprop_MatchesCentralFiniteDifferences(GgufType quantType, int hiddenDim, int ffnDim)
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2,
            hiddenDim: hiddenDim,
            ffnDim: ffnDim,
            nHeadsQ: hiddenDim >= 256 ? 4 : 2,
            nHeadsKv: hiddenDim >= 256 ? 2 : 1,
            vocabSize: 32,
            rank: 4,
            alpha: 8f,
            weightType: quantType,
            zeroMeanWeights: true);

        using var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 77 }, device: "cpu");
        PopulateDeterministicNonZeroAdapters(model, seed: 555, scale: 0.05f);

        int[] inputTokens = [1, 4, 9, 14];
        int[] targetTokens = [4, 9, 14, -100];

        float loss = model.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false);
        Assert.True(loss > 0f && float.IsFinite(loss));

        // Verify Layer 0 QProj & GateProj (which require backprop through Layer 1 + Layer 0 quantized base weights!)
        VerifyTensorGradAgainstFiniteDifferences(model, model.Blocks[0].QProj.AdapterB, inputTokens, targetTokens, sampleCount: 4, $"{quantType}.L0.Q.B");
        VerifyTensorGradAgainstFiniteDifferences(model, model.Blocks[0].GateProj.AdapterB, inputTokens, targetTokens, sampleCount: 4, $"{quantType}.L0.Gate.B");
        VerifyTensorGradAgainstFiniteDifferences(model, model.Blocks[0].GateProj.AdapterA, inputTokens, targetTokens, sampleCount: 4, $"{quantType}.L0.Gate.A");
    }

    // =========================================================================
    // 2. Activation Checkpointing & Live CUDA GPU vs. CPU Mathematical Parity
    // =========================================================================

    [Fact]
    public void ActivationCheckpointing_WithTrainedAdapters_MatchesUncheckpointedLossAndGradients()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 8, alpha: 16f, weightType: GgufType.Q8_0);

        using var modelStandard = fixture.CreateLoraModel(new LoraConfig { Rank = 8, Alpha = 16f, Seed = 42 }, device: "cpu");
        using var modelCheckpointed = fixture.CreateLoraModel(new LoraConfig { Rank = 8, Alpha = 16f, Seed = 42 }, device: "cpu");

        PopulateDeterministicNonZeroAdapters(modelStandard, seed: 2026, scale: 0.07f);
        PopulateDeterministicNonZeroAdapters(modelCheckpointed, seed: 2026, scale: 0.07f);

        int[] inputTokens = [1, 6, 11, 16, 21, 26];
        int[] targetTokens = [-100, 11, 16, 21, 26, 3];

        float lossStd = modelStandard.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false);
        float lossChk = modelCheckpointed.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: true);

        Assert.Equal(lossStd, lossChk, precision: 6);

        for (int p = 0; p < modelStandard.TrainableParameters.Count; p++)
        {
            var gStd = modelStandard.TrainableParameters[p].Grad!.AsSpan();
            var gChk = modelCheckpointed.TrainableParameters[p].Grad!.AsSpan();
            for (int i = 0; i < gStd.Length; i++)
            {
                Assert.True(MathF.Abs(gStd[i] - gChk[i]) < 1e-6f, $"Checkpointing gradient mismatch at param {p}[{i}]: {gStd[i]} vs {gChk[i]}");
            }
        }
    }

    [Theory]
    [InlineData(GgufType.F32, 64, 128, 64, false)]
    [InlineData(GgufType.Q8_0, 64, 128, 64, true)]
    [InlineData(GgufType.Q4_0, 64, 128, 4608, false)]  // Multi-tile vocabulary (> 4096) in VRAM!
    [InlineData(GgufType.Q4_K, 256, 512, 64, true)]    // Super-block Q4_K + trained Gate/Up/Down + checkpointing!
    public void GpuVsCpu_FullForwardLossAndBackward_AchievesMathematicalParity(
        GgufType quantType,
        int hiddenDim,
        int ffnDim,
        int vocabSize,
        bool checkpointActivations)
    {
        if (!GpuContext.IsSupported)
        {
            return; // Graceful skip on headless non-CUDA CI runners
        }

        int nHeadsQ = hiddenDim >= 256 ? 4 : 2;
        int nHeadsKv = hiddenDim >= 256 ? 2 : 1;

        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2,
            hiddenDim: hiddenDim,
            ffnDim: ffnDim,
            nHeadsQ: nHeadsQ,
            nHeadsKv: nHeadsKv,
            vocabSize: vocabSize,
            rank: 16,
            alpha: 32f,
            weightType: quantType,
            zeroMeanWeights: true);

        int[] inputTokens = [1, 5, 9, 13, 17, 21];
        int[] targetTokens = [-100, 9, 13, 17, 21, vocabSize - 3];

        float lossCpu;
        List<float[]> cpuGrads = [];
        using (var cpuModel = fixture.CreateLoraModel(new LoraConfig { Rank = 16, Alpha = 32f, Seed = 101 }, device: "cpu"))
        {
            Assert.False(cpuModel.IsGpuAccelerated);
            PopulateDeterministicNonZeroAdapters(cpuModel, seed: 4242, scale: 0.05f);
            lossCpu = cpuModel.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations);
            foreach (var p in cpuModel.TrainableParameters)
            {
                cpuGrads.Add(p.Grad!.AsSpan().ToArray());
            }
        }

        float lossGpu;
        List<float[]> gpuGrads = [];
        using (var gpuModel = fixture.CreateLoraModel(new LoraConfig { Rank = 16, Alpha = 32f, Seed = 101 }, device: "gpu"))
        {
            Assert.True(gpuModel.IsGpuAccelerated, "Expected GpuLoraEngine to initialize on CUDA-capable machine.");
            PopulateDeterministicNonZeroAdapters(gpuModel, seed: 4242, scale: 0.05f);
            lossGpu = gpuModel.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations);
            foreach (var p in gpuModel.TrainableParameters)
            {
                gpuGrads.Add(p.Grad!.AsSpan().ToArray());
            }
        }

        Assert.True(MathF.Abs(lossCpu - lossGpu) < 2e-4f, $"Loss mismatch ({quantType}): CPU={lossCpu}, GPU={lossGpu}");

        for (int p = 0; p < cpuGrads.Count; p++)
        {
            float[] c = cpuGrads[p];
            float[] g = gpuGrads[p];
            double dot = 0, normC = 0, normG = 0;
            float maxDiff = 0f;

            for (int i = 0; i < c.Length; i++)
            {
                float diff = MathF.Abs(c[i] - g[i]);
                if (diff > maxDiff) maxDiff = diff;
                dot += (double)c[i] * g[i];
                normC += (double)c[i] * c[i];
                normG += (double)g[i] * g[i];
            }

            if (normC > 1e-10 && normG > 1e-10)
            {
                double cosSim = dot / (Math.Sqrt(normC) * Math.Sqrt(normG));
                Assert.True(cosSim > 0.9998, $"GPU/CPU gradient cosine similarity dropped on {quantType} param {p}: cos={cosSim:F6}, maxDiff={maxDiff:E3}");
            }
            Assert.True(maxDiff < 1e-3f, $"GPU/CPU max gradient diff exceeded 1e-3 on {quantType} param {p}: maxDiff={maxDiff:E3}");
        }
    }

    // =========================================================================
    // 3. Unbiased Cross-Entropy Tail Conservation (No -16 Cutoff Bias)
    // =========================================================================

    [Fact]
    public void FusedCrossEntropyLoss_PeakedLogitOverLargeVocab_EliminatesTailTruncationBias()
    {
        // Construct a case where target has logit +8.0 and 16,383 non-target tokens all have logit -8.5,
        // so diff = logits[v] - logSumExp ~= -16.5 (< -16.0f).
        // Under the legacy -16.0f threshold, ALL 16,383 non-target tokens were skipped, corrupting gradients!
        const int seqLen = 2;
        const int hiddenDim = 32;
        const int vocabSize = 16384;

        float[] finalHidden = new float[seqLen * hiddenDim];
        finalHidden[0] = 1.0f;                 // token 0 activates feature 0
        finalHidden[hiddenDim + 2] = 1.0f;     // token 1 activates feature 2

        float[] lmHead = new float[vocabSize * hiddenDim];
        int target0 = 17;                      // in tile 0
        int target1 = 12000;                   // in tile 2

        for (int v = 0; v < vocabSize; v++)
        {
            lmHead[v * hiddenDim + 0] = -8.5f;
            lmHead[v * hiddenDim + 1] = 2.0f;  // constant probe feature dimension 1 across all vocabulary rows
            lmHead[v * hiddenDim + 2] = -8.5f;
        }
        lmHead[target0 * hiddenDim + 0] = 8.0f;
        lmHead[target1 * hiddenDim + 2] = 8.0f;

        int[] inputTokens = [1, 2];
        int[] targetTokens = [target0, target1];
        float[] dFinalHidden = new float[seqLen * hiddenDim];

        fixed (float* pH = finalHidden, pW = lmHead, pDH = dFinalHidden)
        {
            FusedCrossEntropyLoss.ComputeLossAndGradients(
                pH, inputTokens, targetTokens, GgufType.F32, (byte*)pW, pDH, seqLen, hiddenDim, vocabSize);
        }

        // Because every vocabulary row v has identical weight 2.0f in feature column 1,
        // exact probability conservation sum_v (p_v - 1_{v==target}) == 0 DEMANDS dFinalHidden[t, 1] ~= 0
        // (within float32 summation rounding ~5e-6 over 16,384 terms, versus +2.24e-3 bias under legacy -16 cutoff!).
        Assert.True(MathF.Abs(dFinalHidden[1]) < 1e-5f, $"Expected unbiased conservation ~0 on col 1, got {dFinalHidden[1]}");
        Assert.True(MathF.Abs(dFinalHidden[hiddenDim + 1]) < 1e-5f, $"Expected unbiased conservation ~0 on col 1, got {dFinalHidden[hiddenDim + 1]}");
    }

    // =========================================================================
    // 4. ChatML Assistant Header Masking, Gradient Accumulation & TargetModules
    // =========================================================================

    [Fact]
    public void ChatMlDataset_MasksAssistantHeaderAndSupervisesResponseThroughImEnd()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "GlacierChatMlTest_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string ggufPath = Path.Combine(tempDir, "chatml_tok.gguf");
            var vocabList = new List<string> { "<unk>", "<|im_start|>", "<|im_end|>", "\u010A", "\u0120" }; // \u010A = GPT-2 '\n', \u0120 = GPT-2 ' '
            for (char c = '!'; c <= '~'; c++) vocabList.Add(c.ToString());
            string[] vocab = vocabList.ToArray();
            BuildTokenizerTestGguf(ggufPath, vocab);

            using var gguf = new GgufFile(ggufPath);
            var tokenizer = new Glacier.Inference.Tokenizer.BpeTokenizer(gguf);

            var conv = new ChatConversation
            {
                Messages =
                [
                    new ChatMessage { Role = "user", Content = "Hi" },
                    new ChatMessage { Role = "assistant", Content = "Ok" }
                ]
            };

            var example = ChatMlDataset.FormatAndTokenize(conv, tokenizer, maxSeqLength: 128);
            int idImStart = Array.IndexOf(vocab, "<|im_start|>");
            int idImEnd = Array.IndexOf(vocab, "<|im_end|>");
            int idNewline = Array.IndexOf(vocab, "\u010A");
            int idA = Array.IndexOf(vocab, "a");
            int idO = Array.IndexOf(vocab, "O");
            int idK = Array.IndexOf(vocab, "k");

            // Find the assistant turn start in InputIds: <|im_start|> followed by 'a' ("assistant\n")
            int assistantHeaderStart = -1;
            for (int i = 0; i < example.InputIds.Length - 2; i++)
            {
                if (example.InputIds[i] == idImStart && example.InputIds[i + 1] == idA)
                {
                    assistantHeaderStart = i;
                    break;
                }
            }

            Assert.True(assistantHeaderStart >= 0, "Assistant turn header must be present in InputIds");

            // Header "<|im_start|>assistant\n" has 1 + 9 + 1 = 11 tokens; indices [0 .. assistantHeaderStart + 9] must be IgnoreIndex
            int headerNewlineIdx = assistantHeaderStart + 10;
            for (int i = 0; i < headerNewlineIdx; i++)
            {
                Assert.Equal(ChatMlDataset.IgnoreIndex, example.Labels[i]);
            }

            // At header '\n' (headerNewlineIdx), shifted label predicts the first assistant content token ('O')!
            Assert.Equal(idNewline, example.InputIds[headerNewlineIdx]);
            Assert.Equal(idO, example.Labels[headerNewlineIdx]);

            // At 'O', shifted label predicts 'k'; at 'k', shifted label predicts '<|im_end|>'!
            Assert.Equal(idO, example.InputIds[headerNewlineIdx + 1]);
            Assert.Equal(idK, example.Labels[headerNewlineIdx + 1]);

            Assert.Equal(idK, example.InputIds[headerNewlineIdx + 2]);
            Assert.Equal(idImEnd, example.Labels[headerNewlineIdx + 2]);

            // At '<|im_end|>', shifted label is IgnoreIndex (trailing newline is masked)!
            Assert.Equal(idImEnd, example.InputIds[headerNewlineIdx + 3]);
            Assert.Equal(ChatMlDataset.IgnoreIndex, example.Labels[headerNewlineIdx + 3]);
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void LoraTrainer_GradientAccumulation_AveragesGradientsAcrossMicrobatches()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 64, ffnDim: 128, rank: 4, alpha: 8f);

        int[] inputIds = [1, 4, 7, 10, 13, 16];
        int[] labels = [-100, 7, 10, 13, 16, -100];

        // Dataset A: 1 example, GradAccum = 1 -> 1 optimizer step
        var dsSingle = new ChatMlDataset();
        AddExample(dsSingle, new TrainingExample(inputIds, labels));

        // Dataset B: 4 identical copies of the same example, GradAccum = 4 -> 1 optimizer step
        var dsAccum4 = new ChatMlDataset();
        for (int k = 0; k < 4; k++) AddExample(dsAccum4, new TrainingExample(inputIds, labels));

        using var modelSingle = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");
        using var modelAccum4 = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");

        // Disable gradient clipping (high threshold) so unclipped gradient scaling directly governs parameter step
        using var trainerSingle = new LoraTrainer(modelSingle, new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 1,
            WarmupRatio = 0f,
            MaxGradNorm = 1000f,
            OutputDir = Path.Combine(fixture.DirectoryPath, "single_out")
        });

        using var trainerAccum4 = new LoraTrainer(modelAccum4, new TrainingArguments
        {
            LearningRate = 1e-3f,
            Epochs = 1,
            GradientAccumulationSteps = 4,
            WarmupRatio = 0f,
            MaxGradNorm = 1000f,
            OutputDir = Path.Combine(fixture.DirectoryPath, "accum4_out")
        });

        trainerSingle.Train(dsSingle);
        trainerAccum4.Train(dsAccum4);

        for (int p = 0; p < modelSingle.TrainableParameters.Count; p++)
        {
            var s1 = modelSingle.TrainableParameters[p].AsSpan();
            var s4 = modelAccum4.TrainableParameters[p].AsSpan();
            for (int i = 0; i < s1.Length; i++)
            {
                Assert.True(MathF.Abs(s1[i] - s4[i]) < 1e-6f,
                    $"GradAccum=4 step diverged from identical single-batch step at param {p}[{i}]: {s4[i]} vs {s1[i]}");
            }
        }
    }

    [Fact]
    public void LoraConfig_TargetModules_SelectiveAdaptationAndExport()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 2, hiddenDim: 64, ffnDim: 128, rank: 4, alpha: 8f);
        var selectiveConfig = new LoraConfig
        {
            Rank = 4,
            Alpha = 8f,
            TargetModules = ["q_proj", "v_proj"]
        };

        using var model = fixture.CreateLoraModel(selectiveConfig, device: "cpu");
        // 2 layers * 2 projections * 2 matrices (A, B) = 8 trainable tensors
        Assert.Equal(8, model.TrainableParameters.Count);
        Assert.True(model.Blocks[0].QProj.IsEnabled);
        Assert.False(model.Blocks[0].KProj.IsEnabled);
        Assert.True(model.Blocks[0].VProj.IsEnabled);
        Assert.False(model.Blocks[0].GateProj.IsEnabled);

        PopulateDeterministicNonZeroAdapters(model, seed: 314, scale: 0.05f);
        string selectiveDir = Path.Combine(fixture.DirectoryPath, "selective_adapter");
        model.SaveAdapter(selectiveDir);

        string mergedPath = Path.Combine(fixture.DirectoryPath, "selective_merged.gguf");
        var mergeRes = GgufMerger.Merge(
            fixture.GgufPath,
            Path.Combine(selectiveDir, "adapter_model.bin"),
            mergedPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });

        Assert.Equal(4, mergeRes.AdaptedTensorCount); // 2 layers * 2 projections
    }

    // =========================================================================
    // 5. Pure C# GgufMerger Forward Equivalence, Layout Auto-Detection & Safety
    // =========================================================================

    [Fact]
    public void GgufMerger_FusedStandaloneModel_MatchesUnmergedLoraModelForwardLogits()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 8, alpha: 16f, weightType: GgufType.F32);

        int[] promptTokens = [1, 5, 12, 19, 27];
        float[] unmergedLogits;

        string adapterDir = Path.Combine(fixture.DirectoryPath, "trained_for_merge");
        using (var loraModel = fixture.CreateLoraModel(new LoraConfig { Rank = 8, Alpha = 16f, Seed = 88 }, device: "cpu"))
        {
            PopulateDeterministicNonZeroAdapters(loraModel, seed: 7777, scale: 0.06f);
            using var logitsTensor = loraModel.ForwardLogits(promptTokens);
            unmergedLogits = logitsTensor.AsSpan().ToArray();
            loraModel.SaveAdapter(adapterDir);
        }

        string mergedF32Path = Path.Combine(fixture.DirectoryPath, "merged_exact_f32.gguf");
        GgufMerger.Merge(
            fixture.GgufPath,
            Path.Combine(adapterDir, "adapter_model.bin"),
            mergedF32Path,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });

        // Load fused standalone GGUF with Rank = 0 (zero LoRA overhead) and compare logits!
        using var fusedModel = GgufLoraModel.Load(mergedF32Path, new LoraConfig { Rank = 0, Alpha = 0f }, device: "cpu");
        Assert.Empty(fusedModel.TrainableParameters);

        using var fusedLogitsTensor = fusedModel.ForwardLogits(promptTokens);
        var fusedLogits = fusedLogitsTensor.AsSpan();

        for (int i = 0; i < unmergedLogits.Length; i++)
        {
            Assert.True(MathF.Abs(unmergedLogits[i] - fusedLogits[i]) < 1e-4f,
                $"Fused GGUF forward logit mismatch at [{i}]: unmerged={unmergedLogits[i]}, fused={fusedLogits[i]}");
        }
    }

    [Fact]
    public void GgufMerger_TransposedAdapterBinaryLayout_ProducesIdenticalFusedGguf()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 1, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 4, alpha: 8f, weightType: GgufType.F32);

        string nativeAdapterPath = Path.Combine(fixture.DirectoryPath, "native_layout.bin");
        string transposedAdapterPath = Path.Combine(fixture.DirectoryPath, "transposed_layout.bin");

        using (var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 55 }, device: "cpu"))
        {
            PopulateDeterministicNonZeroAdapters(model, seed: 12345, scale: 0.05f);
            model.SaveAdapter(nativeAdapterPath);
            WriteTransposedAdapterBinary(model, transposedAdapterPath);
        }

        string mergedNative = Path.Combine(fixture.DirectoryPath, "merged_native.gguf");
        string mergedTransposed = Path.Combine(fixture.DirectoryPath, "merged_transposed.gguf");

        GgufMerger.Merge(fixture.GgufPath, nativeAdapterPath, mergedNative, new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });
        GgufMerger.Merge(fixture.GgufPath, transposedAdapterPath, mergedTransposed, new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.F32 });

        byte[] bytesNative = File.ReadAllBytes(mergedNative);
        byte[] bytesTransposed = File.ReadAllBytes(mergedTransposed);
        Assert.Equal(bytesNative.Length, bytesTransposed.Length);
        Assert.True(bytesNative.AsSpan().SequenceEqual(bytesTransposed), "Transposed [rank, inDim]/[outDim, rank] merge must be bit-identical to native layout merge.");
    }

    [Fact]
    public void GgufMerger_SafetyGuards_RejectInPlaceOverwriteAndCorruptedAdapterDimensions()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 64, ffnDim: 128, rank: 4, alpha: 8f);

        // 1. In-place overwrite must throw InvalidOperationException
        Assert.Throws<InvalidOperationException>(() =>
            GgufMerger.Merge(fixture.GgufPath, fixture.AdapterPath, fixture.GgufPath));

        // 2. Mismatched adapter dimensions must throw InvalidDataException before writing output
        string badAdapterPath = Path.Combine(fixture.DirectoryPath, "corrupted_dims.bin");
        using (var fs = File.Create(badAdapterPath))
        using (var bw = new BinaryWriter(fs, Encoding.UTF8))
        {
            bw.Write((uint)0x41524F4C); // 'LORA'
            bw.Write((uint)1);
            bw.Write(1);  // layerCount
            bw.Write(4);  // rank
            bw.Write(8f); // alpha
            bw.Write(2);  // tensorCount

            bw.Write("blk.0.attn_q.lora_a.weight");
            bw.Write(32); // wrong inDim (model hiddenDim is 64!)
            bw.Write(4);
            bw.Write(new byte[32 * 4 * sizeof(float)]);

            bw.Write("blk.0.attn_q.lora_b.weight");
            bw.Write(4);
            bw.Write(64);
            bw.Write(new byte[4 * 64 * sizeof(float)]);
        }

        string badOutPath = Path.Combine(fixture.DirectoryPath, "should_not_exist.gguf");
        Assert.Throws<InvalidDataException>(() =>
            GgufMerger.Merge(fixture.GgufPath, badAdapterPath, badOutPath));
        Assert.False(File.Exists(badOutPath), "Output GGUF file must not be created when adapter validation fails.");
    }

    [Fact]
    public void GgufLoraModel_OutOfRangeInputToken_ThrowsArgumentOutOfRangeException()
    {
        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(vocabSize: 32);
        using var model = fixture.CreateLoraModel(device: "cpu");

        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardLossAndBackward([1, 99], [2, -100]));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardLossAndBackward([-1, 2], [2, -100]));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardLogits([0, 32]));
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static void PopulateDeterministicNonZeroAdapters(GgufLoraModel model, int seed, float scale)
    {
        var rng = new Random(seed);
        foreach (var p in model.TrainableParameters)
        {
            var span = p.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
        }
    }

    private static void VerifyTensorGradAgainstFiniteDifferences(
        GgufLoraModel model,
        Tensor<float> param,
        int[] inputTokens,
        int[] targetTokens,
        int sampleCount,
        string label)
    {
        const float eps = 1e-3f;
        var weights = param.AsSpan();
        var grads = param.Grad!.AsSpan();
        int step = Math.Max(1, weights.Length / sampleCount);

        for (int idx = 0; idx < weights.Length; idx += step)
        {
            float orig = weights[idx];

            weights[idx] = orig + eps;
            double lossPlus = ComputeExactDoubleLossFromForwardLogits(model, inputTokens, targetTokens);

            weights[idx] = orig - eps;
            double lossMinus = ComputeExactDoubleLossFromForwardLogits(model, inputTokens, targetTokens);

            weights[idx] = orig;

            float numericalGrad = (float)((lossPlus - lossMinus) / (2.0 * eps));
            float analyticalGrad = grads[idx];
            float absDiff = MathF.Abs(numericalGrad - analyticalGrad);
            float scale = MathF.Max(1.0f, MathF.Max(MathF.Abs(numericalGrad), MathF.Abs(analyticalGrad)));

            Assert.True(absDiff / scale < 2.5e-2f && absDiff < 5e-3f,
                $"Finite-diff mismatch on {label}[{idx}]: analytical={analyticalGrad:E5}, numerical={numericalGrad:E5}, diff={absDiff:E5}");
        }
    }

    private static double ComputeExactDoubleLossFromForwardLogits(GgufLoraModel model, int[] inputTokens, int[] targetTokens)
    {
        using var logitsTensor = model.ForwardLogits(inputTokens);
        var logits = logitsTensor.AsSpan();
        int seqLen = inputTokens.Length;
        int vocabSize = model.VocabSize;

        double totalLoss = 0.0;
        int validCount = 0;

        for (int t = 0; t < seqLen; t++)
        {
            int target = targetTokens[t];
            if (target < 0 || target >= vocabSize) continue;
            validCount++;

            int rowOffset = t * vocabSize;
            double maxL = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits[rowOffset + v];
                if (val > maxL) maxL = val;
            }

            double sumExp = 0.0;
            for (int v = 0; v < vocabSize; v++)
            {
                sumExp += Math.Exp(logits[rowOffset + v] - maxL);
            }

            double lse = maxL + Math.Log(sumExp);
            totalLoss += lse - logits[rowOffset + target];
        }

        return validCount > 0 ? totalLoss / validCount : 0.0;
    }

    private static void AddExample(ChatMlDataset dataset, TrainingExample example)
    {
        var field = typeof(ChatMlDataset).GetField("_examples", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        ((List<TrainingExample>)field.GetValue(dataset)!).Add(example);
    }

    private static void WriteTransposedAdapterBinary(GgufLoraModel model, string path)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write((uint)0x41524F4C);
        bw.Write((uint)1);
        bw.Write(model.LayerCount);
        bw.Write(model.Config.Rank);
        bw.Write(model.Config.Alpha);
        bw.Write(model.LayerCount * 14);

        static void WriteTransposed(BinaryWriter writer, string name, Tensor<float> src)
        {
            int rows = src.Shape[0];
            int cols = src.Shape[1];
            writer.Write(name);
            writer.Write(cols); // transposed dim0
            writer.Write(rows); // transposed dim1
            var span = src.AsSpan();
            for (int c = 0; c < cols; c++)
            {
                for (int r = 0; r < rows; r++)
                {
                    writer.Write(span[r * cols + c]);
                }
            }
        }

        for (int l = 0; l < model.LayerCount; l++)
        {
            var b = model.Blocks[l];
            WriteTransposed(bw, $"blk.{l}.attn_q.lora_a.weight", b.QProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.attn_q.lora_b.weight", b.QProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.attn_k.lora_a.weight", b.KProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.attn_k.lora_b.weight", b.KProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.attn_v.lora_a.weight", b.VProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.attn_v.lora_b.weight", b.VProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.attn_output.lora_a.weight", b.OProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.attn_output.lora_b.weight", b.OProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.ffn_gate.lora_a.weight", b.GateProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.ffn_gate.lora_b.weight", b.GateProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.ffn_up.lora_a.weight", b.UpProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.ffn_up.lora_b.weight", b.UpProj.AdapterB);
            WriteTransposed(bw, $"blk.{l}.ffn_down.lora_a.weight", b.DownProj.AdapterA);
            WriteTransposed(bw, $"blk.{l}.ffn_down.lora_b.weight", b.DownProj.AdapterB);
        }
    }

    private static void BuildTokenizerTestGguf(string path, string[] tokens)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        static void WriteStr(BinaryWriter w, string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        bw.Write((uint)0x46554747); // GGUF
        bw.Write((uint)3);
        bw.Write((ulong)0); // 0 tensors needed for BpeTokenizer

        bw.Write((ulong)4); // 4 metadata entries

        WriteStr(bw, "general.architecture");
        bw.Write((uint)8);
        WriteStr(bw, "qwen2");

        WriteStr(bw, "tokenizer.ggml.model");
        bw.Write((uint)8);
        WriteStr(bw, "gpt2");

        WriteStr(bw, "tokenizer.ggml.tokens");
        bw.Write((uint)9);
        bw.Write((uint)8);
        bw.Write((ulong)tokens.Length);
        foreach (string t in tokens) WriteStr(bw, t);

        // Mark <|im_start|> (1) and <|im_end|> (2) as CONTROL (type 3) so BpeTokenizer matches them greedily
        WriteStr(bw, "tokenizer.ggml.token_type");
        bw.Write((uint)9);
        bw.Write((uint)5); // Int32
        bw.Write((ulong)tokens.Length);
        for (int i = 0; i < tokens.Length; i++)
        {
            int tokenType = (tokens[i] == "<|im_start|>" || tokens[i] == "<|im_end|>") ? 3 : 1;
            bw.Write(tokenType);
        }
    }
}
