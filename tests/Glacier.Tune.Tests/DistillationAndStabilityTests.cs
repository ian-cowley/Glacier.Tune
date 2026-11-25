namespace Glacier.Tune.Tests;

using System;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Tune.Kernels;
using Glacier.Tune.Trainer;
using Xunit;

public sealed unsafe class DistillationAndStabilityTests
{
    [Fact]
    public void FusedCrossEntropyLoss_MultiTileLargeMargin_AvoidsCollapseAndCancellation()
    {
        const int seqLen = 2;
        const int hiddenDim = 4;
        const int vocabSize = 5000; // Spans 2 tiles (VocabTileSize = 4096)
        const int target0 = 17;     // Tile 0
        const int target1 = 4500;   // Tile 1
        const float margin = 45.0f;

        float[] finalHidden = new float[seqLen * hiddenDim];
        finalHidden[0 * hiddenDim + 0] = 1.0f;
        finalHidden[1 * hiddenDim + 2] = 1.0f;

        float[] lmHead = new float[vocabSize * hiddenDim];
        for (int v = 0; v < vocabSize; v++)
            lmHead[v * hiddenDim + 1] = 1.0f;
        lmHead[target0 * hiddenDim + 0] = margin;
        lmHead[target1 * hiddenDim + 2] = margin;

        int[] inputTokens = [1, 2];
        int[] targetTokens = [target0, target1];
        float[] dFinalHidden = new float[seqLen * hiddenDim];

        double expectedSumOther = (vocabSize - 1) * Math.Exp(-margin);
        double expectedLoss = expectedSumOther - 0.5 * expectedSumOther * expectedSumOther;
        double expectedGradFeat = -0.5 * (expectedSumOther / (1.0 + expectedSumOther)) * margin;

        fixed (float* pH = finalHidden, pW = lmHead, pDH = dFinalHidden)
        {
            float fwdLoss = FusedCrossEntropyLoss.ComputeLoss(
                pH, targetTokens, GgufType.F32, (byte*)pW, seqLen, hiddenDim, vocabSize);
            float bwdLoss = FusedCrossEntropyLoss.ComputeLossAndGradients(
                pH, inputTokens, targetTokens, GgufType.F32, (byte*)pW, pDH, seqLen, hiddenDim, vocabSize);

            Assert.True(fwdLoss > 0f && float.IsFinite(fwdLoss), $"ComputeLoss collapsed at +{margin}: {fwdLoss}");
            Assert.True(bwdLoss > 0f && float.IsFinite(bwdLoss), $"ComputeLossAndGradients collapsed at +{margin}: {bwdLoss}");
            Assert.True(Math.Abs(fwdLoss - expectedLoss) / expectedLoss < 1e-4, $"fwdLoss {fwdLoss} != {expectedLoss}");
            Assert.True(Math.Abs(bwdLoss - expectedLoss) / expectedLoss < 1e-4, $"bwdLoss {bwdLoss} != {expectedLoss}");

            Assert.True(dFinalHidden[0] < 0f && Math.Abs(dFinalHidden[0] - expectedGradFeat) / Math.Abs(expectedGradFeat) < 1e-4,
                $"dFinalHidden[0] {dFinalHidden[0]} != {expectedGradFeat}");
            Assert.True(dFinalHidden[hiddenDim + 2] < 0f && Math.Abs(dFinalHidden[hiddenDim + 2] - expectedGradFeat) / Math.Abs(expectedGradFeat) < 1e-4,
                $"dFinalHidden[1,2] {dFinalHidden[hiddenDim + 2]} != {expectedGradFeat}");
        }
    }

    [Fact]
    public void DistillationLoss_LogitKlAndCe_MatchesFiniteDifferences_AndLargeMargin()
    {
        const int numTokens = 2;
        const int vocab = 6;
        float[] student = [1.2f, -0.4f, 0.7f, 0.1f, -1.1f, 0.3f, -0.2f, 1.5f, 0.4f, -0.8f, 0.9f, -0.1f];
        float[] teacher = [0.8f, -0.2f, 1.1f, 0.0f, -0.9f, 0.5f, 0.1f, 1.2f, 0.6f, -0.4f, 0.5f, -0.3f];
        int[] targets = [0, 4];
        float[] dStudent = new float[numTokens * vocab];

        const float temp = 2.0f;
        const float kdAlpha = 0.6f;
        DistillationLoss.ComputeLogitKlAndCe(student, teacher, targets, numTokens, vocab, temp, kdAlpha, dStudent);

        const float eps = 1e-3f;
        float[] dummyGrad = new float[numTokens * vocab];
        for (int i = 0; i < student.Length; i++)
        {
            float orig = student[i];
            student[i] = orig + eps;
            double lPlus = DistillationLoss.ComputeLogitKlAndCe(student, teacher, targets, numTokens, vocab, temp, kdAlpha, dummyGrad);
            student[i] = orig - eps;
            double lMinus = DistillationLoss.ComputeLogitKlAndCe(student, teacher, targets, numTokens, vocab, temp, kdAlpha, dummyGrad);
            student[i] = orig;

            double numGrad = (lPlus - lMinus) / (2.0 * eps);
            Assert.True(Math.Abs(numGrad - dStudent[i]) < 5e-4, $"LogitKlAndCe grad mismatch at {i}: num={numGrad}, ana={dStudent[i]}");
        }

        // Large-margin stability check (+40 CE and +42 vs +40 KL)
        float[] zS = [40f, 0f, 0f];
        float[] zT = [42f, 0f, 0f];
        int[] bigTarget = [0];
        float[] dzBig = new float[3];
        double bigLoss = DistillationLoss.ComputeLogitKlAndCe(zS, zT, bigTarget, 1, 3, 1.0f, 0.5f, dzBig);

        Assert.True(bigLoss > 0.0 && double.IsFinite(bigLoss), $"DistillationLoss collapsed at large margin: {bigLoss}");
        Assert.True(dzBig[0] < 0f && float.IsFinite(dzBig[0]), $"Expected negative argmax gradient at large margin, got {dzBig[0]}");
    }

    [Fact]
    public void DistillationLoss_HiddenMseAndCosine_MatchesFiniteDifferences()
    {
        const int numTokens = 2;
        const int hiddenSize = 4;
        float[] student = [0.5f, -1.2f, 0.8f, 0.3f, -0.6f, 0.9f, 1.4f, -0.2f];
        float[] teacher = [0.2f, -0.7f, 1.1f, -0.1f, -0.3f, 0.4f, 1.0f, 0.5f];
        float[] dStudent = new float[student.Length];
        float[] dummyGrad = new float[student.Length];

        DistillationLoss.ComputeHiddenMseAndCosine(student, teacher, numTokens, hiddenSize, 0.7f, 0.3f, dStudent);

        const float eps = 1e-3f;
        for (int i = 0; i < student.Length; i++)
        {
            float orig = student[i];
            student[i] = orig + eps;
            double lPlus = DistillationLoss.ComputeHiddenMseAndCosine(student, teacher, numTokens, hiddenSize, 0.7f, 0.3f, dummyGrad);
            student[i] = orig - eps;
            double lMinus = DistillationLoss.ComputeHiddenMseAndCosine(student, teacher, numTokens, hiddenSize, 0.7f, 0.3f, dummyGrad);
            student[i] = orig;

            double numGrad = (lPlus - lMinus) / (2.0 * eps);
            Assert.True(Math.Abs(numGrad - dStudent[i]) < 5e-4, $"HiddenMseAndCosine grad mismatch at {i}: num={numGrad}, ana={dStudent[i]}");
        }
    }

    [Fact]
    public void Inkling2HeadLoraTrainer_ComputeSingleTokenLossAndGrad_FiniteDiffAndLargeMargin()
    {
        const int embd = 4;
        const int vocab = 6;
        const int rank = 2;
        var adapter = new Inkling2HeadLoraAdapter(embd, vocab, rank, alpha: 4.0f, useSiLU: true, seed: 123);
        for (int i = 0; i < adapter.B.Length; i++)
            adapter.B[i] = 0.15f * ((i % 5) - 2);

        float[] hidden = [0.6f, -0.3f, 0.8f, -0.2f];
        float[] baseLogits = [0.5f, 1.1f, -0.4f, 0.2f, -0.8f, 0.3f];
        float[] gradA = new float[adapter.A.Length];
        float[] gradB = new float[adapter.B.Length];
        float[] workLogits = new float[vocab];
        float[] workDz = new float[vocab];
        float[] workBase = new float[vocab];

        Inkling2HeadLoraTrainer.ComputeSingleTokenLossAndGrad(
            adapter, hidden, baseLogits, targetToken: 1, softcap: 30.0f,
            workLogits, workDz, gradA, gradB, accumulateGrad: true, klBeta: 0.25f, workBaseProbs: workBase);

        const float eps = 1e-3f;
        float[] dummyA = new float[gradA.Length];
        float[] dummyB = new float[gradB.Length];
        for (int i = 0; i < adapter.B.Length; i++)
        {
            float orig = adapter.B[i];
            adapter.B[i] = orig + eps;
            float lp = Inkling2HeadLoraTrainer.ComputeSingleTokenLossAndGrad(
                adapter, hidden, baseLogits, targetToken: 1, softcap: 30.0f,
                workLogits, workDz, dummyA, dummyB, accumulateGrad: false, klBeta: 0.25f, workBaseProbs: workBase);
            adapter.B[i] = orig - eps;
            float lm = Inkling2HeadLoraTrainer.ComputeSingleTokenLossAndGrad(
                adapter, hidden, baseLogits, targetToken: 1, softcap: 30.0f,
                workLogits, workDz, dummyA, dummyB, accumulateGrad: false, klBeta: 0.25f, workBaseProbs: workBase);
            adapter.B[i] = orig;

            float numGrad = (lp - lm) / (2.0f * eps);
            Assert.True(MathF.Abs(numGrad - gradB[i]) < 5e-4f, $"HeadLoRA gradB[{i}] mismatch: num={numGrad}, ana={gradB[i]}");
        }

        // Large-margin stability (+42 base vs +40 student, softcap = 0)
        var bigAdapter = new Inkling2HeadLoraAdapter(2, 3, 1, alpha: 1.0f, useSiLU: false, seed: 1);
        bigAdapter.A[0] = 1.0f; bigAdapter.A[1] = 0.0f;
        bigAdapter.B[0] = -2.0f; bigAdapter.B[1] = 0.0f; bigAdapter.B[2] = 0.0f;
        float[] bigBase = [42f, 0f, 0f];
        float[] bigHidden = [1f, 0f];
        float[] gA = new float[2], gB = new float[3], wL = new float[3], wD = new float[3], wBase = new float[3];
        float bigLoss = Inkling2HeadLoraTrainer.ComputeSingleTokenLossAndGrad(
            bigAdapter, bigHidden, bigBase, targetToken: 0, softcap: 0.0f,
            wL, wD, gA, gB, accumulateGrad: true, klBeta: 0.5f, workBaseProbs: wBase);

        Assert.True(bigLoss > 0f && float.IsFinite(bigLoss), $"HeadLoRA loss collapsed at large margin: {bigLoss}");
        Assert.True(gB[0] < 0f && float.IsFinite(gB[0]), $"Expected negative gB[0] pulling logit up, got {gB[0]}");
    }
}
