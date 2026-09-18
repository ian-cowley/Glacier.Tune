namespace Glacier.Tune.Tests;

using System;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Xunit;

public unsafe class AdversarialTuneTests
{
    [Fact]
    public void AutogradScratchWorkspace_MultiStepBackward_AssertsAddressReuseAndZeroUnmanagedAllocations()
    {
        const int seqLen = 128;
        const int hiddenDim = 256;
        const int ffnDim = 512;
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int headDim = 64;

        using var ws = AutogradScratchWorkspace.GetOrCreate(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);

        IntPtr initFfnA = (IntPtr)ws.FfnScratchA;
        IntPtr initFfnB = (IntPtr)ws.FfnScratchB;
        IntPtr initFfnC = (IntPtr)ws.FfnScratchC;
        IntPtr initHiddenA = (IntPtr)ws.HiddenScratchA;
        IntPtr initHiddenB = (IntPtr)ws.HiddenScratchB;
        IntPtr initHiddenC = (IntPtr)ws.HiddenScratchC;
        IntPtr initHiddenD = (IntPtr)ws.HiddenScratchD;
        IntPtr initQDimA = (IntPtr)ws.QDimScratchA;
        IntPtr initKvDimA = (IntPtr)ws.KvDimScratchA;

        Assert.NotEqual(IntPtr.Zero, initFfnA);
        Assert.NotEqual(IntPtr.Zero, initHiddenA);
        Assert.Equal(0u, (nuint)initFfnA & 63); // 64-byte aligned
        Assert.Equal(0u, (nuint)initHiddenA & 63);

        // Simulate 50 backward passes: verify memory pointers remain 100% invariant
        for (int step = 0; step < 50; step++)
        {
            var currentWs = AutogradScratchWorkspace.GetOrCreate(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            Assert.Same(ws, currentWs);

            Assert.Equal(initFfnA, (IntPtr)currentWs.FfnScratchA);
            Assert.Equal(initFfnB, (IntPtr)currentWs.FfnScratchB);
            Assert.Equal(initFfnC, (IntPtr)currentWs.FfnScratchC);
            Assert.Equal(initHiddenA, (IntPtr)currentWs.HiddenScratchA);
            Assert.Equal(initHiddenB, (IntPtr)currentWs.HiddenScratchB);
            Assert.Equal(initHiddenC, (IntPtr)currentWs.HiddenScratchC);
            Assert.Equal(initHiddenD, (IntPtr)currentWs.HiddenScratchD);
            Assert.Equal(initQDimA, (IntPtr)currentWs.QDimScratchA);
            Assert.Equal(initKvDimA, (IntPtr)currentWs.KvDimScratchA);

            // Wrap non-owning tensors, write values, dispose
            using var tensorFfn = AutogradScratchWorkspace.Wrap(currentWs.FfnScratchA, seqLen, ffnDim);
            using var tensorHidden = AutogradScratchWorkspace.Wrap(currentWs.HiddenScratchA, seqLen, hiddenDim);

            *(float*)tensorFfn.DataPointer = (float)step;
            *(float*)tensorHidden.DataPointer = (float)(step * 2);

            Assert.Equal((float)step, currentWs.FfnScratchA[0]);
            Assert.Equal((float)(step * 2), currentWs.HiddenScratchA[0]);
        }

        // Capacity expansion test
        ws.EnsureCapacity(seqLen * 2, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
        IntPtr expandedFfnA = (IntPtr)ws.FfnScratchA;
        Assert.NotEqual(IntPtr.Zero, expandedFfnA);
        Assert.Equal(0u, (nuint)expandedFfnA & 63);

        // Run subsequent steps at new capacity: must reuse new addresses
        for (int step = 0; step < 20; step++)
        {
            var expandedWs = AutogradScratchWorkspace.GetOrCreate(seqLen * 2, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            Assert.Equal(expandedFfnA, (IntPtr)expandedWs.FfnScratchA);
        }
    }

    [Theory]
    [InlineData(1024)] // < 4096 (1 tile)
    [InlineData(4096)] // = 4096 (exact 1 tile boundary)
    [InlineData(8192)] // = 8192 (2 tiles)
    [InlineData(16384)] // = 16384 (4 tiles)
    public void FusedCrossEntropyLoss_TiledStreamingSoftmax_MatchesAnalyticalDoublePrecisionOracle(int vocabSize)
    {
        const int seqLen = 4;
        const int hiddenDim = 64;

        var rng = new Random(42 + vocabSize);

        float[] finalHidden = new float[seqLen * hiddenDim];
        for (int i = 0; i < finalHidden.Length; i++) finalHidden[i] = (float)(rng.NextDouble() * 0.4 - 0.2);

        int[] inputTokens = [1, 2, 3, 4];
        int[] targetTokens = [rng.Next(0, vocabSize), rng.Next(0, vocabSize), -100, rng.Next(0, vocabSize)]; // token 2 ignored

        int validCount = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (targetTokens[t] >= 0 && targetTokens[t] < vocabSize) validCount++;
        }
        Assert.Equal(3, validCount);

        float[] lmHeadWeights = new float[vocabSize * hiddenDim];
        for (int i = 0; i < lmHeadWeights.Length; i++) lmHeadWeights[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        // -----------------------------------------------------------------
        // Analytical Double-Precision Oracle Computation
        // -----------------------------------------------------------------
        double oracleTotalLoss = 0.0;
        double[,] oracleDx = new double[seqLen, hiddenDim];

        double invValid = 1.0 / validCount;

        for (int t = 0; t < seqLen; t++)
        {
            int targetId = targetTokens[t];
            if (targetId < 0 || targetId >= vocabSize) continue;

            // 1. Compute logits in float64
            double[] logits = new double[vocabSize];
            double maxLogit = double.NegativeInfinity;

            for (int v = 0; v < vocabSize; v++)
            {
                double dot = 0.0;
                for (int d = 0; d < hiddenDim; d++)
                {
                    dot += (double)finalHidden[t * hiddenDim + d] * (double)lmHeadWeights[v * hiddenDim + d];
                }
                logits[v] = dot;
                if (dot > maxLogit) maxLogit = dot;
            }

            // 2. Compute float64 sum of exp
            double sumExp = 0.0;
            for (int v = 0; v < vocabSize; v++)
            {
                sumExp += Math.Exp(logits[v] - maxLogit);
            }
            double logSumExp = maxLogit + Math.Log(sumExp);
            double tokenLoss = logSumExp - logits[targetId];
            oracleTotalLoss += tokenLoss;

            // 3. Compute float64 gradients
            for (int v = 0; v < vocabSize; v++)
            {
                double p = Math.Exp(logits[v] - logSumExp);
                double gradScale = (v == targetId) ? (p - 1.0) * invValid : p * invValid;

                for (int d = 0; d < hiddenDim; d++)
                {
                    oracleDx[t, d] += gradScale * (double)lmHeadWeights[v * hiddenDim + d];
                }
            }
        }
        oracleTotalLoss *= invValid;

        // -----------------------------------------------------------------
        // Execution of FusedCrossEntropyLoss
        // -----------------------------------------------------------------
        float[] dFinalHidden = new float[seqLen * hiddenDim];

        float fusedLoss;
        fixed (float* pHidden = finalHidden, pHead = lmHeadWeights, pDHidden = dFinalHidden)
        {
            fusedLoss = FusedCrossEntropyLoss.ComputeLossAndGradients(
                pHidden,
                inputTokens,
                targetTokens,
                GgufType.F32,
                (byte*)pHead,
                pDHidden,
                seqLen,
                hiddenDim,
                vocabSize);
        }

        // -----------------------------------------------------------------
        // Numerical Precision Assertions (< 1e-5 error)
        // -----------------------------------------------------------------
        float lossError = MathF.Abs(fusedLoss - (float)oracleTotalLoss);
        Assert.True(lossError < 1e-5f, $"Loss mismatch for vocabSize={vocabSize}: error={lossError}, fused={fusedLoss}, oracle={oracleTotalLoss}");

        float maxGradDiff = 0.0f;
        for (int t = 0; t < seqLen; t++)
        {
            if (targetTokens[t] < 0 || targetTokens[t] >= vocabSize)
            {
                // Ignored token: gradient must be exactly 0
                for (int d = 0; d < hiddenDim; d++)
                {
                    Assert.Equal(0.0f, dFinalHidden[t * hiddenDim + d]);
                }
                continue;
            }

            for (int d = 0; d < hiddenDim; d++)
            {
                float actual = dFinalHidden[t * hiddenDim + d];
                float expected = (float)oracleDx[t, d];
                float diff = MathF.Abs(actual - expected);
                if (diff > maxGradDiff) maxGradDiff = diff;
            }
        }

        Assert.True(maxGradDiff < 1e-5f, $"Gradient maximum difference exceeded 1e-5 for vocabSize={vocabSize}: maxDiff={maxGradDiff}");
    }
}
