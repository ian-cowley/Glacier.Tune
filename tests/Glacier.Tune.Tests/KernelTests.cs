using System;
using Glacier.Tune.Kernels;
using Xunit;

namespace Glacier.Tune.Tests;

public unsafe class KernelTests
{
    [Fact]
    public void RoPE_ForwardAndBackward_RestoresOriginalVectors()
    {
        int seqLen = 4;
        int nHeadsQ = 2;
        int nHeadsKv = 1;
        int headDim = 64;

        int qLen = seqLen * nHeadsQ * headDim;
        int kLen = seqLen * nHeadsKv * headDim;

        float[] originalQ = new float[qLen];
        float[] originalK = new float[kLen];
        float[] testQ = new float[qLen];
        float[] testK = new float[kLen];

        var rng = new Random(42);
        for (int i = 0; i < qLen; i++) testQ[i] = originalQ[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < kLen; i++) testK[i] = originalK[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        fixed (float* pQ = testQ, pK = testK)
        {
            // 1. Forward RoPE
            RoPEKernel.ForwardSequence(pQ, pK, seqLen, nHeadsQ, nHeadsKv, headDim, 10000.0f);

            // 2. Backward RoPE (orthogonal inverse rotation)
            RoPEKernel.BackwardSequence(pQ, pK, seqLen, nHeadsQ, nHeadsKv, headDim, 10000.0f);
        }

        // Must restore original vectors to high precision
        for (int i = 0; i < qLen; i++)
        {
            Assert.Equal(originalQ[i], testQ[i], 4);
        }
        for (int i = 0; i < kLen; i++)
        {
            Assert.Equal(originalK[i], testK[i], 4);
        }
    }

    [Fact]
    public void SwiGlu_ForwardAndBackward_MatchesFiniteDifferences()
    {
        const int length = 16;
        float[] gate = new float[length];
        float[] up = new float[length];
        float[] hidden = new float[length];
        float[] dHidden = new float[length];
        float[] dGate = new float[length];
        float[] dUp = new float[length];

        var rng = new Random(101);
        for (int i = 0; i < length; i++)
        {
            gate[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            up[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            dHidden[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        fixed (float* pGate = gate, pUp = up, pHidden = hidden, pDH = dHidden, pDG = dGate, pDU = dUp)
        {
            SwiGluKernel.Forward(pGate, pUp, pHidden, length);
            SwiGluKernel.Backward(pDH, pGate, pUp, pDG, pDU, length);
        }

        // Numerical finite-difference check on gate
        const float eps = 1e-4f;
        for (int i = 0; i < length; i++)
        {
            float origG = gate[i];

            gate[i] = origG + eps;
            float sigPlus = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            float hPlus = (gate[i] * sigPlus) * up[i];

            gate[i] = origG - eps;
            float sigMinus = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            float hMinus = (gate[i] * sigMinus) * up[i];

            gate[i] = origG;

            float numDGate = dHidden[i] * (hPlus - hMinus) / (2.0f * eps);
            Assert.True(Math.Abs(numDGate - dGate[i]) < 1e-3f, $"Gate diff: Expected {numDGate}, got {dGate[i]}");

            float sig = 1.0f / (1.0f + MathF.Exp(-origG));
            float expectedDUp = dHidden[i] * (origG * sig);
            Assert.True(Math.Abs(expectedDUp - dUp[i]) < 1e-4f, $"Up diff: Expected {expectedDUp}, got {dUp[i]}");
        }
    }

    [Fact]
    public void CausalAttention_EnforcesStrictLowerTriangularCausality()
    {
        int seqLen = 4;
        int nHeadsQ = 1;
        int nHeadsKv = 1;
        int headDim = 8;

        float[] q = new float[seqLen * headDim];
        float[] k = new float[seqLen * headDim];
        float[] v = new float[seqLen * headDim];
        float[] output = new float[seqLen * headDim];
        float[] probs = new float[seqLen * seqLen];

        var rng = new Random(77);
        for (int i = 0; i < q.Length; i++) q[i] = (float)rng.NextDouble();
        for (int i = 0; i < k.Length; i++) k[i] = (float)rng.NextDouble();
        for (int i = 0; i < v.Length; i++) v[i] = (float)rng.NextDouble();

        fixed (float* pQ = q, pK = k, pV = v, pOut = output, pProbs = probs)
        {
            CausalAttentionKernel.Forward(pQ, pK, pV, pOut, pProbs, seqLen, nHeadsQ, nHeadsKv, headDim);
        }

        // All upper-triangular entries (j > i) must be zero
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = i + 1; j < seqLen; j++)
            {
                Assert.Equal(0.0f, probs[i * seqLen + j]);
            }
        }
    }

    [Fact]
    public void LoraKernels_Backward_MatchesOriginal()
    {
        int seqLen = 8;
        int inFeatures = 32;
        int outFeatures = 24;
        int rank = 16;
        float alpha = 32.0f;
        float scaling = alpha / rank;

        var x = new Glacier.Tensor.Core.Tensor<float>(seqLen, inFeatures);
        var dY = new Glacier.Tensor.Core.Tensor<float>(seqLen, outFeatures);
        var adapterOrig = new Glacier.Tune.Model.LoraAdapter(inFeatures, outFeatures, rank, alpha);

        var rng = new Random(42);
        for (int i = 0; i < x.ElementCount; i++) x.DataPointer[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < dY.ElementCount; i++) dY.DataPointer[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var dXOrig = new Glacier.Tensor.Core.Tensor<float>(seqLen, inFeatures);
        var dXFast = new Glacier.Tensor.Core.Tensor<float>(seqLen, inFeatures);

        // Run original
        adapterOrig.Backward(dY, x, dXOrig);

        // Run fast
        var gradA_Fast = new float[inFeatures * rank];
        var gradB_Fast = new float[rank * outFeatures];

        fixed (float* pGradA = gradA_Fast, pGradB = gradB_Fast)
        {
            LoraKernels.Backward(
                x.DataPointer,
                dY.DataPointer,
                adapterOrig.AdapterA.DataPointer,
                adapterOrig.AdapterB.DataPointer,
                pGradA,
                pGradB,
                dXFast.DataPointer,
                seqLen,
                inFeatures,
                outFeatures,
                rank,
                scaling);
        }

        // Compare GradB
        var gradBSpanOrig = adapterOrig.AdapterB.Grad!.AsSpan();
        for (int i = 0; i < gradBSpanOrig.Length; i++)
        {
            Assert.True(Math.Abs(gradBSpanOrig[i] - gradB_Fast[i]) < 1e-4f,
                $"GradB mismatch at {i}: orig={gradBSpanOrig[i]}, fast={gradB_Fast[i]}");
        }

        // Compare GradA
        var gradASpanOrig = adapterOrig.AdapterA.Grad!.AsSpan();
        for (int i = 0; i < gradASpanOrig.Length; i++)
        {
            Assert.True(Math.Abs(gradASpanOrig[i] - gradA_Fast[i]) < 1e-4f,
                $"GradA mismatch at {i}: orig={gradASpanOrig[i]}, fast={gradA_Fast[i]}");
        }

        // Compare dX
        var dXSpanOrig = dXOrig.AsSpan();
        var dXSpanFast = dXFast.AsSpan();
        for (int i = 0; i < dXSpanOrig.Length; i++)
        {
            Assert.True(Math.Abs(dXSpanOrig[i] - dXSpanFast[i]) < 1e-4f,
                $"dX mismatch at {i}: orig={dXSpanOrig[i]}, fast={dXSpanFast[i]}");
        }
    }

    [Fact]
    public void FusedCrossEntropyLoss_ComputesExactGradient_WithAutogradParity()
    {
        const int seqLen = 4;
        const int hiddenDim = 16;
        const int vocabSize = 32;

        float[] finalHidden = new float[seqLen * hiddenDim];
        float[] dFinalHidden = new float[seqLen * hiddenDim];
        float[] lmHead = new float[vocabSize * hiddenDim];
        int[] inputTokens = [1, 2, 3, 4];
        int[] targetTokens = [5, 10, 15, 20];

        var rng = new Random(1234);
        for (int i = 0; i < finalHidden.Length; i++) finalHidden[i] = (float)(rng.NextDouble() * 0.5 - 0.25);
        for (int i = 0; i < lmHead.Length; i++) lmHead[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        // Ground-truth full softmax gradient calculation
        float[] expectedGrad = new float[seqLen * hiddenDim];
        for (int t = 0; t < seqLen; t++)
        {
            float[] logits = new float[vocabSize];
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                float dot = 0f;
                for (int d = 0; d < hiddenDim; d++)
                {
                    dot += finalHidden[t * hiddenDim + d] * lmHead[v * hiddenDim + d];
                }
                logits[v] = dot;
                if (dot > maxLogit) maxLogit = dot;
            }

            float sumExp = 0f;
            for (int v = 0; v < vocabSize; v++) sumExp += MathF.Exp(logits[v] - maxLogit);
            float logSumExp = maxLogit + MathF.Log(sumExp);

            int target = targetTokens[t];
            for (int v = 0; v < vocabSize; v++)
            {
                float p = MathF.Exp(logits[v] - logSumExp);
                float gradP = (p - (v == target ? 1.0f : 0.0f)) / seqLen;
                for (int d = 0; d < hiddenDim; d++)
                {
                    expectedGrad[t * hiddenDim + d] += gradP * lmHead[v * hiddenDim + d];
                }
            }
        }

        fixed (float* pHidden = finalHidden, pDH = dFinalHidden, pLmHead = lmHead)
        {
            float loss = FusedCrossEntropyLoss.ComputeLossAndGradients(
                pHidden,
                inputTokens,
                targetTokens,
                Glacier.Inference.Gguf.GgufType.F32,
                (byte*)pLmHead,
                pDH,
                seqLen,
                hiddenDim,
                vocabSize);

            Assert.True(loss > 0f, "Loss should be positive");
        }

        for (int i = 0; i < expectedGrad.Length; i++)
        {
            Assert.True(Math.Abs(expectedGrad[i] - dFinalHidden[i]) < 1e-5f,
                $"Gradient mismatch at index {i}: expected {expectedGrad[i]}, actual {dFinalHidden[i]}");
        }
    }
}

