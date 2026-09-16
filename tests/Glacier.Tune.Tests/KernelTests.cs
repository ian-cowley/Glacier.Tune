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
}
