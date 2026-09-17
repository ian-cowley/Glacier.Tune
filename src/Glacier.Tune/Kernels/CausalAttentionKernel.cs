using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Full-sequence Causal Multi-Head Self-Attention forward and backward kernels with Grouped-Query Attention (GQA).
/// Handles arbitrary sequence lengths with causal lower-triangular masking.
/// </summary>
public static unsafe class CausalAttentionKernel
{
    public static void Forward(
        float* q, float* k, float* v,
        float* output,
        float* attnProbsStorage, // [nHeadsQ, seqLen, seqLen]
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int gqaRatio = nHeadsQ / nHeadsKv;
        float scale = 1.0f / MathF.Sqrt(headDim);

        Parallel.For(0, nHeadsQ, h =>
        {
            int kvH = h / gqaRatio;
            float* probsHead = attnProbsStorage + (long)h * seqLen * seqLen;

            for (int i = 0; i < seqLen; i++)
            {
                float* qVec = q + ((long)i * nHeadsQ + h) * headDim;
                float* pRow = probsHead + (long)i * seqLen;

                // 1. Q * K^T with causal mask
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j <= i; j++)
                {
                    float* kVec = k + ((long)j * nHeadsKv + kvH) * headDim;
                    float score = 0.0f;
                    for (int d = 0; d < headDim; d++) score += qVec[d] * kVec[d];
                    score *= scale;
                    pRow[j] = score;
                    if (score > maxScore) maxScore = score;
                }

                // Causal mask for j > i
                for (int j = i + 1; j < seqLen; j++) pRow[j] = 0.0f;

                // 2. Softmax over valid lower-triangular entries [0 .. i]
                float sumExp = 0.0f;
                for (int j = 0; j <= i; j++)
                {
                    float e = MathF.Exp(pRow[j] - maxScore);
                    pRow[j] = e;
                    sumExp += e;
                }
                float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;
                for (int j = 0; j <= i; j++) pRow[j] *= invSum;

                // 3. Output = P * V
                float* outVec = output + ((long)i * nHeadsQ + h) * headDim;
                for (int d = 0; d < headDim; d++) outVec[d] = 0.0f;

                for (int j = 0; j <= i; j++)
                {
                    float pVal = pRow[j];
                    float* vVec = v + ((long)j * nHeadsKv + kvH) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        outVec[d] += pVal * vVec[d];
                    }
                }
            }
        });
    }

    public static void Backward(
        float* dOutput,
        float* q, float* k, float* v,
        float* attnProbsStorage, // [nHeadsQ, seqLen, seqLen]
        float* dq, float* dk, float* dv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int gqaRatio = nHeadsQ / nHeadsKv;
        float scale = 1.0f / MathF.Sqrt(headDim);

        // Clear output gradients
        new Span<float>(dq, seqLen * nHeadsQ * headDim).Clear();
        new Span<float>(dk, seqLen * nHeadsKv * headDim).Clear();
        new Span<float>(dv, seqLen * nHeadsKv * headDim).Clear();

        // Allocate lock-free per-head accumulation workspace
        // Each Q-head accumulates into its own isolated slice [seqLen, headDim] with ZERO lock contention
        long perHeadKvElements = (long)seqLen * headDim;
        nuint workspaceBytes = (nuint)(nHeadsQ * perHeadKvElements * sizeof(float));
        float* dkWorkspace = (float*)System.Runtime.InteropServices.NativeMemory.AllocZeroed(workspaceBytes);
        float* dvWorkspace = (float*)System.Runtime.InteropServices.NativeMemory.AllocZeroed(workspaceBytes);

        try
        {
            Parallel.For(0, nHeadsQ, h =>
            {
                int kvH = h / gqaRatio;
                float* probsHead = attnProbsStorage + (long)h * seqLen * seqLen;
                float* headDk = dkWorkspace + (long)h * perHeadKvElements;
                float* headDv = dvWorkspace + (long)h * perHeadKvElements;

                // Temporary per-thread buffers
                Span<float> dpRow = stackalloc float[seqLen];
                Span<float> dsRow = stackalloc float[seqLen];

                for (int i = 0; i < seqLen; i++)
                {
                    float* dOutVec = dOutput + ((long)i * nHeadsQ + h) * headDim;
                    float* pRow = probsHead + (long)i * seqLen;

                    // 1. dP = dOut * V^T (only for j <= i)
                    float dotPdP = 0.0f;
                    for (int j = 0; j <= i; j++)
                    {
                        float* vVec = v + ((long)j * nHeadsKv + kvH) * headDim;
                        float dp = 0.0f;
                        for (int d = 0; d < headDim; d++) dp += dOutVec[d] * vVec[d];
                        dpRow[j] = dp;
                        dotPdP += pRow[j] * dp;
                    }

                    // 2. Softmax backward: dS = P * (dP - dot(P, dP)) * scale
                    for (int j = 0; j <= i; j++)
                    {
                        dsRow[j] = pRow[j] * (dpRow[j] - dotPdP) * scale;
                    }

                    // 3. dQ += dS * K
                    float* dqVec = dq + ((long)i * nHeadsQ + h) * headDim;
                    for (int j = 0; j <= i; j++)
                    {
                        float ds = dsRow[j];
                        float* kVec = k + ((long)j * nHeadsKv + kvH) * headDim;
                        for (int d = 0; d < headDim; d++)
                        {
                            dqVec[d] += ds * kVec[d];
                        }
                    }

                    // 4. dK and dV accumulations into isolated per-head buffer (100% LOCK-FREE)
                    for (int j = 0; j <= i; j++)
                    {
                        float ds = dsRow[j];
                        float pVal = pRow[j];

                        float* headDkVec = headDk + (long)j * headDim;
                        float* headDvVec = headDv + (long)j * headDim;
                        float* qVec = q + ((long)i * nHeadsQ + h) * headDim;

                        for (int d = 0; d < headDim; d++)
                        {
                            headDkVec[d] += ds * qVec[d];
                            headDvVec[d] += pVal * dOutVec[d];
                        }
                    }
                }
            });

            // 5. Parallel lock-free reduction across Q-heads for each KV head
            Parallel.For(0, nHeadsKv, kvH =>
            {
                int startH = kvH * gqaRatio;
                int endH = startH + gqaRatio;

                for (int j = 0; j < seqLen; j++)
                {
                    float* dstK = dk + ((long)j * nHeadsKv + kvH) * headDim;
                    float* dstV = dv + ((long)j * nHeadsKv + kvH) * headDim;

                    for (int h = startH; h < endH; h++)
                    {
                        float* srcK = dkWorkspace + ((long)h * seqLen + j) * headDim;
                        float* srcV = dvWorkspace + ((long)h * seqLen + j) * headDim;

                        for (int d = 0; d < headDim; d++)
                        {
                            dstK[d] += srcK[d];
                            dstV[d] += srcV[d];
                        }
                    }
                }
            });
        }
        finally
        {
            System.Runtime.InteropServices.NativeMemory.Free(dkWorkspace);
            System.Runtime.InteropServices.NativeMemory.Free(dvWorkspace);
        }
    }
}
