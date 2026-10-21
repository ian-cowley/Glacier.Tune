using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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
                    float score = VectorDot(qVec, kVec, headDim) * scale;
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
                new Span<float>(outVec, headDim).Clear();

                for (int j = 0; j <= i; j++)
                {
                    float pVal = pRow[j];
                    if (pVal == 0f) continue;
                    float* vVec = v + ((long)j * nHeadsKv + kvH) * headDim;
                    VectorFma(outVec, vVec, pVal, headDim);
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
                        float dp = VectorDot(dOutVec, vVec, headDim);
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
                        if (ds == 0f) continue;
                        float* kVec = k + ((long)j * nHeadsKv + kvH) * headDim;
                        VectorFma(dqVec, kVec, ds, headDim);
                    }

                    // 4. dK and dV accumulations into isolated per-head buffer (100% LOCK-FREE)
                    for (int j = 0; j <= i; j++)
                    {
                        float ds = dsRow[j];
                        float pVal = pRow[j];

                        float* headDkVec = headDk + (long)j * headDim;
                        float* headDvVec = headDv + (long)j * headDim;
                        float* qVec = q + ((long)i * nHeadsQ + h) * headDim;

                        if (ds != 0f) VectorFma(headDkVec, qVec, ds, headDim);
                        if (pVal != 0f) VectorFma(headDvVec, dOutVec, pVal, headDim);
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

                        VectorAdd(dstK, srcK, headDim);
                        VectorAdd(dstV, srcV, headDim);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static float VectorDot(float* a, float* b, int n)
    {
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            var acc = Vector512<float>.Zero;
            int i = 0;
            for (; i <= n - 16; i += 16)
            {
                var va = Vector512.Load(a + i);
                var vb = Vector512.Load(b + i);
                acc = Vector512.Add(Vector512.Multiply(va, vb), acc);
            }
            float sum = Vector512.Sum(acc);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
        else if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            for (; i <= n - 8; i += 8)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                acc = Vector256.Add(Vector256.Multiply(va, vb), acc);
            }
            float sum = Vector256.Sum(acc);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
        else
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void VectorFma(float* dst, float* src, float scale, int n)
    {
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            var vScale = Vector512.Create(scale);
            int i = 0;
            for (; i <= n - 16; i += 16)
            {
                var vd = Vector512.Load(dst + i);
                var vs = Vector512.Load(src + i);
                Vector512.Store(Vector512.Add(Vector512.Multiply(vs, vScale), vd), dst + i);
            }
            for (; i < n; i++) dst[i] += scale * src[i];
        }
        else if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var vScale = Vector256.Create(scale);
            int i = 0;
            for (; i <= n - 8; i += 8)
            {
                var vd = Vector256.Load(dst + i);
                var vs = Vector256.Load(src + i);
                Vector256.Store(Vector256.Add(Vector256.Multiply(vs, vScale), vd), dst + i);
            }
            for (; i < n; i++) dst[i] += scale * src[i];
        }
        else
        {
            for (int i = 0; i < n; i++) dst[i] += scale * src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void VectorAdd(float* dst, float* src, int n)
    {
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            int i = 0;
            for (; i <= n - 16; i += 16)
            {
                var vd = Vector512.Load(dst + i);
                var vs = Vector512.Load(src + i);
                Vector512.Store(Vector512.Add(vd, vs), dst + i);
            }
            for (; i < n; i++) dst[i] += src[i];
        }
        else if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            int i = 0;
            for (; i <= n - 8; i += 8)
            {
                var vd = Vector256.Load(dst + i);
                var vs = Vector256.Load(src + i);
                Vector256.Store(Vector256.Add(vd, vs), dst + i);
            }
            for (; i < n; i++) dst[i] += src[i];
        }
        else
        {
            for (int i = 0; i < n; i++) dst[i] += src[i];
        }
    }
}
