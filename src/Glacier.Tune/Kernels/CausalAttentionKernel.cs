using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Full-sequence Causal Multi-Head Self-Attention forward and backward kernels with Grouped-Query Attention (GQA).
/// Vectorized across AVX-512 / AVX2+FMA / Vector128 with pooled per-head backward workspaces.
/// </summary>
public static unsafe class CausalAttentionKernel
{
    private const int MaxStackAllocSeqLen = 1024;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
                    float score = LoraKernels.DotSimd(qVec, kVec, headDim) * scale;
                    pRow[j] = score;
                    if (score > maxScore) maxScore = score;
                }

                if (seqLen - (i + 1) > 0)
                {
                    new Span<float>(pRow + i + 1, seqLen - (i + 1)).Clear();
                }

                // 2. Vectorized Softmax over valid lower-triangular entries [0 .. i]
                float sumExp = SoftmaxExpInPlace(pRow, i + 1, maxScore);
                float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;
                ScaleSimd(pRow, invSum, i + 1);

                // 3. Output = P * V
                float* outVec = output + ((long)i * nHeadsQ + h) * headDim;
                new Span<float>(outVec, headDim).Clear();

                for (int j = 0; j <= i; j++)
                {
                    float pVal = pRow[j];
                    if (pVal != 0f)
                    {
                        float* vVec = v + ((long)j * nHeadsKv + kvH) * headDim;
                        LoraKernels.AxpySimd(outVec, vVec, pVal, headDim);
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Backward(
        float* dOutput,
        float* q, float* k, float* v,
        float* attnProbsStorage, // [nHeadsQ, seqLen, seqLen]
        float* dq, float* dk, float* dv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int gqaRatio = nHeadsQ / nHeadsKv;
        float scale = 1.0f / MathF.Sqrt(headDim);

        new Span<float>(dq, checked(seqLen * nHeadsQ * headDim)).Clear();
        new Span<float>(dk, checked(seqLen * nHeadsKv * headDim)).Clear();
        new Span<float>(dv, checked(seqLen * nHeadsKv * headDim)).Clear();

        long perHeadKvElements = (long)seqLen * headDim;
        long totalWorkspaceElements = (long)nHeadsQ * perHeadKvElements;

        float[]? rentedDk = totalWorkspaceElements <= Array.MaxLength ? ArrayPool<float>.Shared.Rent((int)totalWorkspaceElements) : null;
        float[]? rentedDv = totalWorkspaceElements <= Array.MaxLength ? ArrayPool<float>.Shared.Rent((int)totalWorkspaceElements) : null;
        float* nativeDk = null;
        float* nativeDv = null;

        try
        {
            if (rentedDk != null && rentedDv != null)
            {
                Array.Clear(rentedDk, 0, (int)totalWorkspaceElements);
                Array.Clear(rentedDv, 0, (int)totalWorkspaceElements);
                fixed (float* pDkWs = rentedDk)
                fixed (float* pDvWs = rentedDv)
                {
                    RunBackwardCore(dOutput, q, k, v, attnProbsStorage, dq, dk, dv, pDkWs, pDvWs, seqLen, nHeadsQ, nHeadsKv, headDim, gqaRatio, scale, perHeadKvElements);
                }
            }
            else
            {
                nuint workspaceBytes = checked((nuint)(totalWorkspaceElements * sizeof(float)));
                nativeDk = (float*)NativeMemory.AllocZeroed(workspaceBytes);
                nativeDv = (float*)NativeMemory.AllocZeroed(workspaceBytes);
                RunBackwardCore(dOutput, q, k, v, attnProbsStorage, dq, dk, dv, nativeDk, nativeDv, seqLen, nHeadsQ, nHeadsKv, headDim, gqaRatio, scale, perHeadKvElements);
            }
        }
        finally
        {
            if (rentedDk != null) ArrayPool<float>.Shared.Return(rentedDk);
            if (rentedDv != null) ArrayPool<float>.Shared.Return(rentedDv);
            if (nativeDk != null) NativeMemory.Free(nativeDk);
            if (nativeDv != null) NativeMemory.Free(nativeDv);
        }
    }

    private static void RunBackwardCore(
        float* dOutput,
        float* q, float* k, float* v,
        float* attnProbsStorage,
        float* dq, float* dk, float* dv,
        float* dkWorkspace, float* dvWorkspace,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim,
        int gqaRatio, float scale, long perHeadKvElements)
    {
        Parallel.For(0, nHeadsQ, h =>
        {
            int kvH = h / gqaRatio;
            float* probsHead = attnProbsStorage + (long)h * seqLen * seqLen;
            float* headDk = dkWorkspace + (long)h * perHeadKvElements;
            float* headDv = dvWorkspace + (long)h * perHeadKvElements;

            float[]? pooledScratch = seqLen > MaxStackAllocSeqLen
                ? ArrayPool<float>.Shared.Rent(seqLen * 2)
                : null;

            try
            {
                Span<float> dpRow = pooledScratch != null
                    ? pooledScratch.AsSpan(0, seqLen)
                    : stackalloc float[seqLen];
                Span<float> dsRow = pooledScratch != null
                    ? pooledScratch.AsSpan(seqLen, seqLen)
                    : stackalloc float[seqLen];

                for (int i = 0; i < seqLen; i++)
                {
                    float* dOutVec = dOutput + ((long)i * nHeadsQ + h) * headDim;
                    float* pRow = probsHead + (long)i * seqLen;

                    // 1. dP = dOut * V^T (only for j <= i)
                    float dotPdP = 0.0f;
                    for (int j = 0; j <= i; j++)
                    {
                        float* vVec = v + ((long)j * nHeadsKv + kvH) * headDim;
                        float dp = LoraKernels.DotSimd(dOutVec, vVec, headDim);
                        dpRow[j] = dp;
                        dotPdP += pRow[j] * dp;
                    }

                    // 2. Softmax backward: dS = P * (dP - dot(P, dP)) * scale
                    ComputeDsRowSimd(pRow, dpRow, dsRow, dotPdP, scale, i + 1);

                    // 3. dQ += dS * K
                    float* dqVec = dq + ((long)i * nHeadsQ + h) * headDim;
                    for (int j = 0; j <= i; j++)
                    {
                        float ds = dsRow[j];
                        if (ds != 0f)
                        {
                            float* kVec = k + ((long)j * nHeadsKv + kvH) * headDim;
                            LoraKernels.AxpySimd(dqVec, kVec, ds, headDim);
                        }
                    }

                    // 4. dK and dV accumulations into isolated per-head buffer
                    float* qVec = q + ((long)i * nHeadsQ + h) * headDim;
                    for (int j = 0; j <= i; j++)
                    {
                        float ds = dsRow[j];
                        float pVal = pRow[j];
                        if (ds != 0f)
                        {
                            LoraKernels.AxpySimd(headDk + (long)j * headDim, qVec, ds, headDim);
                        }
                        if (pVal != 0f)
                        {
                            LoraKernels.AxpySimd(headDv + (long)j * headDim, dOutVec, pVal, headDim);
                        }
                    }
                }
            }
            finally
            {
                if (pooledScratch != null)
                {
                    ArrayPool<float>.Shared.Return(pooledScratch);
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
                    AddInPlaceSimd(dstK, srcK, headDim);
                    AddInPlaceSimd(dstV, srcV, headDim);
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftmaxExpInPlace(float* pRow, int count, float maxScore)
    {
        float sum = 0f;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && count >= 16)
        {
            var vMax = Vector512.Create(maxScore);
            var vSum = Vector512<float>.Zero;
            for (; i <= count - 16; i += 16)
            {
                var e = TuneSimdMath.FastExpNonPositiveAvx512(Vector512.Load(pRow + i) - vMax);
                Vector512.Store(e, pRow + i);
                vSum += e;
            }
            sum += Vector512.Sum(vSum);
        }
        if (Vector256.IsHardwareAccelerated && i <= count - 8)
        {
            var vMax = Vector256.Create(maxScore);
            var vSum = Vector256<float>.Zero;
            for (; i <= count - 8; i += 8)
            {
                var e = TuneSimdMath.FastExpNonPositiveAvx2(Vector256.Load(pRow + i) - vMax);
                Vector256.Store(e, pRow + i);
                vSum += e;
            }
            sum += Vector256.Sum(vSum);
        }
        if (Vector128.IsHardwareAccelerated && i <= count - 4)
        {
            var vMax = Vector128.Create(maxScore);
            var vSum = Vector128<float>.Zero;
            for (; i <= count - 4; i += 4)
            {
                var e = TuneSimdMath.FastExpNonPositiveAdvSimd(Vector128.Load(pRow + i) - vMax);
                Vector128.Store(e, pRow + i);
                vSum += e;
            }
            sum += Vector128.Sum(vSum);
        }
        for (; i < count; i++)
        {
            float e = TuneSimdMath.FastExpNonPositiveScalar(pRow[i] - maxScore);
            pRow[i] = e;
            sum += e;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleSimd(float* ptr, float scale, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= 16)
        {
            var vScale = Vector512.Create(scale);
            for (; i <= count - 16; i += 16)
            {
                Vector512.Store(Vector512.Load(ptr + i) * vScale, ptr + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= count - 8)
        {
            var vScale = Vector256.Create(scale);
            for (; i <= count - 8; i += 8)
            {
                Vector256.Store(Vector256.Load(ptr + i) * vScale, ptr + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= count - 4)
        {
            var vScale = Vector128.Create(scale);
            for (; i <= count - 4; i += 4)
            {
                Vector128.Store(Vector128.Load(ptr + i) * vScale, ptr + i);
            }
        }
        for (; i < count; i++)
        {
            ptr[i] *= scale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeDsRowSimd(float* pRow, Span<float> dpRow, Span<float> dsRow, float dotPdP, float scale, int count)
    {
        fixed (float* pDp = dpRow, pDs = dsRow)
        {
            int i = 0;
            if (Vector512.IsHardwareAccelerated && count >= 16)
            {
                var vDot = Vector512.Create(dotPdP);
                var vScale = Vector512.Create(scale);
                for (; i <= count - 16; i += 16)
                {
                    var vp = Vector512.Load(pRow + i);
                    var vdp = Vector512.Load(pDp + i);
                    Vector512.Store(vp * (vdp - vDot) * vScale, pDs + i);
                }
            }
            if (Vector256.IsHardwareAccelerated && i <= count - 8)
            {
                var vDot = Vector256.Create(dotPdP);
                var vScale = Vector256.Create(scale);
                for (; i <= count - 8; i += 8)
                {
                    var vp = Vector256.Load(pRow + i);
                    var vdp = Vector256.Load(pDp + i);
                    Vector256.Store(vp * (vdp - vDot) * vScale, pDs + i);
                }
            }
            if (Vector128.IsHardwareAccelerated && i <= count - 4)
            {
                var vDot = Vector128.Create(dotPdP);
                var vScale = Vector128.Create(scale);
                for (; i <= count - 4; i += 4)
                {
                    var vp = Vector128.Load(pRow + i);
                    var vdp = Vector128.Load(pDp + i);
                    Vector128.Store(vp * (vdp - vDot) * vScale, pDs + i);
                }
            }
            for (; i < count; i++)
            {
                pDs[i] = pRow[i] * (pDp[i] - dotPdP) * scale;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddInPlaceSimd(float* dst, float* src, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= 16)
        {
            for (; i <= count - 16; i += 16)
            {
                Vector512.Store(Vector512.Load(dst + i) + Vector512.Load(src + i), dst + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= count - 8)
        {
            for (; i <= count - 8; i += 8)
            {
                Vector256.Store(Vector256.Load(dst + i) + Vector256.Load(src + i), dst + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= count - 4)
        {
            for (; i <= count - 4; i += 4)
            {
                Vector128.Store(Vector128.Load(dst + i) + Vector128.Load(src + i), dst + i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += src[i];
        }
    }
}
