using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Glacier.Tune.Kernels;

/// <summary>
/// High-performance 3-tier SIMD (AVX-512 / AVX2+FMA / Vector128) kernels for LoRA forward and backward passes.
/// Supports arbitrary rank with zero NativeMemory allocation churn via pooled scratch buffers.
/// </summary>
public static unsafe partial class LoraKernels
{
    private const int ParallelWorkThreshold = 16384;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyLoraDelta(
        float* pX,
        float* pA,
        float* pB,
        float* pOutput,
        int seqLen,
        int inFeatures,
        int outFeatures,
        int rank,
        float scaling)
    {
        if (seqLen <= 0 || rank <= 0 || scaling == 0f) return;

        int lrCount = checked(seqLen * rank);
        float[] lowRankArr = ArrayPool<float>.Shared.Rent(lrCount);
        try
        {
            fixed (float* pLowRank = lowRankArr)
            {
                ComputeLowRankForward(pX, pA, pLowRank, seqLen, inFeatures, rank);
                AddScaledLowRankB(pLowRank, pB, pOutput, seqLen, outFeatures, rank, scaling);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(lowRankArr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Backward(
        float* pX,
        float* pDY,
        float* pA,
        float* pB,
        float* pGradA,
        float* pGradB,
        float* pDXInput,
        int seqLen,
        int inFeatures,
        int outFeatures,
        int rank,
        float scaling)
    {
        if (seqLen <= 0 || rank <= 0) return;

        int lrCount = checked(seqLen * rank);
        float[] lowRankArr = ArrayPool<float>.Shared.Rent(lrCount);
        float[] dLowRankArr = ArrayPool<float>.Shared.Rent(lrCount);

        try
        {
            fixed (float* pLowRank = lowRankArr)
            fixed (float* pDLowRank = dLowRankArr)
            {
                // 1. lowRank = X * A  [seqLen, rank]
                ComputeLowRankForward(pX, pA, pLowRank, seqLen, inFeatures, rank);

                // 2. dB += scaling * lowRank^T * dY  [rank, outFeatures]
                if (rank >= 4 && (long)seqLen * outFeatures >= ParallelWorkThreshold)
                {
                    float* pLRLocal = pLowRank;
                    Parallel.For(0, rank, r => AccumulateGradBRow(pLRLocal, pDY, pGradB + (long)r * outFeatures, seqLen, outFeatures, rank, r, scaling));
                }
                else
                {
                    for (int r = 0; r < rank; r++)
                    {
                        AccumulateGradBRow(pLowRank, pDY, pGradB + (long)r * outFeatures, seqLen, outFeatures, rank, r, scaling);
                    }
                }

                // 3. dLowRank = dY * B^T  [seqLen, rank]
                if ((long)seqLen * outFeatures * rank >= ParallelWorkThreshold)
                {
                    float* pDLRLocal = pDLowRank;
                    Parallel.For(0, seqLen, s => ComputeDLowRankRow(pDY + (long)s * outFeatures, pB, pDLRLocal + (long)s * rank, outFeatures, rank));
                }
                else
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        ComputeDLowRankRow(pDY + (long)s * outFeatures, pB, pDLowRank + (long)s * rank, outFeatures, rank);
                    }
                }

                // 4. dA += scaling * X^T * dLowRank  [inFeatures, rank]
                if ((long)seqLen * inFeatures * rank >= ParallelWorkThreshold)
                {
                    float* pDLRLocal = pDLowRank;
                    Parallel.For(0, inFeatures, k => AccumulateGradARow(pX, pDLRLocal, pGradA + (long)k * rank, seqLen, inFeatures, rank, k, scaling));
                }
                else
                {
                    for (int k = 0; k < inFeatures; k++)
                    {
                        AccumulateGradARow(pX, pDLowRank, pGradA + (long)k * rank, seqLen, inFeatures, rank, k, scaling);
                    }
                }

                // 5. dXInput += scaling * dLowRank * A^T  [seqLen, inFeatures]
                if (pDXInput != null && scaling != 0f)
                {
                    if ((long)seqLen * inFeatures * rank >= ParallelWorkThreshold)
                    {
                        float* pDLRLocal = pDLowRank;
                        Parallel.For(0, seqLen, s => AccumulateDXRow(pDLRLocal + (long)s * rank, pA, pDXInput + (long)s * inFeatures, inFeatures, rank, scaling));
                    }
                    else
                    {
                        for (int s = 0; s < seqLen; s++)
                        {
                            AccumulateDXRow(pDLowRank + (long)s * rank, pA, pDXInput + (long)s * inFeatures, inFeatures, rank, scaling);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(lowRankArr);
            ArrayPool<float>.Shared.Return(dLowRankArr);
        }
    }

    private static void ComputeLowRankForward(float* pX, float* pA, float* pLowRank, int seqLen, int inFeatures, int rank)
    {
        if ((long)seqLen * inFeatures * rank >= ParallelWorkThreshold)
        {
            Parallel.For(0, seqLen, s => ComputeLowRankRow(pX + (long)s * inFeatures, pA, pLowRank + (long)s * rank, inFeatures, rank));
        }
        else
        {
            for (int s = 0; s < seqLen; s++)
            {
                ComputeLowRankRow(pX + (long)s * inFeatures, pA, pLowRank + (long)s * rank, inFeatures, rank);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeLowRankRow(float* pRowX, float* pA, float* pRowLR, int inFeatures, int rank)
    {
        new Span<float>(pRowLR, rank).Clear();

        for (int k = 0; k < inFeatures; k++)
        {
            float xk = pRowX[k];
            if (xk == 0f) continue;
            AxpySimd(pRowLR, pA + (long)k * rank, xk, rank);
        }
    }

    private static void AddScaledLowRankB(float* pLowRank, float* pB, float* pOutput, int seqLen, int outFeatures, int rank, float scaling)
    {
        if ((long)seqLen * outFeatures * rank >= ParallelWorkThreshold)
        {
            Parallel.For(0, seqLen, s =>
            {
                float* pRowLR = pLowRank + (long)s * rank;
                float* pRowOut = pOutput + (long)s * outFeatures;
                for (int r = 0; r < rank; r++)
                {
                    float lrVal = pRowLR[r] * scaling;
                    if (lrVal != 0f)
                    {
                        AxpySimd(pRowOut, pB + (long)r * outFeatures, lrVal, outFeatures);
                    }
                }
            });
        }
        else
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* pRowLR = pLowRank + (long)s * rank;
                float* pRowOut = pOutput + (long)s * outFeatures;
                for (int r = 0; r < rank; r++)
                {
                    float lrVal = pRowLR[r] * scaling;
                    if (lrVal != 0f)
                    {
                        AxpySimd(pRowOut, pB + (long)r * outFeatures, lrVal, outFeatures);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateGradBRow(float* pLowRank, float* pDY, float* pGradBRow, int seqLen, int outFeatures, int rank, int r, float scaling)
    {
        for (int s = 0; s < seqLen; s++)
        {
            float lrVal = pLowRank[(long)s * rank + r] * scaling;
            if (lrVal != 0f)
            {
                AxpySimd(pGradBRow, pDY + (long)s * outFeatures, lrVal, outFeatures);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeDLowRankRow(float* pDYRow, float* pB, float* pDLRRow, int outFeatures, int rank)
    {
        for (int r = 0; r < rank; r++)
        {
            pDLRRow[r] = DotSimd(pDYRow, pB + (long)r * outFeatures, outFeatures);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateGradARow(float* pX, float* pDLowRank, float* pGradARow, int seqLen, int inFeatures, int rank, int k, float scaling)
    {
        for (int s = 0; s < seqLen; s++)
        {
            float w = pX[(long)s * inFeatures + k] * scaling;
            if (w != 0f)
            {
                AxpySimd(pGradARow, pDLowRank + (long)s * rank, w, rank);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateDXRow(float* pDLRRow, float* pA, float* pDXRow, int inFeatures, int rank, float scaling)
    {
        for (int k = 0; k < inFeatures; k++)
        {
            pDXRow[k] += DotSimd(pDLRRow, pA + (long)k * rank, rank) * scaling;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void AxpySimd(float* dst, float* src, float alpha, int length)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && length >= 16)
        {
            var vAlpha = Vector512.Create(alpha);
            for (; i <= length - 16; i += 16)
            {
                var vDst = Vector512.Load(dst + i);
                var vSrc = Vector512.Load(src + i);
                vDst = Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(vAlpha, vSrc, vDst) : Vector512.Add(vDst, Vector512.Multiply(vAlpha, vSrc));
                Vector512.Store(vDst, dst + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= length - 8)
        {
            var vAlpha = Vector256.Create(alpha);
            for (; i <= length - 8; i += 8)
            {
                var vDst = Vector256.Load(dst + i);
                var vSrc = Vector256.Load(src + i);
                vDst = Fma.IsSupported ? Fma.MultiplyAdd(vAlpha, vSrc, vDst) : Vector256.Add(vDst, Vector256.Multiply(vAlpha, vSrc));
                Vector256.Store(vDst, dst + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= length - 4)
        {
            var vAlpha = Vector128.Create(alpha);
            for (; i <= length - 4; i += 4)
            {
                var vDst = Vector128.Load(dst + i);
                var vSrc = Vector128.Load(src + i);
                vDst = Fma.IsSupported ? Fma.MultiplyAdd(vAlpha, vSrc, vDst) : Vector128.Add(vDst, Vector128.Multiply(vAlpha, vSrc));
                Vector128.Store(vDst, dst + i);
            }
        }
        for (; i < length; i++)
        {
            dst[i] += alpha * src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float DotSimd(float* a, float* b, int length)
    {
        float sum = 0f;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && length >= 16)
        {
            var vSum = Vector512<float>.Zero;
            for (; i <= length - 16; i += 16)
            {
                var va = Vector512.Load(a + i);
                var vb = Vector512.Load(b + i);
                vSum = Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(va, vb, vSum) : Vector512.Add(vSum, Vector512.Multiply(va, vb));
            }
            sum += Vector512.Sum(vSum);
        }
        if (Vector256.IsHardwareAccelerated && i <= length - 8)
        {
            var vSum = Vector256<float>.Zero;
            for (; i <= length - 8; i += 8)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                vSum = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, vSum) : Vector256.Add(vSum, Vector256.Multiply(va, vb));
            }
            sum += Vector256.Sum(vSum);
        }
        if (Vector128.IsHardwareAccelerated && i <= length - 4)
        {
            var vSum = Vector128<float>.Zero;
            for (; i <= length - 4; i += 4)
            {
                var va = Vector128.Load(a + i);
                var vb = Vector128.Load(b + i);
                vSum = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, vSum) : Vector128.Add(vSum, Vector128.Multiply(va, vb));
            }
            sum += Vector128.Sum(vSum);
        }
        for (; i < length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
