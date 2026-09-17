using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Glacier.Tune.Kernels;

/// <summary>
/// High-performance vectorized kernels for LoRA forward and backward passes.
/// Eliminates intermediate tensor allocations, non-contiguous matrix transpositions,
/// and CPU indexer overhead by directly operating on contiguous pointers using AVX2/FMA SIMD.
/// </summary>
public static unsafe class LoraKernels
{
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
        float* pLowRank = (float*)NativeMemory.Alloc((nuint)(seqLen * rank * sizeof(float)));
        try
        {
            // 1. lowRank = X * A  [seqLen, rank]
            if (Avx.IsSupported && rank == 16)
            {
                Parallel.For(0, seqLen, s =>
                {
                    float* pRowX = pX + (long)s * inFeatures;
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;

                    for (int k = 0; k < inFeatures; k++)
                    {
                        var vX = Vector256.Create(pRowX[k]);
                        float* pRowA = pA + (long)k * 16;
                        var a0 = Avx.LoadVector256(pRowA);
                        var a1 = Avx.LoadVector256(pRowA + 8);
                        acc0 = Fma.IsSupported ? Fma.MultiplyAdd(vX, a0, acc0) : Avx.Add(acc0, Avx.Multiply(vX, a0));
                        acc1 = Fma.IsSupported ? Fma.MultiplyAdd(vX, a1, acc1) : Avx.Add(acc1, Avx.Multiply(vX, a1));
                    }

                    float* pRowLR = pLowRank + s * 16;
                    Avx.Store(pRowLR, acc0);
                    Avx.Store(pRowLR + 8, acc1);
                });

                // 2. output += scaling * lowRank * B  [seqLen, outFeatures]
                Parallel.For(0, seqLen, s =>
                {
                    float* pRowLR = pLowRank + s * 16;
                    float* pRowOut = pOutput + (long)s * outFeatures;

                    for (int r = 0; r < 16; r++)
                    {
                        float lrVal = pRowLR[r] * scaling;
                        if (lrVal == 0f) continue;

                        float* pRowB = pB + (long)r * outFeatures;
                        var vScale = Vector256.Create(lrVal);

                        int n = 0;
                        if (Fma.IsSupported)
                        {
                            for (; n <= outFeatures - 8; n += 8)
                            {
                                var vOut = Avx.LoadVector256(pRowOut + n);
                                var vB = Avx.LoadVector256(pRowB + n);
                                vOut = Fma.MultiplyAdd(vScale, vB, vOut);
                                Avx.Store(pRowOut + n, vOut);
                            }
                        }
                        for (; n < outFeatures; n++)
                        {
                            pRowOut[n] += lrVal * pRowB[n];
                        }
                    }
                });
            }
            else
            {
                Parallel.For(0, seqLen, s =>
                {
                    float* pRowX = pX + (long)s * inFeatures;
                    float* pRowLR = pLowRank + s * rank;
                    for (int r = 0; r < rank; r++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < inFeatures; k++)
                        {
                            sum += pRowX[k] * pA[(long)k * rank + r];
                        }
                        pRowLR[r] = sum;
                    }

                    float* pRowOut = pOutput + (long)s * outFeatures;
                    for (int r = 0; r < rank; r++)
                    {
                        float lrVal = pRowLR[r] * scaling;
                        if (lrVal == 0f) continue;
                        float* pRowB = pB + (long)r * outFeatures;
                        for (int n = 0; n < outFeatures; n++)
                        {
                            pRowOut[n] += lrVal * pRowB[n];
                        }
                    }
                });
            }
        }
        finally
        {
            NativeMemory.Free(pLowRank);
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
        float* pLowRank = (float*)NativeMemory.Alloc((nuint)(seqLen * rank * sizeof(float)));
        float* pDLowRank = (float*)NativeMemory.Alloc((nuint)(seqLen * rank * sizeof(float)));

        try
        {
            // -----------------------------------------------------------------
            // 1. lowRank = X * A  [seqLen, rank]
            // -----------------------------------------------------------------
            if (Avx.IsSupported && rank == 16)
            {
                Parallel.For(0, seqLen, s =>
                {
                    float* pRowX = pX + (long)s * inFeatures;
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;

                    for (int k = 0; k < inFeatures; k++)
                    {
                        var vX = Vector256.Create(pRowX[k]);
                        float* pRowA = pA + (long)k * 16;
                        var a0 = Avx.LoadVector256(pRowA);
                        var a1 = Avx.LoadVector256(pRowA + 8);
                        acc0 = Fma.IsSupported ? Fma.MultiplyAdd(vX, a0, acc0) : Avx.Add(acc0, Avx.Multiply(vX, a0));
                        acc1 = Fma.IsSupported ? Fma.MultiplyAdd(vX, a1, acc1) : Avx.Add(acc1, Avx.Multiply(vX, a1));
                    }

                    float* pRowLR = pLowRank + s * 16;
                    Avx.Store(pRowLR, acc0);
                    Avx.Store(pRowLR + 8, acc1);
                });
            }
            else
            {
                Parallel.For(0, seqLen, s =>
                {
                    float* pRowX = pX + (long)s * inFeatures;
                    float* pRowLR = pLowRank + s * rank;
                    for (int r = 0; r < rank; r++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < inFeatures; k++)
                        {
                            sum += pRowX[k] * pA[(long)k * rank + r];
                        }
                        pRowLR[r] = sum;
                    }
                });
            }

            // -----------------------------------------------------------------
            // 2. dB += scaling * lowRank^T * dY  [rank, outFeatures]
            // -----------------------------------------------------------------
            Parallel.For(0, rank, r =>
            {
                float* pGradBRow = pGradB + (long)r * outFeatures;
                for (int s = 0; s < seqLen; s++)
                {
                    float lrVal = pLowRank[s * rank + r] * scaling;
                    float* pDYRow = pDY + (long)s * outFeatures;
                    int n = 0;
                    if (Avx.IsSupported && Fma.IsSupported)
                    {
                        var vScale = Vector256.Create(lrVal);
                        for (; n <= outFeatures - 8; n += 8)
                        {
                            var vDY = Avx.LoadVector256(pDYRow + n);
                            var vGB = Avx.LoadVector256(pGradBRow + n);
                            vGB = Fma.MultiplyAdd(vScale, vDY, vGB);
                            Avx.Store(pGradBRow + n, vGB);
                        }
                    }
                    for (; n < outFeatures; n++)
                    {
                        pGradBRow[n] += lrVal * pDYRow[n];
                    }
                }
            });

            // -----------------------------------------------------------------
            // 3. dLowRank = dY * B^T  [seqLen, rank]
            // -----------------------------------------------------------------
            Parallel.For(0, seqLen, s =>
            {
                float* pDYRow = pDY + (long)s * outFeatures;
                float* pDLRRow = pDLowRank + s * rank;
                for (int r = 0; r < rank; r++)
                {
                    float* pBRow = pB + (long)r * outFeatures;
                    float dot = 0f;
                    int n = 0;
                    if (Avx.IsSupported && Fma.IsSupported)
                    {
                        var vSum = Vector256<float>.Zero;
                        for (; n <= outFeatures - 8; n += 8)
                        {
                            var vDY = Avx.LoadVector256(pDYRow + n);
                            var vB = Avx.LoadVector256(pBRow + n);
                            vSum = Fma.MultiplyAdd(vDY, vB, vSum);
                        }
                        dot = Vector256.Sum(vSum);
                    }
                    for (; n < outFeatures; n++)
                    {
                        dot += pDYRow[n] * pBRow[n];
                    }
                    pDLRRow[r] = dot;
                }
            });

            // -----------------------------------------------------------------
            // 4. dA += scaling * X^T * dLowRank  [inFeatures, rank]
            // -----------------------------------------------------------------
            if (Avx.IsSupported && rank == 16)
            {
                Parallel.For(0, inFeatures, k =>
                {
                    float* pGradARow = pGradA + (long)k * 16;
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;

                    for (int s = 0; s < seqLen; s++)
                    {
                        float w = pX[(long)s * inFeatures + k] * scaling;
                        var vW = Vector256.Create(w);
                        float* pDLRRow = pDLowRank + s * 16;
                        var d0 = Avx.LoadVector256(pDLRRow);
                        var d1 = Avx.LoadVector256(pDLRRow + 8);
                        acc0 = Fma.IsSupported ? Fma.MultiplyAdd(vW, d0, acc0) : Avx.Add(acc0, Avx.Multiply(vW, d0));
                        acc1 = Fma.IsSupported ? Fma.MultiplyAdd(vW, d1, acc1) : Avx.Add(acc1, Avx.Multiply(vW, d1));
                    }

                    var ga0 = Avx.LoadVector256(pGradARow);
                    var ga1 = Avx.LoadVector256(pGradARow + 8);
                    Avx.Store(pGradARow, Avx.Add(ga0, acc0));
                    Avx.Store(pGradARow + 8, Avx.Add(ga1, acc1));
                });
            }
            else
            {
                Parallel.For(0, inFeatures, k =>
                {
                    float* pGradARow = pGradA + (long)k * rank;
                    for (int r = 0; r < rank; r++)
                    {
                        float sum = 0f;
                        for (int s = 0; s < seqLen; s++)
                        {
                            sum += pX[(long)s * inFeatures + k] * pDLowRank[s * rank + r];
                        }
                        pGradARow[r] += sum * scaling;
                    }
                });
            }

            // -----------------------------------------------------------------
            // 5. dXInput += scaling * dLowRank * A^T  [seqLen, inFeatures]
            // -----------------------------------------------------------------
            if (pDXInput != null)
            {
                if (Avx.IsSupported && rank == 16)
                {
                    Parallel.For(0, seqLen, s =>
                    {
                        float* pDLRRow = pDLowRank + s * 16;
                        var d0 = Avx.LoadVector256(pDLRRow);
                        var d1 = Avx.LoadVector256(pDLRRow + 8);

                        float* pDXRow = pDXInput + (long)s * inFeatures;

                        for (int k = 0; k < inFeatures; k++)
                        {
                            float* pARow = pA + (long)k * 16;
                            var a0 = Avx.LoadVector256(pARow);
                            var a1 = Avx.LoadVector256(pARow + 8);

                            var prod0 = Avx.Multiply(d0, a0);
                            var prod1 = Avx.Multiply(d1, a1);
                            float dot = Vector256.Sum(Avx.Add(prod0, prod1));

                            pDXRow[k] += dot * scaling;
                        }
                    });
                }
                else
                {
                    Parallel.For(0, seqLen, s =>
                    {
                        float* pDLRRow = pDLowRank + s * rank;
                        float* pDXRow = pDXInput + (long)s * inFeatures;
                        for (int k = 0; k < inFeatures; k++)
                        {
                            float dot = 0f;
                            for (int r = 0; r < rank; r++)
                            {
                                dot += pDLRRow[r] * pA[(long)k * rank + r];
                            }
                            pDXRow[k] += dot * scaling;
                        }
                    });
                }
            }
        }
        finally
        {
            NativeMemory.Free(pLowRank);
            NativeMemory.Free(pDLowRank);
        }
    }
}
