using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Glacier.Tune.Kernels;

/// <summary>
/// High-performance 3-tier SIMD Rotary Position Embedding (RoPE) forward and backward kernels.
/// Precomputes inverse frequencies once per sequence and vectorizes 2D Givens rotations.
/// </summary>
public static unsafe class RoPEKernel
{
    private const int StackThreshold = 256;

    public static void ForwardSequence(
        float* q, float* k,
        int seqLen,
        int nHeadsQ, int nHeadsKv, int headDim,
        float freqBase, float freqScale = 1.0f,
        int ropeDim = -1, float* ropeFreqs = null)
        => ProcessSequence(q, k, seqLen, nHeadsQ, nHeadsKv, headDim, freqBase, freqScale, inverse: false, ropeDim, ropeFreqs);

    public static void BackwardSequence(
        float* dq, float* dk,
        int seqLen,
        int nHeadsQ, int nHeadsKv, int headDim,
        float freqBase, float freqScale = 1.0f,
        int ropeDim = -1, float* ropeFreqs = null)
        => ProcessSequence(dq, dk, seqLen, nHeadsQ, nHeadsKv, headDim, freqBase, freqScale, inverse: true, ropeDim, ropeFreqs);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ProcessSequence(
        float* q, float* k,
        int seqLen,
        int nHeadsQ, int nHeadsKv, int headDim,
        float freqBase, float freqScale,
        bool inverse,
        int ropeDim,
        float* ropeFreqs)
    {
        if (seqLen <= 0 || headDim <= 1) return;

        int effectiveRopeDim = (ropeDim > 0 && ropeDim <= headDim) ? ropeDim : headDim;
        int halfDim = effectiveRopeDim / 2;
        if (halfDim <= 0) return;

        float[]? rented = halfDim > StackThreshold ? ArrayPool<float>.Shared.Rent(halfDim * 3) : null;
        try
        {
            Span<float> invFreq = rented != null ? rented.AsSpan(0, halfDim) : stackalloc float[halfDim];
            Span<float> cosTable = rented != null ? rented.AsSpan(halfDim, halfDim) : stackalloc float[halfDim];
            Span<float> sinTable = rented != null ? rented.AsSpan(halfDim * 2, halfDim) : stackalloc float[halfDim];

            for (int i = 0; i < halfDim; i++)
            {
                float baseFreq = 1.0f / MathF.Pow(freqBase, (float)(2 * i) / effectiveRopeDim);
                float freq = ropeFreqs != null ? (baseFreq / ropeFreqs[i]) : baseFreq;
                invFreq[i] = freq * freqScale;
            }

            float sinSign = inverse ? -1.0f : 1.0f;

            fixed (float* pInv = invFreq, pCos = cosTable, pSin = sinTable)
            {
                for (int pos = 0; pos < seqLen; pos++)
                {
                    float fPos = pos;
                    for (int i = 0; i < halfDim; i++)
                    {
                        (float s, float c) = MathF.SinCos(fPos * pInv[i]);
                        pCos[i] = c;
                        pSin[i] = s * sinSign;
                    }

                    float* posQ = q + (long)pos * nHeadsQ * headDim;
                    for (int h = 0; h < nHeadsQ; h++)
                    {
                        RotateHeadSimd(posQ + (long)h * headDim, pCos, pSin, halfDim);
                    }

                    float* posK = k + (long)pos * nHeadsKv * headDim;
                    for (int h = 0; h < nHeadsKv; h++)
                    {
                        RotateHeadSimd(posK + (long)h * headDim, pCos, pSin, halfDim);
                    }
                }
            }
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHeadSimd(float* head, float* pCos, float* pSin, int halfDim)
    {
        float* second = head + halfDim;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && halfDim >= 16)
        {
            for (; i <= halfDim - 16; i += 16)
            {
                var v0 = Vector512.Load(head + i);
                var v1 = Vector512.Load(second + i);
                var c = Vector512.Load(pCos + i);
                var s = Vector512.Load(pSin + i);

                var r0 = v0 * c - v1 * s;
                var r1 = Avx512F.IsSupported
                    ? Avx512F.FusedMultiplyAdd(v0, s, v1 * c)
                    : (v0 * s + v1 * c);

                Vector512.Store(r0, head + i);
                Vector512.Store(r1, second + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= halfDim - 8)
        {
            for (; i <= halfDim - 8; i += 8)
            {
                var v0 = Vector256.Load(head + i);
                var v1 = Vector256.Load(second + i);
                var c = Vector256.Load(pCos + i);
                var s = Vector256.Load(pSin + i);

                var r0 = v0 * c - v1 * s;
                var r1 = Fma.IsSupported
                    ? Fma.MultiplyAdd(v0, s, v1 * c)
                    : (v0 * s + v1 * c);

                Vector256.Store(r0, head + i);
                Vector256.Store(r1, second + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= halfDim - 4)
        {
            for (; i <= halfDim - 4; i += 4)
            {
                var v0 = Vector128.Load(head + i);
                var v1 = Vector128.Load(second + i);
                var c = Vector128.Load(pCos + i);
                var s = Vector128.Load(pSin + i);

                var r0 = v0 * c - v1 * s;
                var r1 = Fma.IsSupported
                    ? Fma.MultiplyAdd(v0, s, v1 * c)
                    : (v0 * s + v1 * c);

                Vector128.Store(r0, head + i);
                Vector128.Store(r1, second + i);
            }
        }

        for (; i < halfDim; i++)
        {
            float c = pCos[i];
            float s = pSin[i];
            float v0 = head[i];
            float v1 = second[i];
            head[i] = v0 * c - v1 * s;
            second[i] = v0 * s + v1 * c;
        }
    }
}
