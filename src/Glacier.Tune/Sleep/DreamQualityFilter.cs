namespace Glacier.Tune.Sleep;

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Quality gating kernel for synthetic dreams, evaluating composite scores from
/// SIMD cosine embedding similarity and normalized Levenshtein edit distance.
/// Rejects hallucinated and degraded rollouts before Generalized Knowledge Distillation (GKD).
/// </summary>
public static unsafe class DreamQualityFilter
{
    /// <summary>
    /// Computes the composite dream quality score:
    /// <c>Score = alpha * CosineSim(e_dream, e_ref) + (1 - alpha) * (1 - Levenshtein(txt_dream, txt_ref) / max_len)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float ComputeScore(
        ReadOnlySpan<char> dreamText,
        ReadOnlySpan<char> referenceText,
        ReadOnlySpan<float> dreamEmbedding,
        ReadOnlySpan<float> referenceEmbedding,
        float alpha = 0.65f)
    {
        float cosSim = CosineSimilarity(dreamEmbedding, referenceEmbedding);
        int maxLen = Math.Max(dreamText.Length, referenceText.Length);
        float editRatio = maxLen == 0
            ? 1.0f
            : 1.0f - ((float)FastLevenshtein(dreamText, referenceText) / maxLen);

        return (alpha * cosSim) + ((1.0f - alpha) * editRatio);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = Math.Min(a.Length, b.Length);
        if (len <= 0) return 0.0f;

        double dot = 0.0, magA = 0.0, magB = 0.0;
        int i = 0;

        fixed (float* pA = &MemoryMarshal.GetReference(a))
        fixed (float* pB = &MemoryMarshal.GetReference(b))
        {
            if (Vector512.IsHardwareAccelerated && len >= Vector512<float>.Count)
            {
                var vDot = Vector512<float>.Zero;
                var vMagA = Vector512<float>.Zero;
                var vMagB = Vector512<float>.Zero;
                int step = Vector512<float>.Count;
                int limit = len - step;
                for (; i <= limit; i += step)
                {
                    var va = Vector512.Load(pA + i);
                    var vb = Vector512.Load(pB + i);
                    vDot = Vector512.FusedMultiplyAdd(va, vb, vDot);
                    vMagA = Vector512.FusedMultiplyAdd(va, va, vMagA);
                    vMagB = Vector512.FusedMultiplyAdd(vb, vb, vMagB);
                }
                dot += Vector512.Sum(vDot);
                magA += Vector512.Sum(vMagA);
                magB += Vector512.Sum(vMagB);
            }

            if (Vector256.IsHardwareAccelerated && len - i >= Vector256<float>.Count)
            {
                var vDot = Vector256<float>.Zero;
                var vMagA = Vector256<float>.Zero;
                var vMagB = Vector256<float>.Zero;
                int step = Vector256<float>.Count;
                int limit = len - step;
                for (; i <= limit; i += step)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    if (Fma.IsSupported)
                    {
                        vDot = Fma.MultiplyAdd(va, vb, vDot);
                        vMagA = Fma.MultiplyAdd(va, va, vMagA);
                        vMagB = Fma.MultiplyAdd(vb, vb, vMagB);
                    }
                    else
                    {
                        vDot += va * vb;
                        vMagA += va * va;
                        vMagB += vb * vb;
                    }
                }
                dot += Vector256.Sum(vDot);
                magA += Vector256.Sum(vMagA);
                magB += Vector256.Sum(vMagB);
            }

            if (Vector128.IsHardwareAccelerated && len - i >= Vector128<float>.Count)
            {
                var vDot = Vector128<float>.Zero;
                var vMagA = Vector128<float>.Zero;
                var vMagB = Vector128<float>.Zero;
                int step = Vector128<float>.Count;
                int limit = len - step;
                for (; i <= limit; i += step)
                {
                    var va = Vector128.Load(pA + i);
                    var vb = Vector128.Load(pB + i);
                    vDot = Vector128.FusedMultiplyAdd(va, vb, vDot);
                    vMagA = Vector128.FusedMultiplyAdd(va, va, vMagA);
                    vMagB = Vector128.FusedMultiplyAdd(vb, vb, vMagB);
                }
                dot += Vector128.Sum(vDot);
                magA += Vector128.Sum(vMagA);
                magB += Vector128.Sum(vMagB);
            }

            for (; i < len; i++)
            {
                double ai = pA[i];
                double bi = pB[i];
                dot += ai * bi;
                magA += ai * ai;
                magB += bi * bi;
            }
        }

        return (magA > 1e-12 && magB > 1e-12)
            ? (float)(dot / (Math.Sqrt(magA) * Math.Sqrt(magB)))
            : 0.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int FastLevenshtein(ReadOnlySpan<char> s, ReadOnlySpan<char> t)
    {
        if (s.IsEmpty) return t.Length;
        if (t.IsEmpty) return s.Length;

        int prefix = s.CommonPrefixLength(t);
        if (prefix > 0)
        {
            s = s.Slice(prefix);
            t = t.Slice(prefix);
        }

        int suffix = 0;
        int minLen = Math.Min(s.Length, t.Length);
        while (suffix < minLen && s[s.Length - 1 - suffix] == t[t.Length - 1 - suffix])
        {
            suffix++;
        }
        if (suffix > 0)
        {
            s = s.Slice(0, s.Length - suffix);
            t = t.Slice(0, t.Length - suffix);
        }

        if (s.IsEmpty) return t.Length;
        if (t.IsEmpty) return s.Length;

        if (s.Length < t.Length)
        {
            var tmp = s;
            s = t;
            t = tmp;
        }

        int n = s.Length;
        int m = t.Length;

        // 64-bit Myers bit-parallel DP for remaining pattern length <= 64
        if (m <= 64)
        {
            return MyersBitParallelLevenshtein(s, t);
        }

        const int StackLimit = 256;
        int rowLen = m + 1;

        if (rowLen <= StackLimit)
        {
            Span<int> prev = stackalloc int[rowLen];
            Span<int> curr = stackalloc int[rowLen];
            return LevenshteinCore(s, t, prev, curr);
        }

        int[] pooledPrev = ArrayPool<int>.Shared.Rent(rowLen);
        int[] pooledCurr = ArrayPool<int>.Shared.Rent(rowLen);
        try
        {
            return LevenshteinCore(s, t, pooledPrev.AsSpan(0, rowLen), pooledCurr.AsSpan(0, rowLen));
        }
        finally
        {
            ArrayPool<int>.Shared.Return(pooledPrev);
            ArrayPool<int>.Shared.Return(pooledCurr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int MyersBitParallelLevenshtein(ReadOnlySpan<char> s, ReadOnlySpan<char> t)
    {
        int m = t.Length;
        bool allAscii = true;
        for (int j = 0; j < m; j++)
        {
            if (t[j] >= 128)
            {
                allAscii = false;
                break;
            }
        }

        Span<ulong> peq = allAscii ? stackalloc ulong[128] : default;
        if (allAscii)
        {
            peq.Clear();
            for (int j = 0; j < m; j++)
            {
                peq[t[j]] |= 1UL << j;
            }
        }

        ulong pv = ~0UL;
        ulong mv = 0UL;
        int score = m;
        ulong mask = 1UL << (m - 1);

        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            ulong eq;
            if (allAscii)
            {
                eq = c < 128 ? peq[c] : 0UL;
            }
            else
            {
                eq = 0UL;
                for (int j = 0; j < m; j++)
                {
                    if (t[j] == c) eq |= 1UL << j;
                }
            }

            ulong xv = eq | mv;
            ulong xh = (((eq & pv) + pv) ^ pv) | eq;
            ulong ph = mv | ~(xh | pv);
            ulong mh = pv & xh;

            if ((ph & mask) != 0) score++;
            else if ((mh & mask) != 0) score--;

            ph = (ph << 1) | 1UL;
            mh <<= 1;

            pv = mh | ~(xv | ph);
            mv = ph & xv;
        }

        return score;
    }

    private static int LevenshteinCore(ReadOnlySpan<char> s, ReadOnlySpan<char> t, Span<int> prev, Span<int> curr)
    {
        int n = s.Length;
        int m = t.Length;

        for (int j = 0; j <= m; j++) prev[j] = j;

        for (int i = 1; i <= n; i++)
        {
            curr[0] = i;
            char sc = s[i - 1];
            for (int j = 1; j <= m; j++)
            {
                int cost = (sc == t[j - 1]) ? 0 : 1;
                curr[j] = Math.Min(Math.Min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            curr.Slice(0, m + 1).CopyTo(prev);
        }

        return prev[m];
    }
}
