namespace Glacier.Tune.Sleep;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Quality gating kernel for synthetic dreams, evaluating composite scores from
/// cosine embedding similarity and normalized Levenshtein edit distance.
/// Rejects hallucinated and degraded rollouts before Generalized Knowledge Distillation (GKD).
/// </summary>
public static class DreamQualityFilter
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
        if (maxLen == 0) return 1.0f;

        int editDist = FastLevenshtein(dreamText, referenceText);
        float editRatio = 1.0f - ((float)editDist / maxLen);

        return (alpha * cosSim) + ((1.0f - alpha) * editRatio);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = Math.Min(a.Length, b.Length);
        float dot = 0.0f, magA = 0.0f, magB = 0.0f;

        for (int i = 0; i < len; i++)
        {
            float ai = a[i];
            float bi = b[i];
            dot += ai * bi;
            magA += ai * ai;
            magB += bi * bi;
        }

        return (magA > 1e-12f && magB > 1e-12f)
            ? dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB))
            : 0.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int FastLevenshtein(ReadOnlySpan<char> s, ReadOnlySpan<char> t)
    {
        int n = s.Length;
        int m = t.Length;
        if (n == 0) return m;
        if (m == 0) return n;

        // Use heap allocation if strings exceed safe stack limit, otherwise stackalloc
        const int StackLimit = 256;
        int rowLen = m + 1;

        if (rowLen <= StackLimit)
        {
            Span<int> prev = stackalloc int[rowLen];
            Span<int> curr = stackalloc int[rowLen];
            return LevenshteinCore(s, t, prev, curr);
        }
        else
        {
            int[] prev = new int[rowLen];
            int[] curr = new int[rowLen];
            return LevenshteinCore(s, t, prev, curr);
        }
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
