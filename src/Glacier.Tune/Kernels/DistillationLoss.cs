namespace Glacier.Tune.Kernels;

using System;
using System.Buffers;
using System.Threading.Tasks;

/// <summary>
/// Knowledge-distillation and hidden-state alignment losses with exact analytical gradients.
/// </summary>
public static class DistillationLoss
{
    /// <summary>
    /// Computes blended temperature-scaled Logit KL divergence <c>alpha * tau^2 * KL(P_T || Q_S)</c>
    /// plus optional next-token Cross-Entropy <c>(1 - alpha) * NLL(z_S, target)</c>, writing <c>dStudentLogits</c>.
    /// </summary>
    public static double ComputeLogitKlAndCe(
        ReadOnlySpan<float> studentLogits,
        ReadOnlySpan<float> teacherLogits,
        ReadOnlySpan<int> targetTokens,
        int numTokens,
        int vocabSize,
        float temperature,
        float kdAlpha,
        Span<float> dStudentLogits)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        if (temperature <= 0.0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));

        int total = checked(numTokens * vocabSize);
        if (studentLogits.Length < total || teacherLogits.Length < total || dStudentLogits.Length < total)
            throw new ArgumentException("Logit buffer lengths do not match numTokens * vocabSize.");

        kdAlpha = Math.Clamp(kdAlpha, 0.0f, 1.0f);
        bool hasTargets = !targetTokens.IsEmpty && kdAlpha < 1.0f;
        int validTargets = 0;
        if (hasTargets)
        {
            for (int t = 0; t < numTokens; t++)
            {
                if ((uint)targetTokens[t] < (uint)vocabSize)
                    validTargets++;
            }
            if (validTargets == 0) hasTargets = false;
        }

        double invTemp = 1.0 / temperature;
        double klWeight = hasTargets ? kdAlpha : 1.0;
        double ceWeight = hasTargets ? (1.0 - kdAlpha) : 0.0;

        if (numTokens > 1 && total >= 16_384)
        {
            double[] tokenLosses = new double[numTokens];
            unsafe
            {
                fixed (float* pZT = teacherLogits)
                fixed (float* pZS = studentLogits)
                fixed (float* pDZ = dStudentLogits)
                fixed (int* pTgt = targetTokens)
                {
                    nint ztAddr = (nint)pZT;
                    nint zsAddr = (nint)pZS;
                    nint dzAddr = (nint)pDZ;
                    nint tgtAddr = (nint)pTgt;

                    Parallel.For(
                        0,
                        numTokens,
                        () => (PT: ArrayPool<double>.Shared.Rent(vocabSize), QS: ArrayPool<double>.Shared.Rent(vocabSize)),
                        (t, _, scratch) =>
                        {
                            int off = t * vocabSize;
                            var zT = new ReadOnlySpan<float>((float*)ztAddr + off, vocabSize);
                            var zS = new ReadOnlySpan<float>((float*)zsAddr + off, vocabSize);
                            var dzS = new Span<float>((float*)dzAddr + off, vocabSize);
                            int targetTok = hasTargets ? ((int*)tgtAddr)[t] : -1;
                            tokenLosses[t] = ComputeSingleTokenKlAndCe(
                                zT, zS, dzS, targetTok, hasTargets, validTargets,
                                numTokens, vocabSize, temperature, invTemp, klWeight, ceWeight,
                                scratch.PT, scratch.QS);
                            return scratch;
                        },
                        scratch =>
                        {
                            ArrayPool<double>.Shared.Return(scratch.PT);
                            ArrayPool<double>.Shared.Return(scratch.QS);
                        });
                }
            }

            double sum = 0.0;
            for (int t = 0; t < numTokens; t++)
                sum += tokenLosses[t];
            return sum;
        }

        double[] pT = ArrayPool<double>.Shared.Rent(vocabSize);
        double[] qS = ArrayPool<double>.Shared.Rent(vocabSize);
        try
        {
            double totalLoss = 0.0;
            for (int t = 0; t < numTokens; t++)
            {
                int off = t * vocabSize;
                ReadOnlySpan<float> zT = teacherLogits.Slice(off, vocabSize);
                ReadOnlySpan<float> zS = studentLogits.Slice(off, vocabSize);
                Span<float> dzS = dStudentLogits.Slice(off, vocabSize);
                int targetTok = hasTargets ? targetTokens[t] : -1;
                totalLoss += ComputeSingleTokenKlAndCe(
                    zT, zS, dzS, targetTok, hasTargets, validTargets,
                    numTokens, vocabSize, temperature, invTemp, klWeight, ceWeight,
                    pT, qS);
            }
            return totalLoss;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(pT);
            ArrayPool<double>.Shared.Return(qS);
        }
    }

    private static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
    }

    private static double ComputeSingleTokenKlAndCe(
        ReadOnlySpan<float> zT,
        ReadOnlySpan<float> zS,
        Span<float> dzS,
        int target,
        bool hasTargets,
        int validTargets,
        int numTokens,
        int vocabSize,
        float temperature,
        double invTemp,
        double klWeight,
        double ceWeight,
        double[] pT,
        double[] qS)
    {
        int argMaxT = 0;
        double maxRawT = zT[0];
        int argMaxS = 0;
        double maxRawS = zS[0];
        for (int v = 1; v < vocabSize; v++)
        {
            if (zT[v] > maxRawT)
            {
                maxRawT = zT[v];
                argMaxT = v;
            }
            if (zS[v] > maxRawS)
            {
                maxRawS = zS[v];
                argMaxS = v;
            }
        }

        double maxT = maxRawT * invTemp;
        double maxS = maxRawS * invTemp;

        double sumOtherT = 0.0;
        double sumOtherS = 0.0;
        for (int v = 0; v < vocabSize; v++)
        {
            if (v != argMaxT)
            {
                double et = Math.Exp(zT[v] * invTemp - maxT);
                pT[v] = et;
                sumOtherT += et;
            }
            else
            {
                pT[v] = 1.0;
            }

            if (v != argMaxS)
            {
                double es = Math.Exp(zS[v] * invTemp - maxS);
                qS[v] = es;
                sumOtherS += es;
            }
            else
            {
                qS[v] = 1.0;
            }
        }

        double logZOffsetT = Log1p(sumOtherT);
        double logZOffsetS = Log1p(sumOtherS);
        double invSumT = 1.0 / (1.0 + sumOtherT);
        double invSumS = 1.0 / (1.0 + sumOtherS);
        double klToken = 0.0;
        double gradKlScale = klWeight * temperature / numTokens;

        for (int v = 0; v < vocabSize; v++)
        {
            double pt = (v == argMaxT) ? invSumT : pT[v] * invSumT;
            double qs = (v == argMaxS) ? invSumS : qS[v] * invSumS;
            if (pt > 0.0)
            {
                double logPt = (zT[v] * invTemp - maxT) - logZOffsetT;
                double logQs = (zS[v] * invTemp - maxS) - logZOffsetS;
                klToken += pt * (logPt - logQs);
            }

            double diff = (v == argMaxS && v == argMaxT)
                ? (sumOtherT - sumOtherS) * invSumS * invSumT
                : (qs - pt);
            dzS[v] = (float)(gradKlScale * diff);
        }

        double loss = klWeight * (temperature * (double)temperature / numTokens) * klToken;

        if (hasTargets && (uint)target < (uint)vocabSize)
        {
            double sumOtherRawS = 0.0;
            for (int v = 0; v < vocabSize; v++)
            {
                if (v != argMaxS)
                {
                    double e = Math.Exp(zS[v] - maxRawS);
                    qS[v] = e;
                    sumOtherRawS += e;
                }
                else
                {
                    qS[v] = 1.0;
                }
            }

            double invRawSum = 1.0 / (1.0 + sumOtherRawS);
            double nll = Log1p(sumOtherRawS) - (zS[target] - maxRawS);
            loss += (ceWeight / validTargets) * nll;

            double gradCeScale = ceWeight / validTargets;
            for (int v = 0; v < vocabSize; v++)
            {
                double g;
                if (v == target && v == argMaxS)
                {
                    g = -sumOtherRawS * invRawSum;
                }
                else
                {
                    double prob = qS[v] * invRawSum;
                    g = prob - (v == target ? 1.0 : 0.0);
                }
                dzS[v] += (float)(gradCeScale * g);
            }
        }

        return loss;
    }

    /// <summary>
    /// Computes weighted Hidden-State MSE + Cosine distance loss and exact analytical gradient <c>dStudentHidden</c>.
    /// </summary>
    public static double ComputeHiddenMseAndCosine(
        ReadOnlySpan<float> studentHidden,
        ReadOnlySpan<float> teacherHidden,
        int numTokens,
        int hiddenSize,
        float mseWeight,
        float cosineWeight,
        Span<float> dStudentHidden)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);

        int total = checked(numTokens * hiddenSize);
        if (studentHidden.Length < total || teacherHidden.Length < total || dStudentHidden.Length < total)
            throw new ArgumentException("Hidden buffer lengths do not match numTokens * hiddenSize.");

        dStudentHidden.Slice(0, total).Clear();

        double mseScale = (double)mseWeight / total;
        double mseGradScale = 2.0 * mseWeight / total;
        double cosScale = (double)cosineWeight / numTokens;
        double totalLoss = 0.0;

        for (int t = 0; t < numTokens; t++)
        {
            int off = t * hiddenSize;
            double dot = 0.0;
            double normSSq = 0.0;
            double normTSq = 0.0;
            double diffSq = 0.0;

            for (int h = 0; h < hiddenSize; h++)
            {
                double s = studentHidden[off + h];
                double r = teacherHidden[off + h];
                double d = s - r;
                diffSq += d * d;
                dot += s * r;
                normSSq += s * s;
                normTSq += r * r;

                if (mseWeight != 0.0f)
                    dStudentHidden[off + h] = (float)(mseGradScale * d);
            }

            totalLoss += mseScale * diffSq;

            if (cosineWeight != 0.0f)
            {
                const double eps = 1e-8;
                double normS = Math.Sqrt(normSSq + eps);
                double normT = Math.Sqrt(normTSq + eps);
                double denom = normS * normT;
                double cos = dot / denom;
                totalLoss += cosScale * (1.0 - cos);

                // d(1 - cos)/d(s_h) = -(r_h / (||s||*||t||) - cos * s_h / ||s||^2)
                double invDenom = 1.0 / denom;
                double cosOverNormSSq = cos / (normSSq + eps);
                for (int h = 0; h < hiddenSize; h++)
                {
                    double s = studentHidden[off + h];
                    double r = teacherHidden[off + h];
                    double gCos = cosScale * (cosOverNormSSq * s - r * invDenom);
                    dStudentHidden[off + h] += (float)gCos;
                }
            }
        }

        return totalLoss;
    }
}
