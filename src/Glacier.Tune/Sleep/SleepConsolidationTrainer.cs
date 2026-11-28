namespace Glacier.Tune.Sleep;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Tune.Expansion;

/// <summary>
/// Autonomous Sleep Consolidation Trainer implementing Google Research's 3-phase continual learning cycle:
/// 1. Parameter Expansion: Unlocks pre-allocated low-rank MoE expert slots in <see cref="PreallocatedMoeExpansionBank"/>.
/// 2. Knowledge Seeding &amp; Synthetic Dreaming: Filters dream candidates via <see cref="DreamQualityFilter"/>,
///    distilling fast-memory knowledge into the newly unlocked MoE expert using Generalized Knowledge Distillation (GKD)
///    with configurable forward/reverse KL-beta divergence regularization.
/// 3. Synaptic Pruning: Resets volatile fast memory to provide a clean slate for the next Wake cycle.
/// </summary>
public sealed unsafe class SleepConsolidationTrainer : ISleepConsolidationPipeline
{
    private readonly PreallocatedMoeExpansionBank _expansionBank;
    private readonly Action? _synapticPruningCallback;
    private int _disposed;

    public PreallocatedMoeExpansionBank ExpansionBank => _expansionBank;

    /// <summary>
    /// GKD KL blending factor <c>beta</c> in <c>[0, 1]</c>:
    /// <c>0.0</c> = pure forward KL <c>KL(P_T || Q_S)</c>,
    /// <c>1.0</c> = pure reverse KL <c>KL(Q_S || P_T)</c>,
    /// intermediate values blend <c>(1 - beta) * KL(P_T || Q_S) + beta * KL(Q_S || P_T)</c>.
    /// </summary>
    public float KlBeta { get; set; } = 0.5f;

    /// <summary>
    /// Weight applied to the temperature-scaled GKD KL divergence term alongside MSE feature distillation.
    /// </summary>
    public float KlWeight { get; set; } = 0.25f;

    /// <summary>
    /// Softmax temperature <c>tau</c> (&gt; 0) used when computing teacher/student KL divergence.
    /// </summary>
    public float KlTemperature { get; set; } = 1.0f;

    public SleepConsolidationTrainer(
        PreallocatedMoeExpansionBank expansionBank,
        Action? synapticPruningCallback = null,
        float klBeta = 0.5f,
        float klWeight = 0.25f,
        float klTemperature = 1.0f)
    {
        _expansionBank = expansionBank ?? throw new ArgumentNullException(nameof(expansionBank));
        _synapticPruningCallback = synapticPruningCallback;
        KlBeta = Math.Clamp(klBeta, 0.0f, 1.0f);
        KlWeight = Math.Max(0.0f, klWeight);
        KlTemperature = klTemperature > 0.0f ? klTemperature : 1.0f;
    }

    /// <summary>
    /// Phase 1: Parameter Expansion — Unlocks the next dormant low-rank MoE expert slot.
    /// </summary>
    public int ParameterExpansion(ReadOnlySpan<float> initialCentroid)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _expansionBank.UnlockNextExpertSlot(initialCentroid);
    }

    /// <summary>
    /// Phase 2: Knowledge Seeding &amp; Synthetic Dreaming — Filters dream candidates and distills into MoE slot asynchronously.
    /// </summary>
    public Task<ConsolidationResult> ConsolidateKnowledgeAsync(
        int expertSlotIndex,
        DreamCandidate[] dreamBatch,
        int distillationSteps = 100,
        float learningRate = 0.20f,
        float qualityThreshold = 0.75f,
        CancellationToken cancellationToken = default)
        => ConsolidateKnowledgeAsync(
            expertSlotIndex,
            dreamBatch,
            distillationSteps,
            learningRate,
            qualityThreshold,
            KlBeta,
            cancellationToken);

    /// <summary>
    /// Phase 2: Filters dream candidates and distills into MoE slot asynchronously using explicit GKD <paramref name="klBeta"/>.
    /// </summary>
    public Task<ConsolidationResult> ConsolidateKnowledgeAsync(
        int expertSlotIndex,
        DreamCandidate[] dreamBatch,
        int distillationSteps,
        float learningRate,
        float qualityThreshold,
        float klBeta,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(dreamBatch);
        if (distillationSteps <= 0) throw new ArgumentOutOfRangeException(nameof(distillationSteps));
        ArgumentOutOfRangeException.ThrowIfNegative(learningRate);

        float clampedBeta = Math.Clamp(klBeta, 0.0f, 1.0f);
        float klWeight = Math.Max(0.0f, KlWeight);
        float klTemp = KlTemperature > 0.0f ? KlTemperature : 1.0f;

        return Task.Run(() =>
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            cancellationToken.ThrowIfCancellationRequested();

            Stopwatch sw = Stopwatch.StartNew();
            int totalDreams = dreamBatch.Length;
            var acceptedDreams = new List<(float[] Key, float[] TeacherOut)>(totalDreams);

            // 1. SIMD Quality Filtering: Reject hallucinated or low-fidelity dreams
            for (int i = 0; i < totalDreams; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                DreamCandidate c = dreamBatch[i];
                ReadOnlySpan<float> dreamEmb = c.DreamEmbedding ?? c.QueryVector;
                float score = (c.CompositeScore > 0f)
                    ? c.CompositeScore
                    : DreamQualityFilter.ComputeScore(c.GeneratedDream, c.PromptPrefix, dreamEmb, c.TargetTeacherOutput);

                if (score >= qualityThreshold && c.QueryVector != null && c.TargetTeacherOutput != null)
                {
                    acceptedDreams.Add((c.QueryVector, c.TargetTeacherOutput));
                }
            }

            int acceptedCount = acceptedDreams.Count;
            float acceptanceRate = totalDreams > 0 ? (float)acceptedCount / totalDreams : 0.0f;

            if (acceptedCount == 0)
            {
                sw.Stop();
                return new ConsolidationResult(
                    expertSlotIndex,
                    totalDreams,
                    0,
                    0.0f,
                    0.0f,
                    0.0f,
                    sw.Elapsed,
                    SynapticPruningSucceeded: false);
            }

            // 2. Generalized Knowledge Distillation (GKD) Factorized Optimization
            int dim = _expansionBank.Dimension;
            int rank = _expansionBank.Rank;
            float* aPtr = _expansionBank.GetMatrixAPtr(expertSlotIndex);
            float* bPtr = _expansionBank.GetMatrixBPtr(expertSlotIndex);

            float[] hiddenR = ArrayPool<float>.Shared.Rent(rank);
            float[] studentOut = ArrayPool<float>.Shared.Rent(dim);
            float[] errOut = ArrayPool<float>.Shared.Rent(dim);
            float[] gradB_k = ArrayPool<float>.Shared.Rent(rank);
            double[] pTeacher = klWeight > 0.0f ? ArrayPool<double>.Shared.Rent(dim) : [];
            double[] qStudent = klWeight > 0.0f ? ArrayPool<double>.Shared.Rent(dim) : [];

            try
            {
                float initialLoss = ComputeBatchLoss(aPtr, bPtr, acceptedDreams, dim, rank, clampedBeta, klWeight, klTemp, hiddenR, studentOut, pTeacher, qStudent);

                fixed (float* pHidden = hiddenR)
                fixed (float* pStudent = studentOut)
                fixed (float* pErr = errOut)
                fixed (float* pGradB = gradB_k)
                {
                    for (int step = 0; step < distillationSteps; step++)
                    {
                        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                        cancellationToken.ThrowIfCancellationRequested();

                        for (int sampleIdx = 0; sampleIdx < acceptedCount; sampleIdx++)
                        {
                            var (kVec, targetVec) = acceptedDreams[sampleIdx];
                            if (kVec.Length < dim || targetVec.Length < dim) continue;

                            fixed (float* pK = kVec)
                            fixed (float* pT = targetVec)
                            {
                                // 1. Forward hidden = B_e * kVec ([Rank x Dimension] * [Dimension x 1])
                                for (int r = 0; r < rank; r++)
                                {
                                    pHidden[r] = DotProductSimd(bPtr + (long)r * dim, pK, dim);
                                }

                                // 2. Forward student = A_e * hidden ([Dimension x Rank] * [Rank x 1])
                                for (int d = 0; d < dim; d++)
                                {
                                    float sum = DotProductSimd(aPtr + (long)d * rank, pHidden, rank);
                                    pStudent[d] = sum;
                                    pErr[d] = sum - pT[d];
                                }

                                // Optional GKD KL(P_T || Q_S) / KL(Q_S || P_T) gradient term
                                if (klWeight > 0.0f)
                                {
                                    AccumulateKlGradient(
                                        pStudent, pT, pErr, dim, clampedBeta, klWeight, klTemp, pTeacher, qStudent);
                                }

                                // 3. Backward gradHidden = A_e^T * errOut
                                for (int r = 0; r < rank; r++) pGradB[r] = 0.0f;
                                for (int d = 0; d < dim; d++)
                                {
                                    float ed = pErr[d];
                                    if (ed != 0.0f)
                                    {
                                        AxpySimd(pGradB, aPtr + (long)d * rank, ed, rank);
                                    }
                                }

                                // 4. Gradient update: A_e -= lr * errOut * hidden^T
                                for (int d = 0; d < dim; d++)
                                {
                                    float scaleA = -learningRate * pErr[d];
                                    if (scaleA != 0.0f)
                                    {
                                        AxpySimd(aPtr + (long)d * rank, pHidden, scaleA, rank);
                                    }
                                }

                                // 5. Gradient update: B_e -= lr * gradHidden * kVec^T
                                for (int r = 0; r < rank; r++)
                                {
                                    float scaleB = -learningRate * pGradB[r];
                                    if (scaleB != 0.0f)
                                    {
                                        AxpySimd(bPtr + (long)r * dim, pK, scaleB, dim);
                                    }
                                }
                            }
                        }
                    }
                }

                float finalLoss = ComputeBatchLoss(aPtr, bPtr, acceptedDreams, dim, rank, clampedBeta, klWeight, klTemp, hiddenR, studentOut, pTeacher, qStudent);

                // Mark slot as active for live inference
                _expansionBank.ActivateConsolidatedSlot(expertSlotIndex);

                // Phase 3: Synaptic Pruning
                SynapticPruning();

                sw.Stop();
                return new ConsolidationResult(
                    expertSlotIndex,
                    totalDreams,
                    acceptedCount,
                    acceptanceRate,
                    initialLoss,
                    finalLoss,
                    sw.Elapsed,
                    SynapticPruningSucceeded: true);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(hiddenR);
                ArrayPool<float>.Shared.Return(studentOut);
                ArrayPool<float>.Shared.Return(errOut);
                ArrayPool<float>.Shared.Return(gradB_k);
                if (pTeacher.Length > 0) ArrayPool<double>.Shared.Return(pTeacher);
                if (qStudent.Length > 0) ArrayPool<double>.Shared.Return(qStudent);
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Phase 3: Synaptic Pruning — Resets volatile fast memory to provide a clean slate for the next Wake cycle.
    /// </summary>
    public void SynapticPruning()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _synapticPruningCallback?.Invoke();
    }

    internal static double ComputeKlLossAndGradient(
        float* student,
        float* teacher,
        float* gradOut,
        int dim,
        float klBeta,
        float klWeight,
        float temperature,
        double[] pT,
        double[] qS,
        bool accumulateGradient)
    {
        if (dim <= 0 || klWeight <= 0.0f) return 0.0;

        double invTemp = 1.0 / temperature;
        double maxT = double.NegativeInfinity;
        double maxS = double.NegativeInfinity;
        for (int d = 0; d < dim; d++)
        {
            double tv = teacher[d] * invTemp;
            double sv = student[d] * invTemp;
            if (tv > maxT) maxT = tv;
            if (sv > maxS) maxS = sv;
        }

        double sumT = 0.0;
        double sumS = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double et = Math.Exp(teacher[d] * invTemp - maxT);
            double es = Math.Exp(student[d] * invTemp - maxS);
            pT[d] = et;
            qS[d] = es;
            sumT += et;
            sumS += es;
        }

        double invSumT = 1.0 / sumT;
        double invSumS = 1.0 / sumS;
        double klFwd = 0.0;
        double klRev = 0.0;

        for (int d = 0; d < dim; d++)
        {
            double pt = pT[d] * invSumT;
            double qs = qS[d] * invSumS;
            pT[d] = pt;
            qS[d] = qs;

            double logRatio = Math.Log(Math.Max(1e-30, pt)) - Math.Log(Math.Max(1e-30, qs));
            if (pt > 0.0) klFwd += pt * logRatio;
            if (qs > 0.0) klRev += qs * (-logRatio);
        }

        double tau2 = (double)temperature * temperature;
        double blendedKl = ((1.0 - klBeta) * klFwd + klBeta * klRev) * tau2;

        if (accumulateGradient)
        {
            double fwdScale = (1.0 - klBeta) * klWeight * temperature;
            double revScale = klBeta * klWeight * temperature;
            for (int d = 0; d < dim; d++)
            {
                double pt = pT[d];
                double qs = qS[d];
                double gFwd = fwdScale * (qs - pt);
                double gRev = 0.0;
                if (klBeta > 0.0f && qs > 0.0)
                {
                    double logQOverP = Math.Log(Math.Max(1e-30, qs)) - Math.Log(Math.Max(1e-30, pt));
                    gRev = revScale * qs * (logQOverP - klRev);
                }
                gradOut[d] += (float)(gFwd + gRev);
            }
        }

        return klWeight * blendedKl;
    }

    private static void AccumulateKlGradient(
        float* student,
        float* teacher,
        float* errOut,
        int dim,
        float klBeta,
        float klWeight,
        float temperature,
        double[] pT,
        double[] qS)
    {
        _ = ComputeKlLossAndGradient(
            student, teacher, errOut, dim, klBeta, klWeight, temperature, pT, qS, accumulateGradient: true);
    }

    private static float ComputeBatchLoss(
        float* aPtr,
        float* bPtr,
        List<(float[] Key, float[] TeacherOut)> dreams,
        int dim,
        int rank,
        float klBeta,
        float klWeight,
        float klTemp,
        float[] hiddenBuf,
        float[] studentBuf,
        double[] pTeacher,
        double[] qStudent)
    {
        double totalLoss = 0.0;

        fixed (float* pHidden = hiddenBuf)
        fixed (float* pStudent = studentBuf)
        {
            for (int i = 0; i < dreams.Count; i++)
            {
                var (kVec, targetVec) = dreams[i];
                if (kVec.Length < dim || targetVec.Length < dim) continue;

                fixed (float* pK = kVec)
                fixed (float* pT = targetVec)
                {
                    for (int r = 0; r < rank; r++)
                    {
                        pHidden[r] = DotProductSimd(bPtr + (long)r * dim, pK, dim);
                    }

                    double mseSum = 0.0;
                    for (int d = 0; d < dim; d++)
                    {
                        float s = DotProductSimd(aPtr + (long)d * rank, pHidden, rank);
                        pStudent[d] = s;
                        double diff = (double)s - pT[d];
                        mseSum += 0.5 * diff * diff;
                    }

                    double sampleLoss = mseSum;
                    if (klWeight > 0.0f)
                    {
                        sampleLoss += ComputeKlLossAndGradient(
                            pStudent, pT, null, dim, klBeta, klWeight, klTemp, pTeacher, qStudent, accumulateGradient: false);
                    }
                    totalLoss += sampleLoss;
                }
            }
        }

        return (float)(totalLoss / Math.Max(1, dreams.Count));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(float* a, float* b, int count)
    {
        int i = 0;
        float sum = 0.0f;

        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var acc = Vector512<float>.Zero;
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                acc = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc);
            }
            sum += Vector512.Sum(acc);
        }
        if (Vector256.IsHardwareAccelerated && count - i >= Vector256<float>.Count)
        {
            var acc = Vector256<float>.Zero;
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                acc = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc) : acc + va * vb;
            }
            sum += Vector256.Sum(acc);
        }
        if (Vector128.IsHardwareAccelerated && count - i >= Vector128<float>.Count)
        {
            var acc = Vector128<float>.Zero;
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                acc = Vector128.FusedMultiplyAdd(Vector128.Load(a + i), Vector128.Load(b + i), acc);
            }
            sum += Vector128.Sum(acc);
        }
        for (; i < count; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void AxpySimd(float* dst, float* src, float scale, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector512.Load(dst + i);
                var vSrc = Vector512.Load(src + i);
                Vector512.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && count - i >= Vector256<float>.Count)
        {
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector256.Load(dst + i);
                var vSrc = Vector256.Load(src + i);
                (Fma.IsSupported ? Fma.MultiplyAdd(vSrc, vScale, vDst) : vDst + vSrc * vScale).Store(dst + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && count - i >= Vector128<float>.Count)
        {
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector128.Load(dst + i);
                var vSrc = Vector128.Load(src + i);
                Vector128.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += scale * src[i];
        }
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
        GC.SuppressFinalize(this);
    }
}
