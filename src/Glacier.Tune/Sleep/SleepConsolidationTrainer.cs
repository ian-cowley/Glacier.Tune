namespace Glacier.Tune.Sleep;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Tune.Expansion;

/// <summary>
/// Autonomous Sleep Consolidation Trainer implementing Google Research's 3-phase continual learning cycle:
/// 1. Parameter Expansion: Unlocks pre-allocated low-rank MoE expert slots in <see cref="PreallocatedMoeExpansionBank"/>.
/// 2. Knowledge Seeding &amp; Synthetic Dreaming: Filters dream candidates via <see cref="DreamQualityFilter"/>,
///    distilling fast-memory knowledge into the newly unlocked MoE expert using Generalized Knowledge Distillation (GKD).
/// 3. Synaptic Pruning: Resets volatile fast memory to provide a clean slate for the next Wake cycle.
/// </summary>
public sealed unsafe class SleepConsolidationTrainer : ISleepConsolidationPipeline
{
    private readonly PreallocatedMoeExpansionBank _expansionBank;
    private readonly Action? _synapticPruningCallback;
    private bool _disposed;

    public PreallocatedMoeExpansionBank ExpansionBank => _expansionBank;

    public SleepConsolidationTrainer(
        PreallocatedMoeExpansionBank expansionBank,
        Action? synapticPruningCallback = null)
    {
        _expansionBank = expansionBank ?? throw new ArgumentNullException(nameof(expansionBank));
        _synapticPruningCallback = synapticPruningCallback;
    }

    /// <summary>
    /// Phase 1: Parameter Expansion — Unlocks the next dormant low-rank MoE expert slot.
    /// </summary>
    public int ParameterExpansion(ReadOnlySpan<float> initialCentroid)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _expansionBank.UnlockNextExpertSlot(initialCentroid);
    }

    /// <summary>
    /// Phase 2: Knowledge Seeding &amp; Synthetic Dreaming — Filters dream candidates and distills into MoE slot.
    /// </summary>
    public Task<ConsolidationResult> ConsolidateKnowledgeAsync(
        int expertSlotIndex,
        DreamCandidate[] dreamBatch,
        int distillationSteps = 100,
        float learningRate = 0.20f,
        float qualityThreshold = 0.75f,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dreamBatch);
        if (distillationSteps <= 0) throw new ArgumentOutOfRangeException(nameof(distillationSteps));
        ArgumentOutOfRangeException.ThrowIfNegative(learningRate);

        Stopwatch sw = Stopwatch.StartNew();
        int totalDreams = dreamBatch.Length;
        var acceptedDreams = new List<(float[] Key, float[] TeacherOut)>(totalDreams);

        // 1. SIMD Quality Filtering: Reject hallucinated or low-fidelity dreams
        for (int i = 0; i < totalDreams; i++)
        {
            DreamCandidate c = dreamBatch[i];
            float score = (c.CompositeScore > 0f)
                ? c.CompositeScore
                : DreamQualityFilter.ComputeScore(c.GeneratedDream, c.PromptPrefix, c.TargetTeacherOutput, c.TargetTeacherOutput);

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
            return Task.FromResult(new ConsolidationResult(
                expertSlotIndex,
                totalDreams,
                0,
                0.0f,
                0.0f,
                0.0f,
                sw.Elapsed,
                SynapticPruningSucceeded: false));
        }

        // 2. Generalized Knowledge Distillation (GKD) Factorized Optimization
        int dim = _expansionBank.Dimension;
        int rank = _expansionBank.Rank;
        float* aPtr = _expansionBank.GetMatrixAPtr(expertSlotIndex);
        float* bPtr = _expansionBank.GetMatrixBPtr(expertSlotIndex);

        float initialLoss = ComputeBatchLoss(aPtr, bPtr, acceptedDreams, dim, rank);
        float finalLoss = initialLoss;

        float[] hiddenR = new float[rank];
        float[] studentOut = new float[dim];
        float[] errOut = new float[dim];
        float[] gradB_k = new float[rank];

        fixed (float* pHidden = hiddenR)
        fixed (float* pStudent = studentOut)
        fixed (float* pErr = errOut)
        fixed (float* pGradB = gradB_k)
        {
            for (int step = 0; step < distillationSteps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                for (int sampleIdx = 0; sampleIdx < acceptedCount; sampleIdx++)
                {
                    var (kVec, targetVec) = acceptedDreams[sampleIdx];

                    fixed (float* pK = kVec)
                    fixed (float* pT = targetVec)
                    {
                        // 1. Forward hidden = B_e * kVec ([Rank x Dimension] * [Dimension x 1])
                        for (int r = 0; r < rank; r++)
                        {
                            float* bRow = bPtr + (r * dim);
                            float sum = 0.0f;
                            for (int c = 0; c < dim; c++) sum += bRow[c] * pK[c];
                            pHidden[r] = sum;
                        }

                        // 2. Forward student = A_e * hidden ([Dimension x Rank] * [Rank x 1])
                        for (int d = 0; d < dim; d++)
                        {
                            float* aRow = aPtr + (d * rank);
                            float sum = 0.0f;
                            for (int r = 0; r < rank; r++) sum += aRow[r] * pHidden[r];
                            pStudent[d] = sum;
                            pErr[d] = sum - pT[d];
                        }

                        // 3. Backward gradHidden = A_e^T * errOut
                        for (int r = 0; r < rank; r++)
                        {
                            float sum = 0.0f;
                            for (int d = 0; d < dim; d++)
                            {
                                sum += aPtr[d * rank + r] * pErr[d];
                            }
                            pGradB[r] = sum;
                        }

                        // 4. Gradient update: A_e -= lr * errOut * hidden^T
                        for (int d = 0; d < dim; d++)
                        {
                            float ed = pErr[d];
                            float* aRow = aPtr + (d * rank);
                            for (int r = 0; r < rank; r++)
                            {
                                aRow[r] -= learningRate * ed * pHidden[r];
                            }
                        }

                        // 5. Gradient update: B_e -= lr * gradHidden * kVec^T
                        for (int r = 0; r < rank; r++)
                        {
                            float gr = pGradB[r];
                            float* bRow = bPtr + (r * dim);
                            for (int c = 0; c < dim; c++)
                            {
                                bRow[c] -= learningRate * gr * pK[c];
                            }
                        }
                    }
                }
            }
        }

        finalLoss = ComputeBatchLoss(aPtr, bPtr, acceptedDreams, dim, rank);

        // Mark slot as active for live inference
        _expansionBank.ActivateConsolidatedSlot(expertSlotIndex);

        // Phase 3: Synaptic Pruning
        SynapticPruning();

        sw.Stop();
        return Task.FromResult(new ConsolidationResult(
            expertSlotIndex,
            totalDreams,
            acceptedCount,
            acceptanceRate,
            initialLoss,
            finalLoss,
            sw.Elapsed,
            SynapticPruningSucceeded: true));
    }

    /// <summary>
    /// Phase 3: Synaptic Pruning — Resets volatile fast memory to provide a clean slate for the next Wake cycle.
    /// </summary>
    public void SynapticPruning()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _synapticPruningCallback?.Invoke();
    }

    private static float ComputeBatchLoss(
        float* aPtr,
        float* bPtr,
        List<(float[] Key, float[] TeacherOut)> dreams,
        int dim,
        int rank)
    {
        float totalLoss = 0.0f;
        Span<float> hidden = stackalloc float[rank];

        for (int i = 0; i < dreams.Count; i++)
        {
            var (kVec, targetVec) = dreams[i];
            for (int r = 0; r < rank; r++)
            {
                float* bRow = bPtr + (r * dim);
                float sum = 0.0f;
                for (int c = 0; c < dim; c++) sum += bRow[c] * kVec[c];
                hidden[r] = sum;
            }

            for (int d = 0; d < dim; d++)
            {
                float* aRow = aPtr + (d * rank);
                float sum = 0.0f;
                for (int r = 0; r < rank; r++) sum += aRow[r] * hidden[r];
                float diff = sum - targetVec[d];
                totalLoss += diff * diff;
            }
        }

        return totalLoss / Math.Max(1, dreams.Count);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
