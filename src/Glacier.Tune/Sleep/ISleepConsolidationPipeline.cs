namespace Glacier.Tune.Sleep;

using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Core contract for the 3-phase Sleep Consolidation Pipeline:
/// 1. Parameter Expansion (unlocking pre-allocated MoE slot)
/// 2. Knowledge Seeding &amp; Dreaming (SIMD gating + Generalized Knowledge Distillation)
/// 3. Synaptic Pruning (resetting volatile fast memory)
/// </summary>
public interface ISleepConsolidationPipeline : IDisposable
{
    /// <summary>
    /// Phase 1: Unlocks and prepares the next low-rank MoE expert slot.
    /// </summary>
    int ParameterExpansion(ReadOnlySpan<float> initialCentroid);

    /// <summary>
    /// Phase 2: Filters synthetic dream candidates via SIMD Levenshtein + Cosine similarity,
    /// then distills knowledge into the unlocked expert via GKD factorized optimization.
    /// </summary>
    Task<ConsolidationResult> ConsolidateKnowledgeAsync(
        int expertSlotIndex,
        DreamCandidate[] dreamBatch,
        int distillationSteps = 100,
        float learningRate = 0.20f,
        float qualityThreshold = 0.75f,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Phase 3: Synaptic Pruning: Clears volatile fast memory buffers to prepare for the subsequent wake cycle.
    /// </summary>
    void SynapticPruning();
}
