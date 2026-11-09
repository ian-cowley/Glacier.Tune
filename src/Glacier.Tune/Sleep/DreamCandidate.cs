namespace Glacier.Tune.Sleep;

/// <summary>
/// A synthetic dream rollout candidate generated during the Sleep consolidation cycle.
/// </summary>
public sealed record DreamCandidate(
    string PromptPrefix,
    string GeneratedDream,
    float[] QueryVector,
    float[] TargetTeacherOutput,
    float SemanticSimilarity,
    float LevenshteinRatio,
    float CompositeScore
);

/// <summary>
/// Telemetry results produced by a completed Sleep consolidation cycle.
/// </summary>
public sealed record ConsolidationResult(
    int ExpertSlotIndex,
    int TotalDreamsGenerated,
    int TotalDreamsAccepted,
    float AcceptanceRate,
    float InitialGkdLoss,
    float FinalGkdLoss,
    TimeSpan ElapsedDuration,
    bool SynapticPruningSucceeded
);
