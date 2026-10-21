namespace Glacier.Tune.Config;

/// <summary>
/// Fine-tuning hyperparameters and execution arguments.
/// </summary>
public sealed class TrainingArguments
{
    public float LearningRate { get; set; } = 2e-4f;
    public int Epochs { get; set; } = 3;
    public int BatchSize { get; set; } = 1;
    public int GradientAccumulationSteps { get; set; } = 8;
    public int MaxSeqLength { get; set; } = 1024;
    public float WarmupRatio { get; set; } = 0.03f;
    public int LoggingSteps { get; set; } = 5;
    public int SaveSteps { get; set; } = 15;
    public string OutputDir { get; set; } = "./output";
    public string LrSchedulerType { get; set; } = "cosine";
    public string Device { get; set; } = "auto";
    public int? MaxSteps { get; set; } = null;
    public bool CheckpointActivations { get; set; } = false;
}
