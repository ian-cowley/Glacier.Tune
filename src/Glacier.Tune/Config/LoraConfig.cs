namespace Glacier.Tune.Config;

/// <summary>
/// Configuration for Parameter-Efficient Fine-Tuning (PEFT) Low-Rank Adaptation (LoRA).
/// </summary>
public sealed class LoraConfig
{
    public int Rank { get; set; } = 16;
    public float Alpha { get; set; } = 32f;
    public float Dropout { get; set; } = 0.05f;
    public string[] TargetModules { get; set; } = 
    [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ];
    public string Bias { get; set; } = "none";
    public int? Seed { get; set; } = 42;

    public float Scaling => Rank > 0 ? Alpha / Rank : 1.0f;
}
