# Glacier.Tune User Manual

## Contents

1. [Installation & Requirements](#1-installation--requirements)
2. [Supervised LoRA Fine-Tuning (`LoraTrainer`)](#2-supervised-lora-fine-tuning-loratrainer)
3. [Direct Preference Optimization (`DpoTrainer`)](#3-direct-preference-optimization-dpotrainer)
4. [Sleep Consolidation & MoE Expansion (`Glacier.Tune.Sleep` & `Expansion`)](#4-sleep-consolidation--moe-expansion-glaciertunesleep--expansion)
5. [Standalone GGUF Merging & Evaluation](#5-standalone-gguf-merging--evaluation)
6. [Inkling2 Head Trainer & Diagnostics](#6-inkling2-head-trainer--diagnostics)
7. [Building & Testing](#7-building--testing)

---

## 1. Installation & Requirements

```bash
dotnet add package Glacier.Tune
```

- **Target framework:** `.NET 10` (`net10.0`), `<IsAotCompatible>true</IsAotCompatible>`.
- **Dependencies:** `Glacier.Inference` and `Glacier.Tensor`.
- **Supported base GGUF weights for CUDA/SIMD training:** `Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, and `F32`.

---

## 2. Supervised LoRA Fine-Tuning (`LoraTrainer`)

### Loading a Model (`GgufLoraModel` & `LoraConfig`)

```csharp
using Glacier.Tune.Config;
using Glacier.Tune.Model;

var config = new LoraConfig
{
    Rank = 16,
    Alpha = 32f,
    Dropout = 0.05f,
    TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    Bias = "none",
    Seed = 42
};

using var model = GgufLoraModel.Load("models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", config, device: "auto");
```

- `device` accepts `"auto"`, `"gpu"` / `"cuda"`, or `"cpu"` (resolved via `GgufLoraModel.ResolveTarget`).
- `GgufLoraModel` supports both standard dense transformer blocks (`TransformerBlock`) and hybrid Gated DeltaNet layers (`GdnBlock`, inspected via `model.IsHybrid`).

### Preparing Datasets (`ChatMlDataset` & `Inkling2Dataset`)

`ChatMlDataset.FromFile` reads JSONL conversations, masks system/user prompt tokens with `ChatMlDataset.IgnoreIndex` (`-100`), and trains only on assistant completions:

```csharp
using Glacier.Tune.Data;

var trainSet = ChatMlDataset.FromFile("train.jsonl", model.Tokenizer, maxSeqLength: 512);
var evalSet = ChatMlDataset.FromFile("eval.jsonl", model.Tokenizer, maxSeqLength: 512);
```

### Training & Checkpoints (`TrainingArguments` & `LoraTrainer`)

```csharp
using Glacier.Tune.Config;
using Glacier.Tune.Trainer;

var args = new TrainingArguments
{
    Device = "auto",
    LearningRate = 2e-4f,
    Epochs = 3,
    GradientAccumulationSteps = 8,
    MaxSeqLength = 512,
    WarmupRatio = 0.03f,
    SaveSteps = 15,
    EvalSteps = 15,
    CheckpointActivations = true,
    MaxGradNorm = 1.0f,
    WeightDecay = 0.01f,
    OutputDir = "./fine_tuned_lora"
};

using var trainer = new LoraTrainer(model, args);
// Optional resume: trainer.ResumeFromCheckpoint("./fine_tuned_lora/checkpoint-15/adapter_model.bin");
trainer.Train(trainSet, evalDataset: evalSet);
```

`StatefulAdamW` saves and restores optimizer moment buffers (`ADMW` binary format) alongside LoRA adapter weights (`LORA` `0x41524F4C` header in `LoraBinaryFormat`).

---

## 3. Direct Preference Optimization (`DpoTrainer`)

Train a policy adapter on preference triples (`prompt`, `chosen`, `rejected`) using standard DPO or conservative cDPO (`LabelSmoothing > 0`):

```csharp
using Glacier.Tune.Trainer;

var pairs = DpoDataset.LoadJsonl("preferences.jsonl");
var tokenized = DpoDataset.Tokenize(pairs, model.Tokenizer, maxSeqLength: 512);

var dpoArgs = new DpoTrainingArguments
{
    Beta = 0.1,
    LabelSmoothing = 0.05,
    LengthNormalizedLogProbs = true,
    SftAnchorWeight = 0.0,
    LearningRate = 5e-5f,
    Epochs = 1,
    ReferenceLogProbsCachePath = "./ref_logprobs.dprf",
    OutputDir = "./dpo_adapter"
};

using var dpoTrainer = new DpoTrainer(model, dpoArgs, referenceModel: null);
List<DpoStepResult> history = dpoTrainer.Train(tokenized, onStep: step =>
{
    Console.WriteLine($"Step {step.Step}: Loss={step.Loss:F4}, Margin={step.RewardMargin:F4}");
});
```

- When `referenceModel` is `null`, `DpoTrainer` evaluates reference log-probabilities using the unadapted base weights (or loads/writes `ReferenceLogProbsCachePath` using `'DPRF'` cache format).
- `DpoLoss.Compute` computes scalar DPO/cDPO loss, reward margin, and exact log-probability gradients independently of the trainer loop.

---

## 4. Sleep Consolidation & MoE Expansion (`Glacier.Tune.Sleep` & `Expansion`)

`SleepConsolidationTrainer` consolidates short-term trajectories into newly unlocked low-rank MoE experts using Generalized Knowledge Distillation (GKD):

```csharp
using Glacier.Tune.Expansion;
using Glacier.Tune.Sleep;

using var bank = new PreallocatedMoeExpansionBank(dimension: model.HiddenDim, rank: 16, maxExperts: 8);
using var sleep = new SleepConsolidationTrainer(
    model,
    bank,
    klBeta: 0.5f,
    klWeight: 0.25f,
    klTemperature: 1.0f);

int slot = sleep.ParameterExpansion(centroidSpan);
ConsolidationResult result = await sleep.ConsolidateKnowledgeAsync(candidates, epochs: 1);
sleep.SynapticPruning();
bank.Save("expansion_bank.bin");
```

Filter synthetic rollouts prior to distillation using `DreamQualityFilter.ComputeScore(dreamText, refText, dreamEmb, refEmb, alpha: 0.65f)`, which combines SIMD `CosineSimilarity` with normalized `FastLevenshtein` edit distance.

---

## 5. Standalone GGUF Merging & Evaluation

### Merging (`GgufMerger`)

Fold trained `.bin` adapter deltas directly into quantized GGUF weights:

```csharp
using Glacier.Tune.Export;

MergeResult merge = GgufMerger.Merge(
    baseGgufPath: "models/base.gguf",
    adapterBinPath: "./fine_tuned_lora/adapter_model.bin",
    outputGgufPath: "./tuned-q8_0.gguf",
    options: new GgufMerger.MergeOptions
    {
        TargetQuantization = TargetQuantization.Q8_0,
        ProgressCallback = Console.WriteLine
    });
```

`TargetQuantization` supports `Auto` (preserves compatible base block quantization), `Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, `F16`, `BF16`, and `F32`.

### Post-Tune Evaluation (`PostTuneEvaluator`)

Compare base and adapted greedy generation on held-out JSONL examples:

```csharp
using Glacier.Tune.Evaluation;

EvalReport report = PostTuneEvaluator.Evaluate(
    "models/base.gguf",
    "./fine_tuned_lora/adapter_model.bin",
    "eval.jsonl",
    new EvalOptions { MaxNewTokens = 64, MaxSeqLength = 1024 },
    device: "auto");

Console.WriteLine(report.FormatSummary());
```

### Sample Demo CLI Flags (`samples/Glacier.Tune.Demo`)

- Default training run: `dotnet run -c Release --project samples/Glacier.Tune.Demo -- --device auto --steps 3 [base.gguf] [train.jsonl]`
- Merge adapter: `dotnet run -c Release --project samples/Glacier.Tune.Demo -- --merge <base.gguf> [adapter_model.bin] [merged_out.gguf]`
- Evaluate adapter: `dotnet run -c Release --project samples/Glacier.Tune.Demo -- --eval <base.gguf> <adapter.bin> <eval.jsonl> [--device auto] [--max-tokens 64] [--out report.json]`

---

## 6. Inkling2 Head Trainer & Diagnostics

- **`Inkling2HeadLoraTrainer` + `Inkling2Dataset`:** Extracts/caches final hidden states (`IK2H` binary cache) and trains output-head adapters.
- **`GlacierDiagnostics`:** Attaches `ConsoleGlacierLogger`, `DelegateGlacierLogger`, or `NullGlacierLogger` (`LogLevel.Trace`..`Critical`).

---

## 7. Building & Testing

```bash
dotnet build Glacier.Tune.slnx -c Release
dotnet test tests/Glacier.Tune.Tests/Glacier.Tune.Tests.csproj -c Release
```
