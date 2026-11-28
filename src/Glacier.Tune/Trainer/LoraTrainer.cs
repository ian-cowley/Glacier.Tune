using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.Intrinsics;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Model;

namespace Glacier.Tune.Trainer;

public sealed record TrainingStepResult(int Step, float Loss, double StepDurationMs, double TokensPerSec);

/// <summary>
/// High-performance LLM fine-tuning engine.
/// Orchestrates forward pass through LoRA adapters, Autograd reverse backpropagation,
/// and in-place AdamW updates with cosine learning rate scheduling.
/// </summary>
public sealed class LoraTrainer : IDisposable
{
    private readonly GgufLoraModel _model;
    private readonly TrainingArguments _args;
    private readonly StatefulAdamW _optimizer;
    private int _startingStep;
    private float? _resumedLearningRate;
    private bool _disposed;

    public GgufLoraModel Model => _model;
    public TrainingArguments Args => _args;
    public StatefulAdamW Optimizer => _optimizer;
    public float? BestValidationLoss { get; private set; }
    public float? LastValidationLoss { get; private set; }

    public int StartingStep
    {
        get => _startingStep;
        set => _startingStep = value;
    }

    public float? ResumedLearningRate
    {
        get => _resumedLearningRate;
        set => _resumedLearningRate = value;
    }

    public LoraTrainer(GgufLoraModel model, TrainingArguments args)
    {
        _model = model;
        _args = args;
        _optimizer = new StatefulAdamW(
            _model.TrainableParameters,
            lr: _args.LearningRate,
            beta1: _args.Beta1,
            beta2: _args.Beta2,
            eps: _args.Eps,
            weightDecay: _args.WeightDecay);
    }

    internal static string GetOptimizerStatePath(string adapterPath)
    {
        if (adapterPath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
        {
            return Path.ChangeExtension(adapterPath, ".opt");
        }
        return Path.Combine(adapterPath, "optimizer_state.bin");
    }

    /// <summary>
    /// Resumes training from an existing saved LoRA adapter checkpoint, optionally
    /// restoring optimizer moments, annealing the learning rate, and setting the initial step offset.
    /// </summary>
    public void ResumeFromCheckpoint(string adapterFilePath, float? annealedLearningRate = null, int startingStep = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        GlacierDiagnostics.LogInformation($"[RESUME] Loading adapter checkpoint from {adapterFilePath}...");
        _model.LoadAdapter(adapterFilePath);

        string optPath = GetOptimizerStatePath(adapterFilePath);
        if (!File.Exists(optPath) && adapterFilePath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
        {
            string? dir = Path.GetDirectoryName(adapterFilePath);
            if (!string.IsNullOrEmpty(dir))
            {
                string dirOpt = Path.Combine(dir, "optimizer_state.bin");
                if (File.Exists(dirOpt)) optPath = dirOpt;
            }
        }

        if (File.Exists(optPath))
        {
            _optimizer.LoadState(optPath);
            GlacierDiagnostics.LogInformation($"[RESUME] Restored AdamW optimizer state from {optPath} (step={_optimizer.StepCount}).");
        }

        _startingStep = Math.Max(0, startingStep > 0 ? startingStep : _optimizer.StepCount);
        if (annealedLearningRate.HasValue)
        {
            _resumedLearningRate = annealedLearningRate.Value;
            _optimizer.LearningRate = annealedLearningRate.Value;
            GlacierDiagnostics.LogInformation($"[RESUME] Annealed learning rate set to {_resumedLearningRate.Value:E2}");
        }
        GlacierDiagnostics.LogInformation($"[RESUME] Training will resume from global step {_startingStep}.");
    }

    /// <summary>
    /// Programmatically resumes QLoRA training directly from an adapter checkpoint with custom starting step and optional annealed learning rate.
    /// </summary>
    public void ResumeTraining(
        string adapterFilePath,
        ChatMlDataset dataset,
        float? annealedLearningRate = null,
        int startingStep = 0,
        int? additionalSteps = null,
        Action<TrainingStepResult>? onStep = null,
        ChatMlDataset? evalDataset = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dataset);
        ResumeFromCheckpoint(adapterFilePath, annealedLearningRate, startingStep);
        int effectiveStartingStep = Math.Max(0, startingStep);
        int? totalSteps = additionalSteps.HasValue ? effectiveStartingStep + Math.Max(0, additionalSteps.Value) : null;
        TrainCore(dataset, onStep, startingStepOverride: effectiveStartingStep, learningRateOverride: annealedLearningRate, totalStepsOverride: totalSteps, evalDataset: evalDataset, preservePriorBestCheckpoint: true);
    }

    /// <summary>
    /// Executes fine-tuning across the provided dataset, optionally starting from a custom step
    /// with an annealed learning rate, custom total step budget, and optional validation split.
    /// </summary>
    public void Train(
        ChatMlDataset dataset,
        Action<TrainingStepResult>? onStep = null,
        int? startingStepOverride = null,
        float? learningRateOverride = null,
        int? totalStepsOverride = null,
        ChatMlDataset? evalDataset = null)
    {
        TrainCore(dataset, onStep, startingStepOverride, learningRateOverride, totalStepsOverride, evalDataset, preservePriorBestCheckpoint: false);
    }

    private void TrainCore(
        ChatMlDataset dataset,
        Action<TrainingStepResult>? onStep,
        int? startingStepOverride,
        float? learningRateOverride,
        int? totalStepsOverride,
        ChatMlDataset? evalDataset,
        bool preservePriorBestCheckpoint)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dataset);

        int gradAccumSteps = Math.Max(1, _args.GradientAccumulationSteps);
        int initialStep = Math.Max(0, startingStepOverride ?? _startingStep);
        float baseLr = learningRateOverride ?? _resumedLearningRate ?? _args.LearningRate;

        int datasetSteps = (dataset.Count + gradAccumSteps - 1) / gradAccumSteps;
        int? maxStepsLimit = totalStepsOverride ?? _args.MaxSteps;
        int totalSteps = maxStepsLimit ?? (initialStep + (_args.Epochs * datasetSteps));
        int globalStep = initialStep;

        BestValidationLoss = null;
        LastValidationLoss = null;
        float bestValLoss = float.PositiveInfinity;
        string bestPath = Path.Combine(_args.OutputDir, "best_adapter.bin");
        string bestLossPath = Path.Combine(_args.OutputDir, "best_adapter.loss");
        if (preservePriorBestCheckpoint && File.Exists(bestPath) && File.Exists(bestLossPath))
        {
            string rawLoss = File.ReadAllText(bestLossPath).Trim();
            if (float.TryParse(rawLoss, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float priorBest) &&
                float.IsFinite(priorBest))
            {
                bestValLoss = priorBest;
                BestValidationLoss = priorBest;
            }
        }

        GlacierDiagnostics.LogInformation("================================================================================");
        GlacierDiagnostics.LogInformation("                GLACIER.TUNE: PURE C# .NET 10 LLM FINE-TUNING                   ");
        GlacierDiagnostics.LogInformation("================================================================================");

        var (trainable, total, pct) = _model.GetParameterStats();
        string accelDesc = _model.IsGpuAccelerated ? "GPU VRAM Accelerated" : "CPU SIMD Accelerated";
        GlacierDiagnostics.LogInformation($"  Compute Device/Target: {_model.Target} ({accelDesc})");
        GlacierDiagnostics.LogInformation($"  Base Model Layers:    {_model.LayerCount} ({_model.HiddenDim} dim, {_model.FfnDim} ffn)");
        GlacierDiagnostics.LogInformation($"  Base Model Params:    {total:N0} parameters (FROZEN in unmanaged memory)");
        GlacierDiagnostics.LogInformation($"  Trainable Parameters: {trainable:N0} ({pct:F2}% trainable - {100.0 - pct:F2}% parameter reduction)");
        GlacierDiagnostics.LogInformation($"  Dataset Size:         {dataset.Count} examples across {_args.Epochs} epochs");
        if (evalDataset != null && _args.EvalSteps > 0)
        {
            GlacierDiagnostics.LogInformation($"  Validation Split:     {evalDataset.Count} examples (every {_args.EvalSteps} steps)");
        }
        GlacierDiagnostics.LogInformation($"  Gradient Accum Steps: {gradAccumSteps}");
        GlacierDiagnostics.LogInformation($"  Base Learning Rate:   {baseLr:E2} ({_args.LrSchedulerType} decay)");
        if (initialStep > 0)
        {
            GlacierDiagnostics.LogInformation($"  Resumed Starting Step: {initialStep}/{totalSteps}\n");
        }
        else
        {
            GlacierDiagnostics.LogInformation("");
        }

        var swTotal = Stopwatch.StartNew();
        var swGlobalStep = Stopwatch.StartNew();
        int accumulatedTokens = 0;
        float accumulatedLoss = 0f;

        for (int epoch = 1; epoch <= _args.Epochs; epoch++)
        {
            if (maxStepsLimit.HasValue && globalStep >= maxStepsLimit.Value)
            {
                break;
            }

            int accumCounter = 0;
            accumulatedLoss = 0f;
            accumulatedTokens = 0;
            _optimizer.ZeroGrad();

            for (int i = 0; i < dataset.Count; i++)
            {
                var example = dataset.Examples[i];
                if (example.InputIds.Length >= 2)
                {
                    accumCounter++;
                    GlacierDiagnostics.LogDebug($"[Epoch {epoch,2}/{_args.Epochs}] Step {globalStep + 1,4}/{totalSteps} | Microbatch {accumCounter}/{gradAccumSteps} (tokens: {example.InputIds.Length})...");

                    // Compute loss over example using LoRA projection adapters
                    float lossVal = TrainMicroBatch(example);
                    accumulatedLoss += lossVal;
                    accumulatedTokens += example.InputIds.Length;
                }

                if (accumCounter > 0 && (accumCounter >= gradAccumSteps || i == dataset.Count - 1))
                {
                    globalStep++;

                    // Average accumulated microbatch gradients before norm clipping and optimizer step
                    if (accumCounter > 1)
                    {
                        ScaleAccumulatedGradients(1.0f / accumCounter);
                    }

                    // Update learning rate with cosine decay schedule
                    float currentLr = ComputeCosineLr(globalStep, totalSteps, baseLr, _args.WarmupRatio);
                    _optimizer.LearningRate = currentLr;

                    // Clip gradients for QLoRA stability
                    float gradNorm = ClipGradients(_args.MaxGradNorm);

                    // Execute optimizer step across all LoRA parameters
                    _optimizer.Step();
                    _optimizer.ZeroGrad();

                    swGlobalStep.Stop();
                    double stepDurationMs = swGlobalStep.Elapsed.TotalMilliseconds;
                    double tokensPerSec = stepDurationMs > 0.0 ? accumulatedTokens / (stepDurationMs / 1000.0) : 0.0;
                    float avgLoss = accumulatedLoss / accumCounter;

                    var result = new TrainingStepResult(globalStep, avgLoss, stepDurationMs, tokensPerSec);
                    onStep?.Invoke(result);

                    string vramInfo = "";
                    if (CuDriver.IsAvailable() && CuDriver.MemGetInfo(out nuint freeBytes, out nuint totalBytes) == 0)
                    {
                        double usedMb = (totalBytes - freeBytes) / (1024.0 * 1024.0);
                        vramInfo = $" | VRAM: {usedMb:F0} MB";
                    }

                    string logLine = $"[Epoch {epoch,2}/{_args.Epochs}] Step {globalStep,4}/{totalSteps} | " +
                                     $"Loss: {avgLoss:F4} | LR: {currentLr:E2} | " +
                                     $"GradNorm: {gradNorm:F4}{vramInfo} | " +
                                     $"Step Latency: {stepDurationMs:F1} ms | Speed: {tokensPerSec:N0} tokens/sec";

                    if (_args.LoggingSteps <= 1 || globalStep % _args.LoggingSteps == 0 || globalStep == totalSteps)
                    {
                        GlacierDiagnostics.LogInformation(logLine);
                    }
                    else
                    {
                        GlacierDiagnostics.LogDebug(logLine);
                    }

                    if (_args.SaveSteps > 0 && globalStep % _args.SaveSteps == 0 && globalStep < totalSteps)
                    {
                        string checkpointDir = Path.Combine(_args.OutputDir, $"checkpoint-{globalStep}");
                        _model.SaveAdapter(checkpointDir);
                        _optimizer.SaveState(GetOptimizerStatePath(checkpointDir));
                        GlacierDiagnostics.LogInformation($"[CHECKPOINT] Saved intermediate LoRA checkpoint + optimizer state to {checkpointDir}");
                    }

                    if (evalDataset != null && evalDataset.Count > 0 && _args.EvalSteps > 0 && globalStep % _args.EvalSteps == 0)
                    {
                        float valLoss = Evaluate(evalDataset);
                        if (float.IsFinite(valLoss))
                        {
                            LastValidationLoss = valLoss;
                            if (valLoss < bestValLoss)
                            {
                                bestValLoss = valLoss;
                                BestValidationLoss = valLoss;
                                _model.SaveAdapter(bestPath);
                                _optimizer.SaveState(GetOptimizerStatePath(bestPath));
                                File.WriteAllText(bestLossPath, valLoss.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                                GlacierDiagnostics.LogInformation($"[EVAL] Step {globalStep}: ValLoss={valLoss:F4} (new best -> {bestPath})");
                            }
                            else
                            {
                                GlacierDiagnostics.LogInformation($"[EVAL] Step {globalStep}: ValLoss={valLoss:F4} (best={bestValLoss:F4})");
                            }
                        }
                    }

                    accumCounter = 0;
                    accumulatedLoss = 0f;
                    accumulatedTokens = 0;
                    swGlobalStep.Restart();

                    if (maxStepsLimit.HasValue && globalStep >= maxStepsLimit.Value)
                    {
                        break;
                    }
                }
            }

            if (maxStepsLimit.HasValue && globalStep >= maxStepsLimit.Value)
            {
                break;
            }
        }
        swTotal.Stop();

        GlacierDiagnostics.LogInformation($"\n[SUCCESS] Fine-tuning completed in {swTotal.Elapsed.TotalMinutes:F2} minutes ({swTotal.Elapsed.TotalSeconds:F1} seconds)!");

        // Save adapters & optimizer state to output directory
        GlacierDiagnostics.LogInformation($"[*] Exporting LoRA adapter weights to {_args.OutputDir}...");
        _model.SaveAdapter(_args.OutputDir);
        _optimizer.SaveState(GetOptimizerStatePath(_args.OutputDir));
        GlacierDiagnostics.LogInformation("[SUCCESS] Model adapter and optimizer state saved successfully.");
    }

    /// <summary>
    /// Computes forward-only mean validation loss across all valid examples in <paramref name="evalDataset"/>
    /// without mutating adapter gradients or optimizer state.
    /// </summary>
    public float Evaluate(ChatMlDataset evalDataset)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (evalDataset == null || evalDataset.Count == 0)
        {
            return float.NaN;
        }

        double totalLoss = 0.0;
        int evaluatedCount = 0;
        for (int i = 0; i < evalDataset.Count; i++)
        {
            var example = evalDataset.Examples[i];
            if (example.InputIds.Length < 2) continue;

            float loss = _model.EvaluateLoss(example.InputIds, example.Labels, out int validTokens);
            if (validTokens > 0 && float.IsFinite(loss))
            {
                totalLoss += loss;
                evaluatedCount++;
            }
        }

        return evaluatedCount > 0 ? (float)(totalLoss / evaluatedCount) : float.NaN;
    }

    private float TrainMicroBatch(TrainingExample example)
    {
        return _model.ForwardLossAndBackward(example.InputIds, example.Labels, _args.CheckpointActivations);
    }

    public void ScaleAccumulatedGradients(float scale)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        StatefulAdamW.ScaleGradients(_model.TrainableParameters, scale);
    }

    public static float ComputeCosineLr(int currentStep, int totalSteps, float baseLr, float warmupRatio, float minLr = 0f)
    {
        if (totalSteps <= 0) return baseLr;
        int warmupSteps = (int)(totalSteps * warmupRatio);
        if (currentStep < warmupSteps)
        {
            float progress = (float)currentStep / Math.Max(1, warmupSteps);
            return Math.Max(minLr, baseLr * progress);
        }

        float decayProgress = (float)(currentStep - warmupSteps) / Math.Max(1, totalSteps - warmupSteps);
        decayProgress = Math.Clamp(decayProgress, 0.0f, 1.0f);
        float cosineDecay = 0.5f * (1.0f + MathF.Cos(MathF.PI * decayProgress));
        return Math.Max(minLr, baseLr * cosineDecay);
    }

    public float ClipGradients(float maxNorm)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return StatefulAdamW.ClipGradients(_model.TrainableParameters, maxNorm);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _optimizer.Dispose();
        }
    }
}
