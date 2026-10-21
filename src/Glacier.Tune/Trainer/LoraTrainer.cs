using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.Intrinsics;
using Glacier.Gpu.Drivers;
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
    private readonly AdamW _optimizer;
    private int _startingStep;
    private float? _resumedLearningRate;
    private bool _disposed;

    public GgufLoraModel Model => _model;
    public TrainingArguments Args => _args;
    public AdamW Optimizer => _optimizer;

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
        _optimizer = new AdamW(
            _model.TrainableParameters,
            lr: _args.LearningRate,
            beta1: _args.Beta1,
            beta2: _args.Beta2,
            eps: _args.Eps,
            weightDecay: _args.WeightDecay);
    }

    /// <summary>
    /// Resumes training from an existing saved LoRA adapter checkpoint, optionally
    /// annealing the base learning rate and setting the initial step offset.
    /// </summary>
    public void ResumeFromCheckpoint(string adapterFilePath, float? annealedLearningRate = null, int startingStep = 0)
    {
        GlacierDiagnostics.LogInformation($"[RESUME] Loading adapter checkpoint from {adapterFilePath}...");
        _model.LoadAdapter(adapterFilePath);
        _startingStep = Math.Max(0, startingStep);
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
        Action<TrainingStepResult>? onStep = null)
    {
        ResumeFromCheckpoint(adapterFilePath, annealedLearningRate, startingStep);
        int? totalSteps = additionalSteps.HasValue ? startingStep + additionalSteps.Value : null;
        Train(dataset, onStep, startingStepOverride: startingStep, learningRateOverride: annealedLearningRate, totalStepsOverride: totalSteps);
    }

    /// <summary>
    /// Executes fine-tuning across the provided dataset, optionally starting from a custom step
    /// with an annealed learning rate and custom total step budget.
    /// </summary>
    public void Train(
        ChatMlDataset dataset,
        Action<TrainingStepResult>? onStep = null,
        int? startingStepOverride = null,
        float? learningRateOverride = null,
        int? totalStepsOverride = null)
    {
        int initialStep = startingStepOverride ?? _startingStep;
        float baseLr = learningRateOverride ?? _resumedLearningRate ?? _args.LearningRate;

        int datasetSteps = (dataset.Count + _args.GradientAccumulationSteps - 1) / _args.GradientAccumulationSteps;
        int totalSteps = totalStepsOverride ?? _args.MaxSteps ?? (initialStep + (_args.Epochs * datasetSteps));
        int globalStep = initialStep;

        GlacierDiagnostics.LogInformation("================================================================================");
        GlacierDiagnostics.LogInformation("                GLACIER.TUNE: PURE C# .NET 10 LLM FINE-TUNING                   ");
        GlacierDiagnostics.LogInformation("================================================================================");

        var (trainable, total, pct) = _model.GetParameterStats();
        GlacierDiagnostics.LogInformation($"  Compute Device/Target: {_model.Target} (Hardware Accelerated)");
        GlacierDiagnostics.LogInformation($"  Base Model Layers:    {_model.LayerCount} ({_model.HiddenDim} dim, {_model.FfnDim} ffn)");
        GlacierDiagnostics.LogInformation($"  Base Model Params:    {total:N0} parameters (FROZEN in unmanaged memory)");
        GlacierDiagnostics.LogInformation($"  Trainable Parameters: {trainable:N0} ({pct:F2}% trainable - {100.0 - pct:F2}% parameter reduction)");
        GlacierDiagnostics.LogInformation($"  Dataset Size:         {dataset.Count} examples across {_args.Epochs} epochs");
        GlacierDiagnostics.LogInformation($"  Gradient Accum Steps: {_args.GradientAccumulationSteps}");
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
            int accumCounter = 0;
            _optimizer.ZeroGrad();

            for (int i = 0; i < dataset.Count; i++)
            {
                var example = dataset.Examples[i];
                if (example.InputIds.Length < 2) continue;

                accumCounter++;
                GlacierDiagnostics.LogDebug($"[Epoch {epoch,2}/{_args.Epochs}] Step {globalStep + 1,4}/{totalSteps} | Microbatch {accumCounter}/{_args.GradientAccumulationSteps} (tokens: {example.InputIds.Length})...");

                // Compute loss over example using LoRA projection adapters
                float lossVal = TrainMicroBatch(example);
                accumulatedLoss += lossVal;
                accumulatedTokens += example.InputIds.Length;

                if (accumCounter >= _args.GradientAccumulationSteps || i == dataset.Count - 1)
                {
                    globalStep++;

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
                    double tokensPerSec = accumulatedTokens / (stepDurationMs / 1000.0);
                    float avgLoss = accumulatedLoss / accumCounter;

                    var result = new TrainingStepResult(globalStep, avgLoss, stepDurationMs, tokensPerSec);
                    onStep?.Invoke(result);

                    string vramInfo = "";
                    if (CuDriver.IsAvailable() && CuDriver.MemGetInfo(out nuint freeBytes, out nuint totalBytes) == 0)
                    {
                        double usedMb = (totalBytes - freeBytes) / (1024.0 * 1024.0);
                        vramInfo = $" | VRAM: {usedMb:F0} MB";
                    }

                    GlacierDiagnostics.LogInformation($"[Epoch {epoch,2}/{_args.Epochs}] Step {globalStep,4}/{totalSteps} | " +
                                      $"Loss: {avgLoss:F4} | LR: {currentLr:E2} | " +
                                      $"GradNorm: {gradNorm:F4}{vramInfo} | " +
                                      $"Step Latency: {stepDurationMs:F1} ms | Speed: {tokensPerSec:N0} tokens/sec");

                    accumCounter = 0;
                    accumulatedLoss = 0f;
                    accumulatedTokens = 0;
                    swGlobalStep.Restart();

                    if (_args.MaxSteps.HasValue && globalStep >= _args.MaxSteps.Value)
                    {
                        break;
                    }
                }
            }

            if (_args.MaxSteps.HasValue && globalStep >= _args.MaxSteps.Value)
            {
                break;
            }
        }
        swTotal.Stop();

        GlacierDiagnostics.LogInformation($"\n[SUCCESS] Fine-tuning completed in {swTotal.Elapsed.TotalMinutes:F2} minutes ({swTotal.Elapsed.TotalSeconds:F1} seconds)!");

        // Save adapters to output directory
        GlacierDiagnostics.LogInformation($"[*] Exporting LoRA adapter weights to {_args.OutputDir}...");
        _model.SaveAdapter(_args.OutputDir);
        GlacierDiagnostics.LogInformation("[SUCCESS] Model adapter saved successfully.");
    }

    private float TrainMicroBatch(TrainingExample example)
    {
        return _model.ForwardLossAndBackward(example.InputIds, example.Labels, _args.CheckpointActivations);
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
        if (maxNorm <= 0f) return 0f;

        double totalNormSq = 0.0;
        foreach (var p in _model.TrainableParameters)
        {
            if (p.Grad == null) continue;
            var gradSpan = p.Grad.AsSpan();
            int n = gradSpan.Length;

            double sumSq = 0.0;
            int i = 0;
            if (Vector256.IsHardwareAccelerated && n >= 8)
            {
                var vSum = Vector256<float>.Zero;
                for (; i <= n - 8; i += 8)
                {
                    var vG = Vector256.LoadUnsafe(ref gradSpan[i]);
                    vSum = Vector256.Add(vSum, Vector256.Multiply(vG, vG));
                }
                sumSq += Vector256.Sum(vSum);
            }
            for (; i < n; i++)
            {
                float g = gradSpan[i];
                sumSq += g * g;
            }
            totalNormSq += sumSq;
        }

        float totalNorm = MathF.Sqrt((float)totalNormSq);
        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / (totalNorm + 1e-6f);
            foreach (var p in _model.TrainableParameters)
            {
                if (p.Grad == null) continue;
                var gradSpan = p.Grad.AsSpan();
                int n = gradSpan.Length;
                int i = 0;
                if (Vector256.IsHardwareAccelerated && n >= 8)
                {
                    var vScale = Vector256.Create(scale);
                    for (; i <= n - 8; i += 8)
                    {
                        var vG = Vector256.LoadUnsafe(ref gradSpan[i]);
                        Vector256.StoreUnsafe(Vector256.Multiply(vG, vScale), ref gradSpan[i]);
                    }
                }
                for (; i < n; i++)
                {
                    gradSpan[i] *= scale;
                }
            }
        }

        return totalNorm;
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
