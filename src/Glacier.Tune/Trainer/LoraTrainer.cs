using System;
using System.Diagnostics;
using System.IO;
using Glacier.Tensor.Autograd;
using Glacier.Tensor.Core;
using Glacier.Tensor.Layers;
using Glacier.Tensor.Losses;
using Glacier.Tensor.Optimizers;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
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
    private bool _disposed;

    public GgufLoraModel Model => _model;
    public TrainingArguments Args => _args;

    public LoraTrainer(GgufLoraModel model, TrainingArguments args)
    {
        _model = model;
        _args = args;
        _optimizer = new AdamW(_model.TrainableParameters, lr: _args.LearningRate);
    }

    /// <summary>
    /// Executes fine-tuning across the provided dataset.
    /// </summary>
    public void Train(ChatMlDataset dataset, Action<TrainingStepResult>? onStep = null)
    {
        int datasetSteps = (dataset.Count + _args.GradientAccumulationSteps - 1) / _args.GradientAccumulationSteps;
        int totalSteps = _args.MaxSteps ?? (_args.Epochs * datasetSteps);
        int globalStep = 0;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("================================================================================");
        Console.WriteLine("                GLACIER.TUNE: PURE C# .NET 10 LLM FINE-TUNING                   ");
        Console.WriteLine("================================================================================");
        Console.ResetColor();

        var (trainable, total, pct) = _model.GetParameterStats();
        Console.WriteLine($"  Compute Device/Target: {_model.Target} (Hardware Accelerated)");
        Console.WriteLine($"  Base Model Layers:    {_model.LayerCount} ({_model.HiddenDim} dim, {_model.FfnDim} ffn)");
        Console.WriteLine($"  Base Model Params:    {total:N0} parameters (FROZEN in unmanaged memory)");
        Console.WriteLine($"  Trainable Parameters: {trainable:N0} ({pct:F2}% trainable - {100.0 - pct:F2}% parameter reduction)");
        Console.WriteLine($"  Dataset Size:         {dataset.Count} examples across {_args.Epochs} epochs");
        Console.WriteLine($"  Gradient Accum Steps: {_args.GradientAccumulationSteps}");
        Console.WriteLine($"  Base Learning Rate:   {_args.LearningRate:E2} ({_args.LrSchedulerType} decay)\n");

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
                Console.Write($"\r  [Epoch {epoch,2}/{_args.Epochs}] Step {globalStep + 1,4}/{totalSteps} | Microbatch {accumCounter}/{_args.GradientAccumulationSteps} (tokens: {example.InputIds.Length})...    ");

                // Compute loss over example using LoRA projection adapters
                float lossVal = TrainMicroBatch(example);
                accumulatedLoss += lossVal;
                accumulatedTokens += example.InputIds.Length;

                if (accumCounter >= _args.GradientAccumulationSteps || i == dataset.Count - 1)
                {
                    globalStep++;

                    // Update learning rate with cosine decay schedule
                    float currentLr = ComputeCosineLr(globalStep, totalSteps, _args.LearningRate, _args.WarmupRatio);
                    _optimizer.LearningRate = currentLr;

                    // Execute optimizer step across all LoRA parameters
                    _optimizer.Step();
                    _optimizer.ZeroGrad();

                    swGlobalStep.Stop();
                    double stepDurationMs = swGlobalStep.Elapsed.TotalMilliseconds;
                    double tokensPerSec = accumulatedTokens / (stepDurationMs / 1000.0);
                    float avgLoss = accumulatedLoss / accumCounter;

                    var result = new TrainingStepResult(globalStep, avgLoss, stepDurationMs, tokensPerSec);
                    onStep?.Invoke(result);

                    Console.WriteLine($"\r  [Epoch {epoch,2}/{_args.Epochs}] Step {globalStep,4}/{totalSteps} | " +
                                      $"Loss: {avgLoss:F4} | LR: {currentLr:E2} | " +
                                      $"Step Latency: {stepDurationMs:F1} ms | Speed: {tokensPerSec:N0} tokens/sec    ");

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

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n[SUCCESS] Fine-tuning completed in {swTotal.Elapsed.TotalMinutes:F2} minutes ({swTotal.Elapsed.TotalSeconds:F1} seconds)!");
        Console.ResetColor();

        // Save adapters to output directory
        Console.WriteLine($"[*] Exporting LoRA adapter weights to {_args.OutputDir}...");
        _model.SaveAdapter(_args.OutputDir);
        Console.WriteLine($"[SUCCESS] Model adapter saved successfully.");
    }

    private float TrainMicroBatch(TrainingExample example)
    {
        return _model.ForwardLossAndBackward(example.InputIds, example.Labels, _args.CheckpointActivations);
    }

    private static float ComputeCosineLr(int currentStep, int totalSteps, float baseLr, float warmupRatio)
    {
        int warmupSteps = (int)(totalSteps * warmupRatio);
        if (currentStep < warmupSteps)
        {
            return baseLr * ((float)currentStep / Math.Max(1, warmupSteps));
        }

        float progress = (float)(currentStep - warmupSteps) / Math.Max(1, totalSteps - warmupSteps);
        return baseLr * 0.5f * (1.0f + MathF.Cos(MathF.PI * progress));
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
