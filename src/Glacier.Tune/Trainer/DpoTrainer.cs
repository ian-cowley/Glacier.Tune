using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Glacier.Tensor.Optimizers;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Model;

namespace Glacier.Tune.Trainer;

public sealed record DpoPreferencePair(
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("chosen")] string Chosen,
    [property: JsonPropertyName("rejected")] string Rejected);

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(DpoPreferencePair))]
[JsonSerializable(typeof(List<DpoPreferencePair>))]
public partial class DpoJsonContext : JsonSerializerContext
{
}

public sealed record TokenizedPreferencePair(
    int[] ChosenInputIds,
    int[] ChosenLabels,
    int[] RejectedInputIds,
    int[] RejectedLabels);

public sealed record DpoLossResult(
    double Loss,
    double ChosenReward,
    double RejectedReward,
    double Margin,
    double GradPolicyChosen,
    double GradPolicyRejected);

public sealed record DpoStepResult(
    int Step,
    double Loss,
    double ChosenReward,
    double RejectedReward,
    double RewardMargin,
    double RewardAccuracy);

public sealed class DpoTrainingArguments
{
    public double Beta { get; set; } = 0.1;
    public double LabelSmoothing { get; set; } = 0.0;
    public bool LengthNormalizedLogProbs { get; set; } = true;
    public float LearningRate { get; set; } = 5e-5f;
    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.999f;
    public float Eps { get; set; } = 1e-8f;
    public float WeightDecay { get; set; } = 0.01f;
    public float MaxGradNorm { get; set; } = 1.0f;
    public float WarmupRatio { get; set; } = 0.05f;
    public int Epochs { get; set; } = 1;
    public string OutputDir { get; set; } = "dpo_adapter";
}

/// <summary>
/// Numerically stable Direct Preference Optimization (DPO / cDPO) scalar objective and exact gradients.
/// </summary>
public static class DpoLoss
{
    public static DpoLossResult Compute(
        double policyChosenLogProb,
        double policyRejectedLogProb,
        double refChosenLogProb,
        double refRejectedLogProb,
        double beta = 0.1,
        double labelSmoothing = 0.0)
    {
        if (beta <= 0.0 || !double.IsFinite(beta))
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be finite and positive.");
        if (labelSmoothing < 0.0 || labelSmoothing >= 0.5)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing must be in [0, 0.5).");

        double chosenReward = beta * (policyChosenLogProb - refChosenLogProb);
        double rejectedReward = beta * (policyRejectedLogProb - refRejectedLogProb);
        double margin = chosenReward - rejectedReward;

        double lossNeg = Softplus(-margin);
        double loss = labelSmoothing > 0.0
            ? (1.0 - labelSmoothing) * lossNeg + labelSmoothing * Softplus(margin)
            : lossNeg;

        double sigmoidNegMargin = Sigmoid(-margin);
        double dLossDMargin = labelSmoothing - sigmoidNegMargin;

        double gradPolicyChosen = beta * dLossDMargin;
        double gradPolicyRejected = -beta * dLossDMargin;

        return new DpoLossResult(loss, chosenReward, rejectedReward, margin, gradPolicyChosen, gradPolicyRejected);
    }

    private static double Softplus(double x)
    {
        if (x > 40.0) return x;
        if (x < -40.0) return Math.Exp(x);
        return x >= 0.0 ? x + Math.Log(1.0 + Math.Exp(-x)) : Math.Log(1.0 + Math.Exp(x));
    }

    private static double Sigmoid(double x)
    {
        if (x >= 40.0) return 1.0;
        if (x <= -40.0) return 0.0;
        if (x >= 0.0)
        {
            double e = Math.Exp(-x);
            return 1.0 / (1.0 + e);
        }
        else
        {
            double e = Math.Exp(x);
            return e / (1.0 + e);
        }
    }
}

public static class DpoDataset
{
    public static List<DpoPreferencePair> LoadJsonl(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"DPO preference file not found: {filePath}");

        var pairs = new List<DpoPreferencePair>();
        using var reader = new StreamReader(filePath);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                var pair = JsonSerializer.Deserialize(line, DpoJsonContext.Default.DpoPreferencePair);
                if (pair != null && !string.IsNullOrEmpty(pair.Chosen) && !string.IsNullOrEmpty(pair.Rejected))
                {
                    pairs.Add(pair);
                }
            }
            catch (JsonException)
            {
                // Skip malformed JSONL lines gracefully
            }
        }
        return pairs;
    }
}

/// <summary>
/// Pure C# Direct Preference Optimization (DPO) LoRA alignment trainer for <see cref="GgufLoraModel"/>.
/// Supports either an explicit frozen reference <see cref="GgufLoraModel"/> or automatic precomputed reference log-probs.
/// </summary>
public sealed class DpoTrainer : IDisposable
{
    private readonly GgufLoraModel _policyModel;
    private readonly GgufLoraModel? _referenceModel;
    private readonly DpoTrainingArguments _args;
    private readonly AdamW _optimizer;
    private bool _disposed;

    public GgufLoraModel PolicyModel => _policyModel;
    public DpoTrainingArguments Args => _args;

    public DpoTrainer(GgufLoraModel policyModel, DpoTrainingArguments args, GgufLoraModel? referenceModel = null)
    {
        _policyModel = policyModel ?? throw new ArgumentNullException(nameof(policyModel));
        _args = args ?? throw new ArgumentNullException(nameof(args));
        _referenceModel = referenceModel;
        _optimizer = new AdamW(
            _policyModel.TrainableParameters,
            lr: _args.LearningRate,
            beta1: _args.Beta1,
            beta2: _args.Beta2,
            eps: _args.Eps,
            weightDecay: _args.WeightDecay);
    }

    public List<DpoStepResult> Train(IReadOnlyList<TokenizedPreferencePair> dataset, Action<DpoStepResult>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dataset);
        if (dataset.Count == 0) return [];

        // Cache initial reference log-probabilities once when no dedicated reference model instance is supplied.
        var cachedRefLogProbs = new (double RefChosenLogP, double RefRejectedLogP)[dataset.Count];
        var refEvalModel = _referenceModel ?? _policyModel;
        for (int i = 0; i < dataset.Count; i++)
        {
            var pair = dataset[i];
            double refChosen = ComputeSequenceLogProb(refEvalModel, pair.ChosenInputIds, pair.ChosenLabels, _args.LengthNormalizedLogProbs, out _);
            double refRejected = ComputeSequenceLogProb(refEvalModel, pair.RejectedInputIds, pair.RejectedLabels, _args.LengthNormalizedLogProbs, out _);
            cachedRefLogProbs[i] = (refChosen, refRejected);
        }

        int totalSteps = Math.Max(1, _args.Epochs * dataset.Count);
        var trainable = _policyModel.TrainableParameters;
        var chosenGradBuffers = new float[trainable.Count][];

        var results = new List<DpoStepResult>(totalSteps);
        int globalStep = 0;

        for (int epoch = 0; epoch < _args.Epochs; epoch++)
        {
            for (int idx = 0; idx < dataset.Count; idx++)
            {
                var pair = dataset[idx];
                var (refChosen, refRejected) = cachedRefLogProbs[idx];

                // 1. Evaluate policy sequence log-probs cleanly
                double policyChosenLogP = ComputeSequenceLogProb(_policyModel, pair.ChosenInputIds, pair.ChosenLabels, _args.LengthNormalizedLogProbs, out int chosenTokens);
                double policyRejectedLogP = ComputeSequenceLogProb(_policyModel, pair.RejectedInputIds, pair.RejectedLabels, _args.LengthNormalizedLogProbs, out int rejectedTokens);

                var dpo = DpoLoss.Compute(
                    policyChosenLogP,
                    policyRejectedLogP,
                    refChosen,
                    refRejected,
                    _args.Beta,
                    _args.LabelSmoothing);

                if (chosenTokens > 0 && rejectedTokens > 0 && double.IsFinite(dpo.Loss))
                {
                    // 2. Backprop chosen sequence (ForwardLossAndBackward accumulates d(-avgLogP)/dTheta)
                    _optimizer.ZeroGrad();
                    _policyModel.ForwardLossAndBackward(pair.ChosenInputIds, pair.ChosenLabels);
                    for (int p = 0; p < trainable.Count; p++)
                    {
                        var gradTensor = trainable[p].Grad;
                        if (gradTensor != null)
                        {
                            var gradSpan = gradTensor.AsSpan();
                            if (chosenGradBuffers[p] == null || chosenGradBuffers[p].Length != gradSpan.Length)
                                chosenGradBuffers[p] = new float[gradSpan.Length];
                            gradSpan.CopyTo(chosenGradBuffers[p]);
                        }
                        else if (chosenGradBuffers[p] != null)
                        {
                            Array.Clear(chosenGradBuffers[p]);
                        }
                    }

                    // 3. Backprop rejected sequence
                    _optimizer.ZeroGrad();
                    _policyModel.ForwardLossAndBackward(pair.RejectedInputIds, pair.RejectedLabels);

                    // Combine: d(loss)/dTheta = (-gradLogP_w * scaleW) * gCE_w + (-gradLogP_l * scaleL) * gCE_l
                    double scaleW = _args.LengthNormalizedLogProbs ? 1.0 : chosenTokens;
                    double scaleL = _args.LengthNormalizedLogProbs ? 1.0 : rejectedTokens;
                    float coefW = (float)(-dpo.GradPolicyChosen * scaleW);
                    float coefL = (float)(-dpo.GradPolicyRejected * scaleL);

                    for (int p = 0; p < trainable.Count; p++)
                    {
                        var gradTensor = trainable[p].Grad;
                        if (gradTensor == null) continue;
                        Span<float> gSpan = gradTensor.AsSpan();
                        float[]? gChosen = chosenGradBuffers[p];
                        if (gChosen != null && gChosen.Length == gSpan.Length)
                        {
                            for (int j = 0; j < gSpan.Length; j++)
                            {
                                gSpan[j] = coefW * gChosen[j] + coefL * gSpan[j];
                            }
                        }
                    }

                    globalStep++;
                    _optimizer.LearningRate = LoraTrainer.ComputeCosineLr(globalStep, totalSteps, _args.LearningRate, _args.WarmupRatio, minLr: _args.LearningRate * 0.1f);
                    ClipGradients(_args.MaxGradNorm);
                    _optimizer.Step();
                    _optimizer.ZeroGrad();
                }
                else
                {
                    globalStep++;
                }

                var stepResult = new DpoStepResult(
                    globalStep,
                    dpo.Loss,
                    dpo.ChosenReward,
                    dpo.RejectedReward,
                    dpo.Margin,
                    dpo.Margin > 0.0 ? 1.0 : 0.0);
                results.Add(stepResult);
                onStep?.Invoke(stepResult);
            }
        }

        GlacierDiagnostics.LogInformation($"[DPO] Completed {globalStep} DPO optimization steps.");
        return results;
    }

    private static double ComputeSequenceLogProb(
        GgufLoraModel model,
        int[] inputIds,
        int[] labels,
        bool lengthNormalized,
        out int validTokens)
    {
        float ceLoss = model.EvaluateLoss(inputIds, labels, out validTokens);
        if (validTokens <= 0 || !float.IsFinite(ceLoss)) return 0.0;
        return lengthNormalized ? -ceLoss : -(double)ceLoss * validTokens;
    }

    private void ClipGradients(float maxNorm)
    {
        if (maxNorm <= 0f) return;
        double normSq = 0.0;
        foreach (var p in _policyModel.TrainableParameters)
        {
            if (p.Grad == null) continue;
            foreach (float g in p.Grad.AsSpan())
                normSq += (double)g * g;
        }
        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm && norm > 0.0)
        {
            float scale = (float)(maxNorm / norm);
            foreach (var p in _policyModel.TrainableParameters)
            {
                if (p.Grad == null) continue;
                Span<float> span = p.Grad.AsSpan();
                for (int i = 0; i < span.Length; i++) span[i] *= scale;
            }
        }
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
