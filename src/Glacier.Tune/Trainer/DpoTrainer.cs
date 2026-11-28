using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Glacier.Inference.Tokenizer;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Model;

namespace Glacier.Tune.Trainer;

public sealed record DpoPreferencePair(
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("chosen")] string Chosen,
    [property: JsonPropertyName("rejected")] string Rejected,
    [property: JsonPropertyName("prompt_messages")] List<ChatMessage>? PromptMessages = null);

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(DpoPreferencePair))]
[JsonSerializable(typeof(List<DpoPreferencePair>))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(List<ChatMessage>))]
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
    public int GradientAccumulationSteps { get; set; } = 1;
    public double SftAnchorWeight { get; set; } = 0.0;
    public bool CheckpointActivations { get; set; } = false;
    public int SaveSteps { get; set; } = 0;
    public int? MaxSteps { get; set; } = null;
    public int LoggingSteps { get; set; } = 1;
    public string? ReferenceLogProbsCachePath { get; set; } = null;
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

    public static List<TokenizedPreferencePair> Tokenize(
        IReadOnlyList<DpoPreferencePair> pairs,
        BpeTokenizer tokenizer,
        int maxSeqLength = 2048)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSeqLength, 2);

        var tokenized = new List<TokenizedPreferencePair>(pairs.Count);
        for (int i = 0; i < pairs.Count; i++)
        {
            var pair = pairs[i];
            if (string.IsNullOrEmpty(pair.Chosen) || string.IsNullOrEmpty(pair.Rejected))
                continue;

            var chosenConv = BuildConversation(pair, pair.Chosen);
            var rejectedConv = BuildConversation(pair, pair.Rejected);

            var chosenEx = ChatMlDataset.FormatAndTokenize(chosenConv, tokenizer, maxSeqLength);
            var rejectedEx = ChatMlDataset.FormatAndTokenize(rejectedConv, tokenizer, maxSeqLength);

            if (chosenEx.InputIds.Length >= 2 && rejectedEx.InputIds.Length >= 2)
            {
                tokenized.Add(new TokenizedPreferencePair(
                    chosenEx.InputIds,
                    chosenEx.Labels,
                    rejectedEx.InputIds,
                    rejectedEx.Labels));
            }
        }

        return tokenized;
    }

    private static ChatConversation BuildConversation(DpoPreferencePair pair, string responseContent)
    {
        var conv = new ChatConversation();
        if (pair.PromptMessages is { Count: > 0 })
        {
            foreach (var msg in pair.PromptMessages)
            {
                conv.Messages.Add(new ChatMessage { Role = msg.Role, Content = msg.Content });
            }
        }
        else if (!string.IsNullOrEmpty(pair.Prompt))
        {
            conv.Messages.Add(new ChatMessage { Role = "user", Content = pair.Prompt });
        }

        conv.Messages.Add(new ChatMessage { Role = "assistant", Content = responseContent });
        return conv;
    }
}

/// <summary>
/// Pure C# Direct Preference Optimization (DPO / cDPO / RPO-anchor) LoRA alignment trainer for <see cref="GgufLoraModel"/>.
/// Uses memoryless 2-pass scalar chain-rule scaling, disk reference log-prob caching (<c>DPRF</c>),
/// and <see cref="StatefulAdamW"/> persistence.
/// </summary>
public sealed class DpoTrainer : IDisposable
{
    public const uint RefCacheMagic = 0x46525044u; // 'DPRF'
    public const int RefCacheVersion = 1;

    private readonly GgufLoraModel _policyModel;
    private readonly GgufLoraModel? _referenceModel;
    private readonly DpoTrainingArguments _args;
    private readonly StatefulAdamW _optimizer;
    private int _startingStep;
    private bool _disposed;

    public GgufLoraModel PolicyModel => _policyModel;
    public DpoTrainingArguments Args => _args;
    public StatefulAdamW Optimizer => _optimizer;

    public DpoTrainer(GgufLoraModel policyModel, DpoTrainingArguments args, GgufLoraModel? referenceModel = null)
    {
        _policyModel = policyModel ?? throw new ArgumentNullException(nameof(policyModel));
        _args = args ?? throw new ArgumentNullException(nameof(args));
        _referenceModel = referenceModel;
        _optimizer = new StatefulAdamW(
            _policyModel.TrainableParameters,
            lr: _args.LearningRate,
            beta1: _args.Beta1,
            beta2: _args.Beta2,
            eps: _args.Eps,
            weightDecay: _args.WeightDecay);
    }

    public void ResumeFromCheckpoint(string adapterFilePath, float? annealedLearningRate = null, int startingStep = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _policyModel.LoadAdapter(adapterFilePath);

        string optPath = LoraTrainer.GetOptimizerStatePath(adapterFilePath);
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
        }

        _startingStep = Math.Max(0, startingStep > 0 ? startingStep : _optimizer.StepCount);
        if (annealedLearningRate.HasValue)
        {
            _optimizer.LearningRate = annealedLearningRate.Value;
        }
    }

    public List<DpoStepResult> Train(IReadOnlyList<TokenizedPreferencePair> dataset, Action<DpoStepResult>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dataset);
        if (dataset.Count == 0) return [];
        if (!double.IsFinite(_args.SftAnchorWeight) || _args.SftAnchorWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(_args.SftAnchorWeight), "SftAnchorWeight must be finite and non-negative.");

        int gradAccumSteps = Math.Max(1, _args.GradientAccumulationSteps);
        var cachedRefLogProbs = LoadOrComputeRefLogProbs(dataset);

        int stepsPerEpoch = (dataset.Count + gradAccumSteps - 1) / gradAccumSteps;
        int plannedTotalSteps = _args.MaxSteps.HasValue && _args.MaxSteps.Value > 0
            ? _args.MaxSteps.Value
            : Math.Max(1, _args.Epochs * stepsPerEpoch);

        var results = new List<DpoStepResult>(plannedTotalSteps);
        int globalStep = _startingStep;
        int totalSteps = Math.Max(globalStep + 1, plannedTotalSteps);

        for (int epoch = 0; epoch < _args.Epochs; epoch++)
        {
            if (_args.MaxSteps.HasValue && globalStep >= _args.MaxSteps.Value) break;

            int accumCounter = 0;
            double accumLoss = 0.0;
            double accumChosenReward = 0.0;
            double accumRejectedReward = 0.0;
            double accumMargin = 0.0;
            double accumAccuracy = 0.0;
            _optimizer.ZeroGrad();

            for (int idx = 0; idx < dataset.Count; idx++)
            {
                var pair = dataset[idx];
                var (refChosen, refRejected) = cachedRefLogProbs[idx];

                // Pass 1: Evaluate policy log-probs without gradient allocation
                double policyChosenLogP = ComputeSequenceLogProb(_policyModel, pair.ChosenInputIds, pair.ChosenLabels, _args.LengthNormalizedLogProbs, out int chosenTokens, out double chosenAvgNll);
                double policyRejectedLogP = ComputeSequenceLogProb(_policyModel, pair.RejectedInputIds, pair.RejectedLabels, _args.LengthNormalizedLogProbs, out int rejectedTokens, out _);

                var dpo = DpoLoss.Compute(
                    policyChosenLogP,
                    policyRejectedLogP,
                    refChosen,
                    refRejected,
                    _args.Beta,
                    _args.LabelSmoothing);

                double totalPairLoss = dpo.Loss + (_args.SftAnchorWeight > 0.0 && chosenTokens > 0 ? _args.SftAnchorWeight * chosenAvgNll : 0.0);

                if (chosenTokens > 0 && rejectedTokens > 0 && double.IsFinite(totalPairLoss))
                {
                    // Pass 2: Accumulate exact scaled gradients directly via lossScale (zero chosenGradBuffers clone)
                    double scaleW = _args.LengthNormalizedLogProbs ? 1.0 : chosenTokens;
                    double scaleL = _args.LengthNormalizedLogProbs ? 1.0 : rejectedTokens;
                    float lossScaleChosen = (float)((-dpo.GradPolicyChosen * scaleW) + _args.SftAnchorWeight);
                    float lossScaleRejected = (float)(-dpo.GradPolicyRejected * scaleL);

                    _policyModel.ForwardLossAndBackward(pair.ChosenInputIds, pair.ChosenLabels, _args.CheckpointActivations, lossScaleChosen);
                    _policyModel.ForwardLossAndBackward(pair.RejectedInputIds, pair.RejectedLabels, _args.CheckpointActivations, lossScaleRejected);

                    accumCounter++;
                    accumLoss += totalPairLoss;
                    accumChosenReward += dpo.ChosenReward;
                    accumRejectedReward += dpo.RejectedReward;
                    accumMargin += dpo.Margin;
                    accumAccuracy += dpo.Margin > 0.0 ? 1.0 : 0.0;
                }

                bool stepBoundary = accumCounter >= gradAccumSteps || idx == dataset.Count - 1;
                if (stepBoundary)
                {
                    globalStep++;
                    if (accumCounter > 0)
                    {
                        if (accumCounter > 1)
                        {
                            StatefulAdamW.ScaleGradients(_policyModel.TrainableParameters, 1.0f / accumCounter);
                        }

                        _optimizer.LearningRate = LoraTrainer.ComputeCosineLr(
                            globalStep,
                            totalSteps,
                            _args.LearningRate,
                            _args.WarmupRatio,
                            minLr: _args.LearningRate * 0.1f);
                        StatefulAdamW.ClipGradients(_policyModel.TrainableParameters, _args.MaxGradNorm);
                        _optimizer.Step();
                        _optimizer.ZeroGrad();
                    }

                    double denom = Math.Max(1, accumCounter);
                    var stepResult = new DpoStepResult(
                        globalStep,
                        accumCounter > 0 ? accumLoss / denom : totalPairLoss,
                        accumCounter > 0 ? accumChosenReward / denom : dpo.ChosenReward,
                        accumCounter > 0 ? accumRejectedReward / denom : dpo.RejectedReward,
                        accumCounter > 0 ? accumMargin / denom : dpo.Margin,
                        accumCounter > 0 ? accumAccuracy / denom : (dpo.Margin > 0.0 ? 1.0 : 0.0));
                    results.Add(stepResult);
                    onStep?.Invoke(stepResult);

                    if (_args.LoggingSteps <= 1 || globalStep % _args.LoggingSteps == 0 || globalStep == totalSteps)
                    {
                        GlacierDiagnostics.LogInformation(
                            $"[DPO] Step {globalStep}/{totalSteps} | Loss={stepResult.Loss:F4} | Margin={stepResult.RewardMargin:F4} | Acc={stepResult.RewardAccuracy:P0} | LR={_optimizer.LearningRate:E2}");
                    }

                    if (_args.SaveSteps > 0 && globalStep % _args.SaveSteps == 0 && !string.IsNullOrWhiteSpace(_args.OutputDir))
                    {
                        string ckptDir = Path.Combine(_args.OutputDir, $"checkpoint-{globalStep}");
                        _policyModel.SaveAdapter(ckptDir);
                        _optimizer.SaveState(LoraTrainer.GetOptimizerStatePath(ckptDir));
                    }

                    accumCounter = 0;
                    accumLoss = 0.0;
                    accumChosenReward = 0.0;
                    accumRejectedReward = 0.0;
                    accumMargin = 0.0;
                    accumAccuracy = 0.0;

                    if (_args.MaxSteps.HasValue && globalStep >= _args.MaxSteps.Value)
                    {
                        break;
                    }
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(_args.OutputDir))
        {
            _policyModel.SaveAdapter(_args.OutputDir);
            _optimizer.SaveState(LoraTrainer.GetOptimizerStatePath(_args.OutputDir));
        }

        GlacierDiagnostics.LogInformation($"[DPO] Completed {globalStep} DPO optimization steps.");
        return results;
    }

    private (double RefChosenLogP, double RefRejectedLogP)[] LoadOrComputeRefLogProbs(IReadOnlyList<TokenizedPreferencePair> dataset)
    {
        string? cachePath = _args.ReferenceLogProbsCachePath;
        if (!string.IsNullOrWhiteSpace(cachePath) && File.Exists(cachePath))
        {
            if (TryLoadRefCache(cachePath, dataset.Count, _args.LengthNormalizedLogProbs, out var cached))
            {
                GlacierDiagnostics.LogInformation($"[DPO] Loaded {dataset.Count} cached reference log-probabilities from {cachePath}.");
                return cached;
            }
        }

        var computed = new (double RefChosenLogP, double RefRejectedLogP)[dataset.Count];
        var refEvalModel = _referenceModel ?? _policyModel;
        for (int i = 0; i < dataset.Count; i++)
        {
            var pair = dataset[i];
            double refChosen = ComputeSequenceLogProb(refEvalModel, pair.ChosenInputIds, pair.ChosenLabels, _args.LengthNormalizedLogProbs, out _, out _);
            double refRejected = ComputeSequenceLogProb(refEvalModel, pair.RejectedInputIds, pair.RejectedLabels, _args.LengthNormalizedLogProbs, out _, out _);
            computed[i] = (refChosen, refRejected);
        }

        if (!string.IsNullOrWhiteSpace(cachePath))
        {
            SaveRefCache(cachePath, _args.LengthNormalizedLogProbs, computed);
        }

        return computed;
    }

    private static bool TryLoadRefCache(
        string path,
        int expectedCount,
        bool expectedLengthNorm,
        out (double RefChosenLogP, double RefRejectedLogP)[] entries)
    {
        entries = [];
        try
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            if (fs.Length < 16) return false;
            if (br.ReadUInt32() != RefCacheMagic) return false;
            if (br.ReadInt32() != RefCacheVersion) return false;
            if (br.ReadBoolean() != expectedLengthNorm) return false;
            int count = br.ReadInt32();
            if (count != expectedCount) return false;
            if (fs.Length - fs.Position != (long)count * 2 * sizeof(double)) return false;

            var loaded = new (double, double)[count];
            for (int i = 0; i < count; i++)
            {
                double c = br.ReadDouble();
                double r = br.ReadDouble();
                if (!double.IsFinite(c) || !double.IsFinite(r)) return false;
                loaded[i] = (c, r);
            }
            entries = loaded;
            return true;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static void SaveRefCache(string path, bool lengthNorm, (double RefChosenLogP, double RefRejectedLogP)[] entries)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        string tmpPath = LoraBinaryFormat.CreateTempSiblingPath(path);
        try
        {
            using (var fs = File.Create(tmpPath))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(RefCacheMagic);
                bw.Write(RefCacheVersion);
                bw.Write(lengthNorm);
                bw.Write(entries.Length);
                for (int i = 0; i < entries.Length; i++)
                {
                    bw.Write(entries[i].RefChosenLogP);
                    bw.Write(entries[i].RefRejectedLogP);
                }
                bw.Flush();
                fs.Flush(true);
            }
            File.Move(tmpPath, path, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(tmpPath)) File.Delete(tmpPath); } catch { }
            throw;
        }
    }

    private static double ComputeSequenceLogProb(
        GgufLoraModel model,
        int[] inputIds,
        int[] labels,
        bool lengthNormalized,
        out int validTokens,
        out double avgNll)
    {
        float ceLoss = model.EvaluateLoss(inputIds, labels, out validTokens);
        if (validTokens <= 0 || !float.IsFinite(ceLoss))
        {
            avgNll = 0.0;
            return 0.0;
        }
        avgNll = ceLoss;
        return lengthNormalized ? -ceLoss : -(double)ceLoss * validTokens;
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
