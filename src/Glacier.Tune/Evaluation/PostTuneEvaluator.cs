using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Glacier.Inference.Tokenizer;
using Glacier.Tune.Config;
using Glacier.Tune.Model;

namespace Glacier.Tune.Evaluation;

public sealed record EvalExample(string BasePrompt, string FineTunedPrompt, string Expected);

public sealed class EvalOptions
{
    public int MaxNewTokens { get; set; } = 64;
    public int MaxSeqLength { get; set; } = 1024;
    public bool IgnoreCase { get; set; } = true;
    public bool NormalizeWhitespace { get; set; } = true;
    public bool StripSystemTurnForFineTuned { get; set; } = false;
    public bool ApplyChatTemplateToPlainPrompts { get; set; } = false;
    public string? BaseSystemPromptPrefix { get; set; } = null;
}

public sealed record EvalSampleResult(
    int Index,
    string BasePrompt,
    string FineTunedPrompt,
    string Expected,
    string BasePrediction,
    string TunedPrediction,
    double BaseExactMatch,
    double TunedExactMatch,
    double BaseF1,
    double TunedF1,
    int BasePromptTokens,
    int TunedPromptTokens);

public sealed class EvalReport
{
    public int TotalExamples { get; init; }
    public double BaseExactMatch { get; init; }
    public double TunedExactMatch { get; init; }
    public double BaseF1 { get; init; }
    public double TunedF1 { get; init; }
    public long TotalBasePromptTokens { get; init; }
    public long TotalTunedPromptTokens { get; init; }
    public double PromptTokenReductionPercent { get; init; }
    public IReadOnlyList<EvalSampleResult> Samples { get; init; } = [];

    public string FormatSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine("================================================================================");
        sb.AppendLine("                    GLACIER.TUNE: POST-TUNE EVALUATION REPORT                   ");
        sb.AppendLine("================================================================================");
        sb.AppendLine($"  Examples Evaluated:      {TotalExamples}");
        sb.AppendLine($"  Base Exact Match:        {BaseExactMatch * 100.0:F2}%");
        sb.AppendLine($"  Fine-Tuned Exact Match:  {TunedExactMatch * 100.0:F2}%");
        sb.AppendLine($"  Base Token F1:           {BaseF1 * 100.0:F2}%");
        sb.AppendLine($"  Fine-Tuned Token F1:     {TunedF1 * 100.0:F2}%");
        sb.AppendLine($"  Base Prompt Tokens:      {TotalBasePromptTokens:N0}");
        sb.AppendLine($"  Tuned Prompt Tokens:     {TotalTunedPromptTokens:N0}");
        sb.AppendLine($"  Prompt Token Reduction:  {PromptTokenReductionPercent:F2}%");
        return sb.ToString();
    }

    public string ToJson()
    {
        using var ms = new MemoryStream();
        using var json = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteNumber("total_examples", TotalExamples);
        json.WriteNumber("base_exact_match", BaseExactMatch);
        json.WriteNumber("tuned_exact_match", TunedExactMatch);
        json.WriteNumber("base_f1", BaseF1);
        json.WriteNumber("tuned_f1", TunedF1);
        json.WriteNumber("total_base_prompt_tokens", TotalBasePromptTokens);
        json.WriteNumber("total_tuned_prompt_tokens", TotalTunedPromptTokens);
        json.WriteNumber("prompt_token_reduction_percent", PromptTokenReductionPercent);
        json.WritePropertyName("samples");
        json.WriteStartArray();
        for (int i = 0; i < Samples.Count; i++)
        {
            var s = Samples[i];
            json.WriteStartObject();
            json.WriteNumber("index", s.Index);
            json.WriteString("expected", s.Expected);
            json.WriteString("base_prediction", s.BasePrediction);
            json.WriteString("tuned_prediction", s.TunedPrediction);
            json.WriteNumber("base_exact_match", s.BaseExactMatch);
            json.WriteNumber("tuned_exact_match", s.TunedExactMatch);
            json.WriteNumber("base_f1", s.BaseF1);
            json.WriteNumber("tuned_f1", s.TunedF1);
            json.WriteNumber("base_prompt_tokens", s.BasePromptTokens);
            json.WriteNumber("tuned_prompt_tokens", s.TunedPromptTokens);
            json.WriteEndObject();
        }
        json.WriteEndArray();
        json.WriteEndObject();
        json.Flush();
        return Encoding.UTF8.GetString(ms.ToArray());
    }
}

/// <summary>
/// AOT-safe post-tune evaluation harness comparing Base vs. Fine-Tuned predictions on <c>eval.jsonl</c>
/// using Exact Match (EM), token-level F1, and prompt token reduction %.
/// </summary>
public static unsafe class PostTuneEvaluator
{
    public static IReadOnlyList<EvalExample> LoadExamples(string evalJsonlPath, EvalOptions? options = null, BpeTokenizer? tokenizer = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(evalJsonlPath);
        if (!File.Exists(evalJsonlPath))
            throw new FileNotFoundException($"Evaluation JSONL file not found: {evalJsonlPath}");

        using var reader = new StreamReader(evalJsonlPath);
        return ParseExamples(reader, options, tokenizer);
    }

    public static IReadOnlyList<EvalExample> ParseExamples(TextReader reader, EvalOptions? options = null, BpeTokenizer? tokenizer = null)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _ = tokenizer; // Reserved for future template introspection while preserving signature parity.
        var opts = options ?? new EvalOptions();
        var examples = new List<EvalExample>();

        string? line;
        int lineNumber = 0;
        while ((line = reader.ReadLine()) != null)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line)) continue;

            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                {
                    throw new InvalidDataException($"Expected JSON object at line {lineNumber}.");
                }

                string? explicitExpected = TryGetFirstString(root, "expected", "completion", "target", "reference", "answer", "output");
                string? explicitBasePrompt = TryGetFirstString(root, "base_prompt");
                bool hasBaseMessages = root.TryGetProperty("base_messages", out var baseMessagesElem) && baseMessagesElem.ValueKind == JsonValueKind.Array;

                string fineTunedPrompt;
                string basePrompt;
                string expected;

                if (root.TryGetProperty("messages", out var messagesElem) && messagesElem.ValueKind == JsonValueKind.Array)
                {
                    var turns = ParseMessageTurns(messagesElem, lineNumber);
                    if (turns.Count == 0)
                    {
                        throw new InvalidDataException($"Empty 'messages' array at line {lineNumber}.");
                    }

                    int promptEndIndex = turns.Count;
                    if (explicitExpected != null)
                    {
                        expected = explicitExpected;
                        if (turns[^1].Role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                        {
                            promptEndIndex = turns.Count - 1;
                        }
                    }
                    else
                    {
                        int lastAssistantIdx = -1;
                        for (int i = turns.Count - 1; i >= 0; i--)
                        {
                            if (turns[i].Role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                            {
                                lastAssistantIdx = i;
                                break;
                            }
                        }

                        if (lastAssistantIdx < 0)
                        {
                            throw new InvalidDataException($"No assistant turn or 'expected' field found at line {lineNumber}.");
                        }

                        expected = turns[lastAssistantIdx].Content;
                        promptEndIndex = lastAssistantIdx;
                    }

                    fineTunedPrompt = FormatChatPrompt(turns, promptEndIndex, stripSystemTurn: opts.StripSystemTurnForFineTuned);
                    if (explicitBasePrompt != null)
                    {
                        basePrompt = explicitBasePrompt;
                    }
                    else if (hasBaseMessages)
                    {
                        var baseTurns = ParseMessageTurns(baseMessagesElem, lineNumber);
                        int baseEnd = baseTurns.Count > 0 && baseTurns[^1].Role.Equals("assistant", StringComparison.OrdinalIgnoreCase)
                            ? baseTurns.Count - 1
                            : baseTurns.Count;
                        basePrompt = FormatChatPrompt(baseTurns, baseEnd, stripSystemTurn: false);
                    }
                    else
                    {
                        basePrompt = FormatChatPrompt(turns, promptEndIndex, stripSystemTurn: false);
                        if (!string.IsNullOrEmpty(opts.BaseSystemPromptPrefix))
                        {
                            basePrompt = opts.BaseSystemPromptPrefix + basePrompt;
                        }
                    }
                }
                else
                {
                    string? prompt = TryGetFirstString(root, "prompt", "input", "instruction");
                    if (prompt == null || explicitExpected == null)
                    {
                        throw new InvalidDataException($"Missing prompt or expected field at line {lineNumber}.");
                    }

                    expected = explicitExpected;
                    string? systemTurn = TryGetFirstString(root, "system");
                    if (!string.IsNullOrEmpty(systemTurn) || (opts.ApplyChatTemplateToPlainPrompts && !prompt.Contains("<|im_start|>", StringComparison.Ordinal)))
                    {
                        var sb = new StringBuilder();
                        if (!string.IsNullOrEmpty(systemTurn) && !opts.StripSystemTurnForFineTuned)
                        {
                            sb.Append("<|im_start|>system\n").Append(systemTurn).Append("<|im_end|>\n");
                        }
                        sb.Append("<|im_start|>user\n").Append(prompt).Append("<|im_end|>\n<|im_start|>assistant\n");
                        fineTunedPrompt = sb.ToString();
                    }
                    else
                    {
                        fineTunedPrompt = prompt;
                    }

                    if (explicitBasePrompt != null)
                    {
                        basePrompt = explicitBasePrompt;
                    }
                    else if (hasBaseMessages)
                    {
                        var baseTurns = ParseMessageTurns(baseMessagesElem, lineNumber);
                        int baseEnd = baseTurns.Count > 0 && baseTurns[^1].Role.Equals("assistant", StringComparison.OrdinalIgnoreCase)
                            ? baseTurns.Count - 1
                            : baseTurns.Count;
                        basePrompt = FormatChatPrompt(baseTurns, baseEnd, stripSystemTurn: false);
                    }
                    else if (!string.IsNullOrEmpty(systemTurn))
                    {
                        var sb = new StringBuilder();
                        sb.Append("<|im_start|>system\n").Append(systemTurn).Append("<|im_end|>\n");
                        sb.Append("<|im_start|>user\n").Append(prompt).Append("<|im_end|>\n<|im_start|>assistant\n");
                        basePrompt = string.IsNullOrEmpty(opts.BaseSystemPromptPrefix) ? sb.ToString() : opts.BaseSystemPromptPrefix + sb.ToString();
                    }
                    else
                    {
                        basePrompt = string.IsNullOrEmpty(opts.BaseSystemPromptPrefix) ? fineTunedPrompt : opts.BaseSystemPromptPrefix + fineTunedPrompt;
                    }
                }

                examples.Add(new EvalExample(basePrompt, fineTunedPrompt, expected));
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Invalid JSON at evaluation line {lineNumber}: {ex.Message}", ex);
            }
        }

        return examples;
    }

    public static EvalReport Evaluate(
        string ggufPath,
        string adapterPath,
        string evalJsonlPath,
        EvalOptions? options = null,
        string device = "auto")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ggufPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(adapterPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(evalJsonlPath);

        var opts = options ?? new EvalOptions();
        var (rank, alpha) = ReadAdapterHeader(adapterPath);
        var loraConfig = new LoraConfig
        {
            Rank = rank,
            Alpha = alpha,
            TargetModules = ["all"]
        };

        using var model = GgufLoraModel.Load(ggufPath, loraConfig, device);
        var examples = LoadExamples(evalJsonlPath, opts, model.Tokenizer);

        int n = examples.Count;
        var basePredictions = new string[n];
        var baseTokenCounts = new int[n];
        for (int i = 0; i < n; i++)
        {
            baseTokenCounts[i] = model.Tokenizer.Encode(examples[i].BasePrompt).Length;
            basePredictions[i] = GenerateGreedy(model, examples[i].BasePrompt, opts.MaxNewTokens, opts.MaxSeqLength);
        }

        model.LoadAdapter(adapterPath);

        var tunedPredictions = new string[n];
        var tunedTokenCounts = new int[n];
        for (int i = 0; i < n; i++)
        {
            tunedTokenCounts[i] = model.Tokenizer.Encode(examples[i].FineTunedPrompt).Length;
            tunedPredictions[i] = GenerateGreedy(model, examples[i].FineTunedPrompt, opts.MaxNewTokens, opts.MaxSeqLength);
        }

        return BuildReport(examples, basePredictions, tunedPredictions, baseTokenCounts, tunedTokenCounts, opts);
    }

    public static EvalReport Evaluate(
        GgufLoraModel baseModel,
        GgufLoraModel tunedModel,
        string evalJsonlPath,
        EvalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(baseModel);
        ArgumentNullException.ThrowIfNull(tunedModel);
        var opts = options ?? new EvalOptions();
        var examples = LoadExamples(evalJsonlPath, opts, baseModel.Tokenizer);
        return Evaluate(baseModel, tunedModel, examples, opts);
    }

    public static EvalReport Evaluate(
        GgufLoraModel baseModel,
        GgufLoraModel tunedModel,
        IReadOnlyList<EvalExample> examples,
        EvalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(baseModel);
        ArgumentNullException.ThrowIfNull(tunedModel);
        ArgumentNullException.ThrowIfNull(examples);

        var opts = options ?? new EvalOptions();
        int n = examples.Count;
        var basePredictions = new string[n];
        var tunedPredictions = new string[n];
        var baseTokenCounts = new int[n];
        var tunedTokenCounts = new int[n];

        for (int i = 0; i < n; i++)
        {
            var ex = examples[i];
            baseTokenCounts[i] = baseModel.Tokenizer.Encode(ex.BasePrompt).Length;
            tunedTokenCounts[i] = tunedModel.Tokenizer.Encode(ex.FineTunedPrompt).Length;
            basePredictions[i] = GenerateGreedy(baseModel, ex.BasePrompt, opts.MaxNewTokens, opts.MaxSeqLength);
            tunedPredictions[i] = GenerateGreedy(tunedModel, ex.FineTunedPrompt, opts.MaxNewTokens, opts.MaxSeqLength);
        }

        return BuildReport(examples, basePredictions, tunedPredictions, baseTokenCounts, tunedTokenCounts, opts);
    }

    public static string GenerateGreedy(GgufLoraModel model, string prompt, int maxNewTokens = 64, int maxSeqLength = 1024)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (string.IsNullOrEmpty(prompt) || maxNewTokens <= 0)
        {
            return string.Empty;
        }

        int[] promptTokens = model.Tokenizer.Encode(prompt);
        if (promptTokens.Length == 0)
        {
            return string.Empty;
        }

        int maxPromptTokens = maxSeqLength > 1 ? maxSeqLength - 1 : maxSeqLength;
        var tokens = new List<int>(promptTokens.Length + maxNewTokens);
        if (maxPromptTokens > 0 && promptTokens.Length > maxPromptTokens)
        {
            int start = promptTokens.Length - maxPromptTokens;
            for (int i = start; i < promptTokens.Length; i++)
            {
                tokens.Add(promptTokens[i]);
            }
        }
        else
        {
            tokens.AddRange(promptTokens);
        }

        var generated = new List<int>(maxNewTokens);
        int vocabSize = model.VocabSize;

        for (int step = 0; step < maxNewTokens; step++)
        {
            if (maxSeqLength > 0 && tokens.Count >= maxSeqLength)
            {
                break;
            }

            using var logits = model.ForwardLastTokenLogits(CollectionsMarshal.AsSpan(tokens));
            float* pLogits = (float*)logits.DataPointer;
            int bestToken = -1;
            float bestVal = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                float val = pLogits[v];
                if (!float.IsNaN(val) && val > bestVal)
                {
                    bestVal = val;
                    bestToken = v;
                }
            }

            if (bestToken < 0 || model.Tokenizer.IsStopToken(bestToken) || bestToken == model.Tokenizer.EosTokenId)
            {
                break;
            }

            generated.Add(bestToken);
            tokens.Add(bestToken);
        }

        return generated.Count > 0 ? model.Tokenizer.Decode(CollectionsMarshal.AsSpan(generated)) : string.Empty;
    }

    public static double ComputeExactMatch(string prediction, string expected, bool ignoreCase = true, bool normalizeWhitespace = true)
    {
        string normPred = NormalizeText(prediction, ignoreCase, normalizeWhitespace);
        string normExp = NormalizeText(expected, ignoreCase, normalizeWhitespace);
        return string.Equals(normPred, normExp, ignoreCase ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal) ? 1.0 : 0.0;
    }

    public static double ComputeTokenF1(string prediction, string expected, bool ignoreCase = true, bool normalizeWhitespace = true)
    {
        string normPred = NormalizeText(prediction, ignoreCase, normalizeWhitespace);
        string normExp = NormalizeText(expected, ignoreCase, normalizeWhitespace);

        var predTokens = SplitWords(normPred);
        var expTokens = SplitWords(normExp);

        if (predTokens.Count == 0 && expTokens.Count == 0) return 1.0;
        if (predTokens.Count == 0 || expTokens.Count == 0) return 0.0;

        var comparer = ignoreCase ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        var expCounts = new Dictionary<string, int>(comparer);
        foreach (string tok in expTokens)
        {
            expCounts[tok] = expCounts.TryGetValue(tok, out int c) ? c + 1 : 1;
        }

        int common = 0;
        foreach (string tok in predTokens)
        {
            if (expCounts.TryGetValue(tok, out int rem) && rem > 0)
            {
                common++;
                expCounts[tok] = rem - 1;
            }
        }

        if (common == 0) return 0.0;
        double precision = (double)common / predTokens.Count;
        double recall = (double)common / expTokens.Count;
        return 2.0 * precision * recall / (precision + recall);
    }

    public static double ComputePromptTokenReductionPercent(long totalBaseTokens, long totalTunedTokens)
    {
        if (totalBaseTokens <= 0) return 0.0;
        return (double)(totalBaseTokens - totalTunedTokens) / totalBaseTokens * 100.0;
    }

    private static EvalReport BuildReport(
        IReadOnlyList<EvalExample> examples,
        string[] basePredictions,
        string[] tunedPredictions,
        int[] baseTokenCounts,
        int[] tunedTokenCounts,
        EvalOptions opts)
    {
        int n = examples.Count;
        var samples = new List<EvalSampleResult>(n);
        double sumBaseEm = 0.0;
        double sumTunedEm = 0.0;
        double sumBaseF1 = 0.0;
        double sumTunedF1 = 0.0;
        long totalBaseTokens = 0;
        long totalTunedTokens = 0;

        for (int i = 0; i < n; i++)
        {
            var ex = examples[i];
            double baseEm = ComputeExactMatch(basePredictions[i], ex.Expected, opts.IgnoreCase, opts.NormalizeWhitespace);
            double tunedEm = ComputeExactMatch(tunedPredictions[i], ex.Expected, opts.IgnoreCase, opts.NormalizeWhitespace);
            double baseF1 = ComputeTokenF1(basePredictions[i], ex.Expected, opts.IgnoreCase, opts.NormalizeWhitespace);
            double tunedF1 = ComputeTokenF1(tunedPredictions[i], ex.Expected, opts.IgnoreCase, opts.NormalizeWhitespace);

            sumBaseEm += baseEm;
            sumTunedEm += tunedEm;
            sumBaseF1 += baseF1;
            sumTunedF1 += tunedF1;
            totalBaseTokens += baseTokenCounts[i];
            totalTunedTokens += tunedTokenCounts[i];

            samples.Add(new EvalSampleResult(
                i,
                ex.BasePrompt,
                ex.FineTunedPrompt,
                ex.Expected,
                basePredictions[i],
                tunedPredictions[i],
                baseEm,
                tunedEm,
                baseF1,
                tunedF1,
                baseTokenCounts[i],
                tunedTokenCounts[i]));
        }

        double denom = n > 0 ? n : 1.0;
        return new EvalReport
        {
            TotalExamples = n,
            BaseExactMatch = n > 0 ? sumBaseEm / denom : 0.0,
            TunedExactMatch = n > 0 ? sumTunedEm / denom : 0.0,
            BaseF1 = n > 0 ? sumBaseF1 / denom : 0.0,
            TunedF1 = n > 0 ? sumTunedF1 / denom : 0.0,
            TotalBasePromptTokens = totalBaseTokens,
            TotalTunedPromptTokens = totalTunedTokens,
            PromptTokenReductionPercent = ComputePromptTokenReductionPercent(totalBaseTokens, totalTunedTokens),
            Samples = samples
        };
    }

    private static (int Rank, float Alpha) ReadAdapterHeader(string adapterPath)
    {
        string binPath = adapterPath;
        if (Directory.Exists(adapterPath))
        {
            string candidate = Path.Combine(adapterPath, "adapter_model.bin");
            if (File.Exists(candidate)) binPath = candidate;
        }

        if (!File.Exists(binPath))
            throw new FileNotFoundException($"LoRA adapter file not found: {adapterPath}");

        using var fs = File.OpenRead(binPath);
        using var br = new BinaryReader(fs);
        Glacier.Tune.Model.LoraBinaryFormat.ValidateAndReadHeader(fs, br, out _, out int rank, out float alpha, out _);
        return (rank, alpha);
    }

    private static List<(string Role, string Content)> ParseMessageTurns(JsonElement messagesElem, int lineNumber)
    {
        var turns = new List<(string Role, string Content)>();
        foreach (var item in messagesElem.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException($"Expected message object in array at line {lineNumber}.");
            }
            string role = item.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String ? (r.GetString() ?? "") : "";
            string content = item.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String ? (c.GetString() ?? "") : "";
            turns.Add((role, content));
        }
        return turns;
    }

    private static string FormatChatPrompt(List<(string Role, string Content)> turns, int promptEndIndex, bool stripSystemTurn)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < promptEndIndex; i++)
        {
            if (stripSystemTurn && turns[i].Role.Equals("system", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            sb.Append("<|im_start|>").Append(turns[i].Role).Append('\n').Append(turns[i].Content).Append("<|im_end|>\n");
        }
        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    private static string? TryGetFirstString(JsonElement obj, params ReadOnlySpan<string> keys)
    {
        foreach (string key in keys)
        {
            if (obj.TryGetProperty(key, out var val) && val.ValueKind == JsonValueKind.String)
            {
                return val.GetString();
            }
        }
        return null;
    }

    private static string NormalizeText(string text, bool ignoreCase, bool normalizeWhitespace)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;
        if (normalizeWhitespace)
        {
            string trimmed = text.Trim();
            var sb = new StringBuilder(trimmed.Length);
            bool inSpace = false;
            for (int i = 0; i < trimmed.Length; i++)
            {
                char ch = trimmed[i];
                if (char.IsWhiteSpace(ch))
                {
                    if (!inSpace)
                    {
                        sb.Append(' ');
                        inSpace = true;
                    }
                }
                else
                {
                    sb.Append(ignoreCase ? char.ToLowerInvariant(ch) : ch);
                    inSpace = false;
                }
            }
            return sb.ToString();
        }
        return ignoreCase ? text.ToLowerInvariant() : text;
    }

    private static List<string> SplitWords(string normalized)
    {
        var list = new List<string>();
        if (string.IsNullOrEmpty(normalized)) return list;
        int start = -1;
        for (int i = 0; i < normalized.Length; i++)
        {
            if (char.IsWhiteSpace(normalized[i]))
            {
                if (start >= 0)
                {
                    list.Add(normalized.Substring(start, i - start));
                    start = -1;
                }
            }
            else if (start < 0)
            {
                start = i;
            }
        }
        if (start >= 0)
        {
            list.Add(normalized.Substring(start));
        }
        return list;
    }
}
