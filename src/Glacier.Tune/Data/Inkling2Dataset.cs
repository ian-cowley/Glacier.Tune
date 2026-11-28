namespace Glacier.Tune.Data;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Reads JSONL chat datasets and tokenizes them using Thinking Machines Lab's official Inkling-2-Mini
/// control tokens (&lt;|message_system|&gt;, &lt;|message_user|&gt;, &lt;|message_model|&gt;, &lt;|message_tool|&gt;,
/// &lt;|content_text|&gt;, &lt;|content_thinking|&gt;, &lt;|end_message|&gt;, &lt;|content_model_end_sampling|&gt;)
/// with completion-only label masking (Labels = -100 on all non-assistant tokens and headers).
/// </summary>
public sealed class Inkling2Dataset
{
    public const int IgnoreIndex = -100;
    private readonly List<TrainingExample> _examples = [];

    public IReadOnlyList<TrainingExample> Examples => _examples;
    public int Count => _examples.Count;

    public static Inkling2Dataset FromFile(string filePath, BpeTokenizer tokenizer, int maxSeqLength = 1024, int? maxExamples = null)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Dataset file not found: {filePath}");

        var dataset = new Inkling2Dataset();
        using var reader = new StreamReader(filePath);
        string? line;

        while ((line = reader.ReadLine()) != null)
        {
            if (maxExamples.HasValue && dataset._examples.Count >= maxExamples.Value)
                break;
            if (string.IsNullOrWhiteSpace(line)) continue;

            try
            {
                using var doc = JsonDocument.Parse(line);
                if (!doc.RootElement.TryGetProperty("messages", out var messagesElem) ||
                    messagesElem.ValueKind != JsonValueKind.Array)
                    continue;

                var inputIds = new List<int>(Math.Min(maxSeqLength, 1024));
                var labels = new List<int>(Math.Min(maxSeqLength, 1024));
                int supervisedCount = 0;

                foreach (var item in messagesElem.EnumerateArray())
                {
                    if (inputIds.Count >= maxSeqLength) break;

                    string role = item.TryGetProperty("role", out var r) ? (r.GetString() ?? "") : "";
                    string content = item.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String
                        ? (c.GetString() ?? "") : "";
                    string thinking = item.TryGetProperty("reasoning_content", out var rc) && rc.ValueKind == JsonValueKind.String
                        ? (rc.GetString() ?? "") : "";

                    // Also include OpenAI-style tool_calls JSON if present on assistant turns
                    if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase) &&
                        string.IsNullOrEmpty(content) &&
                        item.TryGetProperty("tool_calls", out var tc) &&
                        tc.ValueKind == JsonValueKind.Array &&
                        tc.GetArrayLength() > 0)
                    {
                        content = tc.GetRawText();
                    }

                    if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                    {
                        if (string.IsNullOrEmpty(content) && string.IsNullOrEmpty(thinking))
                            continue;

                        if (!string.IsNullOrEmpty(thinking))
                        {
                            AppendTokens(tokenizer.Encode("<|message_model|><|content_thinking|>"), false, inputIds, labels, maxSeqLength, ref supervisedCount);
                            AppendTokens(tokenizer.Encode($"{thinking}<|end_message|><|content_text|>{content}<|end_message|><|content_model_end_sampling|>"), true, inputIds, labels, maxSeqLength, ref supervisedCount);
                        }
                        else
                        {
                            AppendTokens(tokenizer.Encode("<|message_model|><|content_text|>"), false, inputIds, labels, maxSeqLength, ref supervisedCount);
                            AppendTokens(tokenizer.Encode($"{content}<|end_message|><|content_model_end_sampling|>"), true, inputIds, labels, maxSeqLength, ref supervisedCount);
                        }
                    }
                    else
                    {
                        string roleTag = role.ToLowerInvariant() switch
                        {
                            "system" => "<|message_system|>",
                            "tool" => "<|message_tool|>",
                            _ => "<|message_user|>"
                        };
                        AppendTokens(tokenizer.Encode($"{roleTag}<|content_text|>{content}<|end_message|>"), false, inputIds, labels, maxSeqLength, ref supervisedCount);
                    }
                }

                // Causal Language Modeling shift matching ChatMlDataset: token at position t predicts token at position t + 1
                var shiftedLabels = new int[labels.Count];
                int shiftedSupervisedCount = 0;
                for (int i = 0; i < labels.Count - 1; i++)
                {
                    int nextLabel = labels[i + 1];
                    shiftedLabels[i] = nextLabel;
                    if (nextLabel != IgnoreIndex)
                    {
                        shiftedSupervisedCount++;
                    }
                }
                if (labels.Count > 0)
                {
                    shiftedLabels[labels.Count - 1] = IgnoreIndex;
                }

                if (inputIds.Count >= 4 && shiftedSupervisedCount >= 4)
                    dataset._examples.Add(new TrainingExample(inputIds.ToArray(), shiftedLabels));
            }
            catch (JsonException)
            {
                continue;
            }
        }

        return dataset;
    }

    private static void AppendTokens(int[] tokens, bool supervise, List<int> inputIds, List<int> labels, int maxSeqLength, ref int supervisedCount)
    {
        for (int i = 0; i < tokens.Length && inputIds.Count < maxSeqLength; i++)
        {
            inputIds.Add(tokens[i]);
            if (supervise)
            {
                labels.Add(tokens[i]);
                supervisedCount++;
            }
            else
            {
                labels.Add(IgnoreIndex);
            }
        }
    }
}
