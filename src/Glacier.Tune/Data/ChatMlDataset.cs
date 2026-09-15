using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Glacier.Inference.Tokenizer;

namespace Glacier.Tune.Data;

public sealed record TrainingExample(int[] InputIds, int[] Labels);

/// <summary>
/// Reads ChatML JSONL datasets, applies chat formatting, and produces tokenized training batches
/// with prompt masking (Labels = -100 for system/user prompts so loss is only evaluated on assistant responses).
/// </summary>
public sealed class ChatMlDataset
{
    public const int IgnoreIndex = -100;
    private readonly List<TrainingExample> _examples = [];

    public IReadOnlyList<TrainingExample> Examples => _examples;
    public int Count => _examples.Count;

    public static ChatMlDataset FromFile(string filePath, BpeTokenizer tokenizer, int maxSeqLength = 1024)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Dataset file not found: {filePath}");

        var dataset = new ChatMlDataset();
        using var reader = new StreamReader(filePath);
        string? line;

        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            ChatConversation? conv;
            try
            {
                conv = JsonSerializer.Deserialize<ChatConversation>(line);
            }
            catch
            {
                continue;
            }

            if (conv?.Messages == null || conv.Messages.Count == 0) continue;

            var example = FormatAndTokenize(conv, tokenizer, maxSeqLength);
            if (example.InputIds.Length > 0)
            {
                dataset._examples.Add(example);
            }
        }

        return dataset;
    }

    public static TrainingExample FormatAndTokenize(ChatConversation conv, BpeTokenizer tokenizer, int maxSeqLength)
    {
        var inputIds = new List<int>(maxSeqLength);
        var labels = new List<int>(maxSeqLength);

        foreach (var msg in conv.Messages)
        {
            string formatted = $"<|im_start|>{msg.Role}\n{msg.Content}<|im_end|>\n";
            int[] tokens = tokenizer.Encode(formatted);

            bool isAssistant = msg.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase);

            for (int i = 0; i < tokens.Length; i++)
            {
                if (inputIds.Count >= maxSeqLength) break;

                inputIds.Add(tokens[i]);
                // Mask prompt tokens with IgnoreIndex (-100) so cross entropy ignores them
                labels.Add(isAssistant ? tokens[i] : IgnoreIndex);
            }

            if (inputIds.Count >= maxSeqLength) break;
        }

        return new TrainingExample(inputIds.ToArray(), labels.ToArray());
    }
}
