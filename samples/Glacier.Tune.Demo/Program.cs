using System;
using System.Diagnostics;
using System.IO;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Export;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;

if (args.Length > 0 && (args[0] == "--merge" || args[0] == "-m"))
{
    string? baseModel = args.Length > 1 ? args[1] : Environment.GetEnvironmentVariable("GLACIER_BASE_GGUF");
    string adapterPath = args.Length > 2
        ? args[2]
        : (Environment.GetEnvironmentVariable("GLACIER_LORA_ADAPTER") ?? Path.Combine(Directory.GetCurrentDirectory(), "tune_output", "adapter_model.bin"));
    string outModel = args.Length > 3
        ? args[3]
        : (Environment.GetEnvironmentVariable("GLACIER_MERGED_GGUF") ?? Path.Combine(Directory.GetCurrentDirectory(), "tune_output", "merged_model.gguf"));

    if (string.IsNullOrWhiteSpace(baseModel))
    {
        Console.WriteLine("Usage: dotnet run --project samples/Glacier.Tune.Demo -- --merge <base.gguf> [adapter_model.bin] [merged_out.gguf]");
        return;
    }

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("================================================================================");
    Console.WriteLine("         GLACIER.TUNE: PURE C# .NET 10 GGUF STANDALONE ADAPTER MERGER           ");
    Console.WriteLine("        Zero Python - Direct Native Memory SIMD Quantized Fusion                ");
    Console.WriteLine("================================================================================\n");
    Console.ResetColor();

    GgufMerger.Merge(baseModel, adapterPath, outModel);
    return;
}

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("              GLACIER.TUNE: PURE C# .NET 10 LLM FINE-TUNING DEMO                ");
Console.WriteLine("          Instant GGUF Zero-Copy Memory Mapping & LoRA Backpropagation           ");
Console.WriteLine("================================================================================\n");
Console.ResetColor();

string device = "auto";
int maxSteps = 3;
string? userGguf = null;
string? userJsonl = null;

for (int i = 0; i < args.Length; i++)
{
    if ((args[i] == "--device" || args[i] == "-d") && i + 1 < args.Length)
    {
        device = args[++i];
    }
    else if ((args[i] == "--steps" || args[i] == "-s" || args[i] == "--max-steps") && i + 1 < args.Length)
    {
        maxSteps = int.Parse(args[++i]);
    }
    else if (args[i] == "--gpu" || args[i] == "-g")
    {
        device = "gpu";
    }
    else if (args[i] == "--cpu" || args[i] == "-c")
    {
        device = "cpu";
    }
    else if (!args[i].StartsWith('-'))
    {
        if (userGguf == null) userGguf = args[i];
        else if (userJsonl == null) userJsonl = args[i];
    }
}

string? ggufPath = userGguf ?? Environment.GetEnvironmentVariable("GLACIER_BASE_GGUF");
string trainJsonl = userJsonl ?? Environment.GetEnvironmentVariable("GLACIER_TUNE_DATASET") ?? FindDatasetPath();

if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("Usage: dotnet run --project samples/Glacier.Tune.Demo -- <base_model.gguf> [dataset.jsonl] [--gpu|--cpu] [--steps N]");
    Console.WriteLine("   or set GLACIER_BASE_GGUF=<path/to/base_model.gguf> and GLACIER_TUNE_DATASET=<path/to/dataset.jsonl>.");
    Console.ResetColor();
    return;
}

if (!File.Exists(trainJsonl))
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"[ERROR] Training dataset not found at {trainJsonl}");
    Console.ResetColor();
    return;
}

static string FindDatasetPath()
{
    string[] candidates =
    [
        Path.Combine(AppContext.BaseDirectory, "data", "enterprise_dev_train.jsonl"),
        Path.Combine(Directory.GetCurrentDirectory(), "data", "enterprise_dev_train.jsonl"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "enterprise_dev_train.jsonl"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "data", "enterprise_dev_train.jsonl")
    ];
    foreach (var path in candidates)
    {
        if (File.Exists(path)) return Path.GetFullPath(path);
    }
    return candidates[0];
}

// 1. Load GGUF Model via Zero-Copy Memory Mapping
Console.WriteLine($"[1/4] Loading Raw Untuned GGUF Base Model (Target Device: {device.ToUpperInvariant()})...");
var swMmf = Stopwatch.StartNew();
using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 16, Alpha = 32f }, device: device);
swMmf.Stop();

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  Model Loaded in {swMmf.Elapsed.TotalMilliseconds:F2} ms!");
Console.ResetColor();

var (trainable, total, pct) = model.GetParameterStats();
Console.WriteLine($"  Base Model:    {model.Gguf.Architecture} ({model.LayerCount} layers, {model.HiddenDim} dim, {model.FfnDim} ffn)");
Console.WriteLine($"  Total Params:  {total:N0} parameters [FROZEN in unmanaged memory]");
Console.WriteLine($"  LoRA Adapters: {trainable:N0} parameters [TRAINABLE: r=16, alpha=32]");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"  Parameter Efficiency: {100.0 - pct:F2}% parameter reduction!\n");
Console.ResetColor();

// 2. Ingest and Tokenize ChatML Dataset
Console.WriteLine("[2/4] Ingesting and Tokenizing ChatML Dataset...");
var swData = Stopwatch.StartNew();
var dataset = ChatMlDataset.FromFile(trainJsonl, model.Tokenizer, maxSeqLength: 160);
swData.Stop();

Console.WriteLine($"  Ingested {dataset.Count:N0} samples in {swData.Elapsed.TotalMilliseconds:F2} ms!");
if (dataset.Count > 0)
{
    Console.WriteLine($"  First Example Length: {dataset.Examples[0].InputIds.Length} tokens (Labels masked for prompt & assistant header tokens)\n");
}

// 3. Fine-Tuning Execution
Console.WriteLine("[3/4] Executing Fine-Tuning with Reverse-Mode Backpropagation & AdamW...");
string outputDir = Path.Combine(Directory.GetCurrentDirectory(), "tune_output");
var trainingArgs = new TrainingArguments
{
    LearningRate = 5e-4f,
    Epochs = 1,
    BatchSize = 1,
    GradientAccumulationSteps = 2,
    MaxSteps = maxSteps,
    LoggingSteps = 1,
    Device = device,
    OutputDir = outputDir
};

using var trainer = new LoraTrainer(model, trainingArgs);
trainer.Train(dataset);

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n[SUCCESS] Trained LoRA Adapter exported to: {outputDir}");
Console.ResetColor();
