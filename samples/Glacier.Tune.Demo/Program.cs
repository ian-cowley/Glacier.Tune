using System;
using System.Diagnostics;
using System.IO;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Export;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Glacier.Tune.Diagnostics;

if (args.Length > 0 && (args[0] == "--merge" || args[0] == "-m"))
{
    string baseModel = args.Length > 1 ? args[1] : @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
    string adapterPath = args.Length > 2 ? args[2] : @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Tune\tune_output\adapter_model.bin";
    string outModel = args.Length > 3 ? args[3] : @"D:\lmstudio\models\Local\Qwen2.5-Coder-7B-Enterprise-GGUF\qwen2.5-7b-glacier-enterprise-q8_0.gguf";

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("================================================================================");
    Console.WriteLine("         GLACIER.TUNE: PURE C# .NET 10 GGUF STANDALONE ADAPTER MERGER           ");
    Console.WriteLine("        Zero Python - Direct Native Memory SIMD Quantized Fusion                ");
    Console.WriteLine("================================================================================\\n");
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

GlacierDiagnostics.Logger = new ConsoleGlacierLogger(LogLevel.Information);

// Determine device target: --device <dev>, --gpu, --cpu, or auto
string device = "auto";
int maxSteps = 3;
int epochs = 1;
int batchSize = 1;
int gradientAccumSteps = 1;
int maxSeqLen = 1024;
float learningRate = 2e-4f;
int rank = 16;
float alpha = 32f;
string? userGguf = null;
string? userJsonl = null;
string? userOutputDir = null;

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
    else if ((args[i] == "--epochs" || args[i] == "-e") && i + 1 < args.Length)
    {
        epochs = int.Parse(args[++i]);
    }
    else if ((args[i] == "--lr" || args[i] == "--learning-rate") && i + 1 < args.Length)
    {
        learningRate = float.Parse(args[++i]);
    }
    else if ((args[i] == "--accum" || args[i] == "-a" || args[i] == "--gradient-accumulation-steps") && i + 1 < args.Length)
    {
        gradientAccumSteps = int.Parse(args[++i]);
    }
    else if ((args[i] == "--batch-size" || args[i] == "-b") && i + 1 < args.Length)
    {
        batchSize = int.Parse(args[++i]);
    }
    else if ((args[i] == "--max-seq-len" || args[i] == "--seq-len") && i + 1 < args.Length)
    {
        maxSeqLen = int.Parse(args[++i]);
    }
    else if ((args[i] == "--rank" || args[i] == "-r") && i + 1 < args.Length)
    {
        rank = int.Parse(args[++i]);
    }
    else if (args[i] == "--alpha" && i + 1 < args.Length)
    {
        alpha = float.Parse(args[++i]);
    }
    else if ((args[i] == "--output" || args[i] == "-o" || args[i] == "--output-dir") && i + 1 < args.Length)
    {
        userOutputDir = args[++i];
    }
    else if ((args[i] == "--model" || args[i] == "--gguf") && i + 1 < args.Length)
    {
        userGguf = args[++i];
    }
    else if ((args[i] == "--dataset" || args[i] == "--data" || args[i] == "--jsonl") && i + 1 < args.Length)
    {
        userJsonl = args[++i];
    }
    else if (args[i] == "--gpu" || args[i] == "-g")
    {
        device = "gpu";
    }
    else if (args[i] == "--cpu" || args[i] == "-c")
    {
        device = "cpu";
    }
    else if (!args[i].StartsWith("-"))
    {
        if (userGguf == null) userGguf = args[i];
        else if (userJsonl == null) userJsonl = args[i];
    }
}

// Path to raw untuned GGUF base model and enterprise dataset
string ggufPath = userGguf ?? @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
string trainJsonl = userJsonl ?? FindDatasetPath();

if (!File.Exists(ggufPath))
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"[ERROR] Raw base GGUF model not found at {ggufPath}");
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
    string[] candidates = [
        Path.Combine(AppContext.BaseDirectory, "data", "enterprise_dev_train.jsonl"),
        Path.Combine(Directory.GetCurrentDirectory(), "data", "enterprise_dev_train.jsonl"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "enterprise_dev_train.jsonl"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "data", "enterprise_dev_train.jsonl"),
        @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Tune\data\enterprise_dev_train.jsonl",
        @"C:\Users\spuri\source\repos\modelTrain\data\enterprise_dev_train.jsonl"
    ];
    foreach (var path in candidates)
    {
        if (File.Exists(path)) return Path.GetFullPath(path);
    }
    return candidates[0];
}

// 1. Load GGUF Model via Zero-Copy Memory Mapping
Console.WriteLine($"[1/4] Loading Raw Untuned Base Model: {Path.GetFileName(ggufPath)} (Target Device: {device.ToUpperInvariant()})...");
var swMmf = Stopwatch.StartNew();
using var model = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = rank, Alpha = alpha }, device: device);
swMmf.Stop();

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  Model Loaded in {swMmf.Elapsed.TotalMilliseconds:F2} ms (Virtual Address Mapping)!");
Console.ResetColor();

var (trainable, total, pct) = model.GetParameterStats();
Console.WriteLine($"  Base Model:    {model.Gguf.Architecture} ({model.LayerCount} layers, {model.HiddenDim} dim, {model.FfnDim} ffn)");
Console.WriteLine($"  Total Params:  {total:N0} parameters [FROZEN in unmanaged memory]");
Console.WriteLine($"  LoRA Adapters: {trainable:N0} parameters [TRAINABLE: r={rank}, alpha={alpha}]");
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"  Parameter Efficiency: {100.0 - pct:F2}% parameter reduction!\n");
Console.ResetColor();

// 2. Ingest and Tokenize ChatML Dataset
Console.WriteLine("[2/4] Ingesting and Tokenizing ChatML Dataset...");
var swData = Stopwatch.StartNew();
var dataset = ChatMlDataset.FromFile(trainJsonl, model.Tokenizer, maxSeqLength: maxSeqLen);
swData.Stop();

Console.WriteLine($"  Ingested {dataset.Count:N0} samples in {swData.Elapsed.TotalMilliseconds:F2} ms!");
Console.WriteLine($"  First Example Length: {dataset.Examples[0].InputIds.Length} tokens (Labels masked for prompt tokens)\n");

// 3. Fine-Tuning Execution
Console.WriteLine("[3/4] Executing Fine-Tuning with AutogradTape & AdamW...");
string outputDir = userOutputDir ?? Path.Combine(AppContext.BaseDirectory, "tune_output");
var trainingArgs = new TrainingArguments
{
    LearningRate = learningRate,
    Epochs = epochs,
    BatchSize = batchSize,
    GradientAccumulationSteps = gradientAccumSteps,
    MaxSteps = maxSteps,
    LoggingSteps = 1,
    Device = device,
    OutputDir = outputDir,
    MaxSeqLength = maxSeqLen
};

using var trainer = new LoraTrainer(model, trainingArgs);
trainer.Train(dataset);

// Also copy adapter to convenient locations (repo root tune_output and Glacier.Tune/tune_output)
string tuneRepoOutputDir = Path.Combine(@"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Tune", "tune_output");
string polarsPlusOutputDir = Path.Combine(@"C:\Users\spuri\source\repos\PolarsPlus", "tune_output");

Directory.CreateDirectory(tuneRepoOutputDir);
Directory.CreateDirectory(polarsPlusOutputDir);

foreach (var file in Directory.GetFiles(outputDir))
{
    string fname = Path.GetFileName(file);
    File.Copy(file, Path.Combine(tuneRepoOutputDir, fname), overwrite: true);
    File.Copy(file, Path.Combine(polarsPlusOutputDir, fname), overwrite: true);
}

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n[SUCCESS] Trained LoRA Adapter exported to:");
Console.WriteLine($"  1. {outputDir}");
Console.WriteLine($"  2. {tuneRepoOutputDir}");
Console.WriteLine($"  3. {polarsPlusOutputDir}");
Console.ResetColor();

// 4. Standalone GGUF Model Weight Fusion
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("\n[4/4] Fusing LoRA Adapter into Standalone GGUF Model with GgufMerger...");
Console.ResetColor();

string adapterBin = Path.Combine(outputDir, "adapter_model.bin");
string fusedGgufPath = Path.Combine(@"C:\Users\spuri\source\repos\PolarsPlus", "Glacier-Grandmaster-7B-Q4_K_M.gguf");

var mergeResult = GgufMerger.Merge(
    ggufPath,
    adapterBin,
    fusedGgufPath,
    new GgufMerger.MergeOptions
    {
        TargetQuantization = TargetQuantization.Q4_K
    });

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n[SUCCESS] Standalone Fused GGUF Model Created: {fusedGgufPath}");
Console.WriteLine($"  File Size: {mergeResult.OutputSizeBytes:N0} bytes ({mergeResult.OutputSizeBytes / (1024.0 * 1024.0 * 1024.0):F2} GB)");
Console.WriteLine($"  Adapted Tensors: {mergeResult.AdaptedTensorCount}");
Console.WriteLine($"  Fusion Elapsed: {mergeResult.Elapsed.TotalSeconds:F2} seconds\n");
Console.ResetColor();
