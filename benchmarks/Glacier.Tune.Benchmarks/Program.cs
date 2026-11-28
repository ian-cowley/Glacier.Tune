namespace Glacier.Tune.Benchmarks;

using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.Tune.Sleep;
using Glacier.Tune.Trainer;

public static class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("================================================================================");
        Console.WriteLine("              GLACIER.TUNE HIGH-PERFORMANCE TRAINING BENCHMARKS                 ");
        Console.WriteLine("================================================================================");

        if (args.Length > 0 && Array.Exists(args, a => a.Equals("--bdn", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine("Launching BenchmarkDotNet suite...\n");
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
        }
        else
        {
            Console.WriteLine("Running fast console microbenchmarks (pass --bdn for BenchmarkDotNet execution)...\n");

            LoraKernelBenchmarks.RunConsole();
            FusedCrossEntropyLossBenchmarks.RunConsole();
            RoPEKernelBenchmarks.RunConsole();
            SwiGluKernelBenchmarks.RunConsole();
            RunDpoAndSleepConsoleBenchmarks();

            Console.WriteLine("\n[SUCCESS] All microbenchmarks executed successfully.");
        }
    }

    private static void RunDpoAndSleepConsoleBenchmarks()
    {
        Console.WriteLine("\n--- DpoLoss & DreamQualityFilter Microbenchmarks ---");

        // Warmup DpoLoss
        double acc = 0;
        for (int i = 0; i < 10_000; i++)
        {
            var r = DpoLoss.Compute(-12.4, -15.1, -13.0, -14.8, beta: 0.1, labelSmoothing: 0.05);
            acc += r.Loss;
        }

        const int dpoIters = 1_000_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < dpoIters; i++)
        {
            var r = DpoLoss.Compute(-12.4 - (i & 3) * 0.1, -15.1, -13.0, -14.8, beta: 0.1, labelSmoothing: 0.05);
            acc += r.Loss;
        }
        sw.Stop();
        Console.WriteLine($"[DpoLoss.Compute (1M calls)]        Latency: {sw.Elapsed.TotalNanoseconds / dpoIters:F2} ns / call (chk={acc:F0})");

        const int dim = 1024;
        float[] embA = new float[dim];
        float[] embB = new float[dim];
        var rng = new Random(42);
        for (int i = 0; i < dim; i++)
        {
            embA[i] = (float)rng.NextDouble();
            embB[i] = (float)rng.NextDouble();
        }
        ReadOnlySpan<char> txtA = "The binary search algorithm halves the sorted interval each step.";
        ReadOnlySpan<char> txtB = "Binary search halves the sorted search interval at every step.";

        // Warmup
        float scoreSum = 0;
        for (int i = 0; i < 1000; i++)
        {
            scoreSum += DreamQualityFilter.ComputeScore(txtA, txtB, embA, embB);
        }

        const int cosIters = 200_000;
        sw.Restart();
        for (int i = 0; i < cosIters; i++)
        {
            scoreSum += DreamQualityFilter.CosineSimilarity(embA, embB);
        }
        sw.Stop();
        Console.WriteLine($"[DreamQualityFilter.Cosine D={dim}] Latency: {sw.Elapsed.TotalNanoseconds / cosIters:F2} ns / call");

        const int gateIters = 20_000;
        sw.Restart();
        for (int i = 0; i < gateIters; i++)
        {
            scoreSum += DreamQualityFilter.ComputeScore(txtA, txtB, embA, embB);
        }
        sw.Stop();
        Console.WriteLine($"[DreamQualityFilter.ComputeScore]    Latency: {sw.Elapsed.TotalMicroseconds / gateIters:F3} us / call (chk={scoreSum:F0})");
    }
}
