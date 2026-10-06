namespace Glacier.Tune.Benchmarks;

using System;
using BenchmarkDotNet.Running;

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

            Console.WriteLine("\n[SUCCESS] All microbenchmarks executed successfully.");
        }
    }
}
