namespace Glacier.Tune.Benchmarks;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Tune.Kernels;

[MemoryDiagnoser]
public unsafe class LoraKernelBenchmarks : IDisposable
{
    private const int InFeatures = 3584;
    private const int OutFeatures = 3584;
    private const float Scaling = 2.0f;

    [Params(128, 512)]
    public int SeqLen { get; set; } = 128;

    [Params(8, 16)]
    public int Rank { get; set; } = 16;

    private float* _pX;
    private float* _pA;
    private float* _pB;
    private float* _pOutput;
    private float* _pDY;
    private float* _pGradA;
    private float* _pGradB;
    private float* _pDXInput;

    [GlobalSetup]
    public void Setup()
    {
        _pX = (float*)NativeMemory.Alloc((nuint)(SeqLen * InFeatures * sizeof(float)));
        _pA = (float*)NativeMemory.Alloc((nuint)(InFeatures * Rank * sizeof(float)));
        _pB = (float*)NativeMemory.Alloc((nuint)(Rank * OutFeatures * sizeof(float)));
        _pOutput = (float*)NativeMemory.Alloc((nuint)(SeqLen * OutFeatures * sizeof(float)));

        _pDY = (float*)NativeMemory.Alloc((nuint)(SeqLen * OutFeatures * sizeof(float)));
        _pGradA = (float*)NativeMemory.Alloc((nuint)(InFeatures * Rank * sizeof(float)));
        _pGradB = (float*)NativeMemory.Alloc((nuint)(Rank * OutFeatures * sizeof(float)));
        _pDXInput = (float*)NativeMemory.Alloc((nuint)(SeqLen * InFeatures * sizeof(float)));

        // Initialize with non-zero dummy values
        for (int i = 0; i < SeqLen * InFeatures; i++) _pX[i] = 0.01f;
        for (int i = 0; i < InFeatures * Rank; i++) _pA[i] = 0.02f;
        for (int i = 0; i < Rank * OutFeatures; i++) _pB[i] = 0.03f;
        for (int i = 0; i < SeqLen * OutFeatures; i++) _pDY[i] = 0.04f;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_pX != null) { NativeMemory.Free(_pX); _pX = null; }
        if (_pA != null) { NativeMemory.Free(_pA); _pA = null; }
        if (_pB != null) { NativeMemory.Free(_pB); _pB = null; }
        if (_pOutput != null) { NativeMemory.Free(_pOutput); _pOutput = null; }
        if (_pDY != null) { NativeMemory.Free(_pDY); _pDY = null; }
        if (_pGradA != null) { NativeMemory.Free(_pGradA); _pGradA = null; }
        if (_pGradB != null) { NativeMemory.Free(_pGradB); _pGradB = null; }
        if (_pDXInput != null) { NativeMemory.Free(_pDXInput); _pDXInput = null; }
        GC.SuppressFinalize(this);
    }

    [Benchmark]
    public void ApplyLoraDelta()
    {
        LoraKernels.ApplyLoraDelta(_pX, _pA, _pB, _pOutput, SeqLen, InFeatures, OutFeatures, Rank, Scaling);
    }

    [Benchmark]
    public void Backward()
    {
        LoraKernels.Backward(_pX, _pDY, _pA, _pB, _pGradA, _pGradB, _pDXInput, SeqLen, InFeatures, OutFeatures, Rank, Scaling);
    }

    public static void RunConsole()
    {
        Console.WriteLine("--- LoraKernels Microbenchmarks ---");
        int[] seqLens = [128, 512];
        int[] ranks = [8, 16];

        foreach (var seq in seqLens)
        {
            foreach (var rank in ranks)
            {
                using var bench = new LoraKernelBenchmarks { SeqLen = seq, Rank = rank };
                bench.Setup();

                // Warmup
                bench.ApplyLoraDelta();
                bench.Backward();

                int iterations = 10;
                var swFwd = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    bench.ApplyLoraDelta();
                }
                swFwd.Stop();

                var swBwd = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    bench.Backward();
                }
                swBwd.Stop();

                double avgFwdMs = swFwd.Elapsed.TotalMilliseconds / iterations;
                double avgBwdMs = swBwd.Elapsed.TotalMilliseconds / iterations;

                Console.WriteLine($"[LoRA T={seq}, R={rank}] Fwd: {avgFwdMs:F3} ms | Bwd: {avgBwdMs:F3} ms");
            }
        }
    }
}
