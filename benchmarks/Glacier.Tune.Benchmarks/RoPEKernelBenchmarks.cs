namespace Glacier.Tune.Benchmarks;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Tune.Kernels;

[MemoryDiagnoser]
public unsafe class RoPEKernelBenchmarks : IDisposable
{
    private const int NHeadsQ = 28;
    private const int NHeadsKv = 4;
    private const int HeadDim = 128;
    private const float FreqBase = 10000.0f;

    [Params(128, 512)]
    public int SeqLen { get; set; } = 128;

    private float* _pQ;
    private float* _pK;
    private float* _pDq;
    private float* _pDk;

    [GlobalSetup]
    public void Setup()
    {
        int qDim = NHeadsQ * HeadDim;
        int kvDim = NHeadsKv * HeadDim;

        _pQ = (float*)NativeMemory.Alloc((nuint)(SeqLen * qDim * sizeof(float)));
        _pK = (float*)NativeMemory.Alloc((nuint)(SeqLen * kvDim * sizeof(float)));
        _pDq = (float*)NativeMemory.Alloc((nuint)(SeqLen * qDim * sizeof(float)));
        _pDk = (float*)NativeMemory.Alloc((nuint)(SeqLen * kvDim * sizeof(float)));

        for (int i = 0; i < SeqLen * qDim; i++) { _pQ[i] = 0.05f; _pDq[i] = 0.01f; }
        for (int i = 0; i < SeqLen * kvDim; i++) { _pK[i] = 0.05f; _pDk[i] = 0.01f; }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_pQ != null) { NativeMemory.Free(_pQ); _pQ = null; }
        if (_pK != null) { NativeMemory.Free(_pK); _pK = null; }
        if (_pDq != null) { NativeMemory.Free(_pDq); _pDq = null; }
        if (_pDk != null) { NativeMemory.Free(_pDk); _pDk = null; }
        GC.SuppressFinalize(this);
    }

    [Benchmark]
    public void ForwardSequence()
    {
        RoPEKernel.ForwardSequence(_pQ, _pK, SeqLen, NHeadsQ, NHeadsKv, HeadDim, FreqBase);
    }

    [Benchmark]
    public void BackwardSequence()
    {
        RoPEKernel.BackwardSequence(_pDq, _pDk, SeqLen, NHeadsQ, NHeadsKv, HeadDim, FreqBase);
    }

    public static void RunConsole()
    {
        Console.WriteLine("\n--- RoPEKernel Microbenchmarks ---");
        int[] seqLens = [128, 512];

        foreach (var seq in seqLens)
        {
            using var bench = new RoPEKernelBenchmarks { SeqLen = seq };
            bench.Setup();

            // Warmup
            bench.ForwardSequence();
            bench.BackwardSequence();

            int iterations = 100;
            var swFwd = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                bench.ForwardSequence();
            }
            swFwd.Stop();

            var swBwd = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                bench.BackwardSequence();
            }
            swBwd.Stop();

            double avgFwdUs = (swFwd.Elapsed.TotalMilliseconds * 1000.0) / iterations;
            double avgBwdUs = (swBwd.Elapsed.TotalMilliseconds * 1000.0) / iterations;

            Console.WriteLine($"[RoPE T={seq}, Hq={NHeadsQ}, Hkv={NHeadsKv}, D={HeadDim}] Fwd: {avgFwdUs:F1} us | Bwd: {avgBwdUs:F1} us");
        }
    }
}
