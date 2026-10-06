namespace Glacier.Tune.Benchmarks;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Tune.Kernels;

[MemoryDiagnoser]
public unsafe class SwiGluKernelBenchmarks : IDisposable
{
    private const int FfnDim = 18944;

    [Params(128, 512)]
    public int SeqLen { get; set; } = 128;

    private int TotalElements => SeqLen * FfnDim;

    private float* _pGate;
    private float* _pUp;
    private float* _pHidden;
    private float* _pDHidden;
    private float* _pDGate;
    private float* _pDUp;

    [GlobalSetup]
    public void Setup()
    {
        int length = TotalElements;
        _pGate = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));
        _pUp = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));
        _pHidden = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));
        _pDHidden = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));
        _pDGate = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));
        _pDUp = (float*)NativeMemory.Alloc((nuint)(length * sizeof(float)));

        for (int i = 0; i < length; i++)
        {
            _pGate[i] = 0.5f;
            _pUp[i] = 0.25f;
            _pDHidden[i] = 0.01f;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_pGate != null) { NativeMemory.Free(_pGate); _pGate = null; }
        if (_pUp != null) { NativeMemory.Free(_pUp); _pUp = null; }
        if (_pHidden != null) { NativeMemory.Free(_pHidden); _pHidden = null; }
        if (_pDHidden != null) { NativeMemory.Free(_pDHidden); _pDHidden = null; }
        if (_pDGate != null) { NativeMemory.Free(_pDGate); _pDGate = null; }
        if (_pDUp != null) { NativeMemory.Free(_pDUp); _pDUp = null; }
        GC.SuppressFinalize(this);
    }

    [Benchmark]
    public void Forward()
    {
        SwiGluKernel.Forward(_pGate, _pUp, _pHidden, TotalElements);
    }

    [Benchmark]
    public void Backward()
    {
        SwiGluKernel.Backward(_pDHidden, _pGate, _pUp, _pDGate, _pDUp, TotalElements);
    }

    public static void RunConsole()
    {
        Console.WriteLine("\n--- SwiGluKernel Microbenchmarks ---");
        int[] seqLens = [128, 512];

        foreach (var seq in seqLens)
        {
            using var bench = new SwiGluKernelBenchmarks { SeqLen = seq };
            bench.Setup();

            // Warmup
            bench.Forward();
            bench.Backward();

            int iterations = 20;
            var swFwd = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                bench.Forward();
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
            double mElements = (seq * FfnDim) / 1e6;

            Console.WriteLine($"[SwiGLU T={seq}, Dffn={FfnDim} ({mElements:F2}M elem)] Fwd: {avgFwdMs:F3} ms | Bwd: {avgBwdMs:F3} ms");
        }
    }
}
