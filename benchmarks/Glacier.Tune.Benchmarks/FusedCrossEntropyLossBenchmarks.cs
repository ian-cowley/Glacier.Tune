namespace Glacier.Tune.Benchmarks;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Inference.Gguf;
using Glacier.Tune.Kernels;

[MemoryDiagnoser]
public unsafe class FusedCrossEntropyLossBenchmarks : IDisposable
{
    private const int HiddenDim = 3584;
    private const int SeqLen = 128;

    [Params(1024, 4096, 32000)]
    public int VocabSize { get; set; } = 4096;

    private float* _pHidden;
    private float* _pHead;
    private float* _pDHidden;
    private int[] _inputTokens = null!;
    private int[] _targetTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        _pHidden = (float*)NativeMemory.Alloc((nuint)(SeqLen * HiddenDim * sizeof(float)));
        _pHead = (float*)NativeMemory.Alloc((nuint)((long)VocabSize * HiddenDim * sizeof(float)));
        _pDHidden = (float*)NativeMemory.Alloc((nuint)(SeqLen * HiddenDim * sizeof(float)));

        for (int i = 0; i < SeqLen * HiddenDim; i++) _pHidden[i] = 0.01f;
        for (long i = 0; i < (long)VocabSize * HiddenDim; i++) _pHead[i] = 0.005f;

        _inputTokens = new int[SeqLen];
        _targetTokens = new int[SeqLen];
        var rng = new Random(42);
        for (int i = 0; i < SeqLen; i++)
        {
            _inputTokens[i] = rng.Next(0, VocabSize);
            _targetTokens[i] = rng.Next(0, VocabSize);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_pHidden != null) { NativeMemory.Free(_pHidden); _pHidden = null; }
        if (_pHead != null) { NativeMemory.Free(_pHead); _pHead = null; }
        if (_pDHidden != null) { NativeMemory.Free(_pDHidden); _pDHidden = null; }
        GC.SuppressFinalize(this);
    }

    [Benchmark]
    public float ComputeLossAndGradients()
    {
        return FusedCrossEntropyLoss.ComputeLossAndGradients(
            _pHidden,
            _inputTokens,
            _targetTokens,
            GgufType.F32,
            (byte*)_pHead,
            _pDHidden,
            SeqLen,
            HiddenDim,
            VocabSize);
    }

    public static void RunConsole()
    {
        Console.WriteLine("\n--- FusedCrossEntropyLoss Microbenchmarks ---");
        int[] vocabs = [1024, 4096, 32000];

        foreach (var vocab in vocabs)
        {
            using var bench = new FusedCrossEntropyLossBenchmarks { VocabSize = vocab };
            bench.Setup();

            // Warmup
            bench.ComputeLossAndGradients();

            int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                bench.ComputeLossAndGradients();
            }
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            Console.WriteLine($"[CrossEntropy T={SeqLen}, D={HiddenDim}, V={vocab,5}] Latency: {avgMs:F3} ms");
        }
    }
}
