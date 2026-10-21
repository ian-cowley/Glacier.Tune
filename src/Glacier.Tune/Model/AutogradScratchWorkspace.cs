namespace Glacier.Tune.Model;

using System;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Tensor.Core;

/// <summary>
/// Pre-allocated unmanaged autograd scratch workspace shared across all transformer layers.
/// Eliminates all NativeMemory.Alloc and NativeMemory.Free invocations during training steps.
/// </summary>
public sealed unsafe class AutogradScratchWorkspace : IDisposable
{
    private static readonly ThreadLocal<AutogradScratchWorkspace?> s_threadLocalWorkspace = new(() => null);

    public float* FfnScratchA { get; private set; }
    public float* FfnScratchB { get; private set; }
    public float* FfnScratchC { get; private set; }

    public float* HiddenScratchA { get; private set; }
    public float* HiddenScratchB { get; private set; }
    public float* HiddenScratchC { get; private set; }
    public float* HiddenScratchD { get; private set; }
    public float* HiddenScratchE { get; private set; }
    public float* HiddenScratchF { get; private set; }

    public float* QDimScratchA { get; private set; }
    public float* QDimScratchB { get; private set; }
    public float* KvDimScratchA { get; private set; }
    public float* KvDimScratchB { get; private set; }

    private int _maxSeqLen;
    private int _hiddenDim;
    private int _ffnDim;
    private int _qDim;
    private int _kvDim;
    private bool _disposed;

    public static AutogradScratchWorkspace GetOrCreate(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        var ws = s_threadLocalWorkspace.Value;
        if (ws == null || ws._disposed)
        {
            ws = new AutogradScratchWorkspace(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            s_threadLocalWorkspace.Value = ws;
        }
        else
        {
            ws.EnsureCapacity(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
        }
        return ws;
    }

    public AutogradScratchWorkspace(int maxSeqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        Allocate(maxSeqLen, hiddenDim, ffnDim, nHeadsQ * headDim, nHeadsKv * headDim);
    }

    public void EnsureCapacity(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;

        if (seqLen > _maxSeqLen || hiddenDim > _hiddenDim || ffnDim > _ffnDim || qDim > _qDim || kvDim > _kvDim)
        {
            FreeAll();
            int newSeqLen = Math.Max(seqLen, _maxSeqLen);
            int newHiddenDim = Math.Max(hiddenDim, _hiddenDim);
            int newFfnDim = Math.Max(ffnDim, _ffnDim);
            int newQDim = Math.Max(qDim, _qDim);
            int newKvDim = Math.Max(kvDim, _kvDim);
            Allocate(newSeqLen, newHiddenDim, newFfnDim, newQDim, newKvDim);
        }
    }

    private void Allocate(int seqLen, int hiddenDim, int ffnDim, int qDim, int kvDim)
    {
        _maxSeqLen = seqLen;
        _hiddenDim = hiddenDim;
        _ffnDim = ffnDim;
        _qDim = qDim;
        _kvDim = kvDim;

        nuint bytesFfn = (nuint)seqLen * (nuint)ffnDim * sizeof(float);
        nuint bytesHidden = (nuint)seqLen * (nuint)hiddenDim * sizeof(float);
        nuint bytesQ = (nuint)seqLen * (nuint)qDim * sizeof(float);
        nuint bytesKv = (nuint)seqLen * (nuint)kvDim * sizeof(float);

        FfnScratchA = (float*)NativeMemory.AlignedAlloc(bytesFfn, 64);
        FfnScratchB = (float*)NativeMemory.AlignedAlloc(bytesFfn, 64);
        FfnScratchC = (float*)NativeMemory.AlignedAlloc(bytesFfn, 64);

        HiddenScratchA = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);
        HiddenScratchB = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);
        HiddenScratchC = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);
        HiddenScratchD = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);
        HiddenScratchE = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);
        HiddenScratchF = (float*)NativeMemory.AlignedAlloc(bytesHidden, 64);

        QDimScratchA = (float*)NativeMemory.AlignedAlloc(bytesQ, 64);
        QDimScratchB = (float*)NativeMemory.AlignedAlloc(bytesQ, 64);
        KvDimScratchA = (float*)NativeMemory.AlignedAlloc(bytesKv, 64);
        KvDimScratchB = (float*)NativeMemory.AlignedAlloc(bytesKv, 64);
    }

    private void FreeAll()
    {
        if (FfnScratchA != null) { NativeMemory.AlignedFree(FfnScratchA); FfnScratchA = null; }
        if (FfnScratchB != null) { NativeMemory.AlignedFree(FfnScratchB); FfnScratchB = null; }
        if (FfnScratchC != null) { NativeMemory.AlignedFree(FfnScratchC); FfnScratchC = null; }

        if (HiddenScratchA != null) { NativeMemory.AlignedFree(HiddenScratchA); HiddenScratchA = null; }
        if (HiddenScratchB != null) { NativeMemory.AlignedFree(HiddenScratchB); HiddenScratchB = null; }
        if (HiddenScratchC != null) { NativeMemory.AlignedFree(HiddenScratchC); HiddenScratchC = null; }
        if (HiddenScratchD != null) { NativeMemory.AlignedFree(HiddenScratchD); HiddenScratchD = null; }
        if (HiddenScratchE != null) { NativeMemory.AlignedFree(HiddenScratchE); HiddenScratchE = null; }
        if (HiddenScratchF != null) { NativeMemory.AlignedFree(HiddenScratchF); HiddenScratchF = null; }

        if (QDimScratchA != null) { NativeMemory.AlignedFree(QDimScratchA); QDimScratchA = null; }
        if (QDimScratchB != null) { NativeMemory.AlignedFree(QDimScratchB); QDimScratchB = null; }
        if (KvDimScratchA != null) { NativeMemory.AlignedFree(KvDimScratchA); KvDimScratchA = null; }
        if (KvDimScratchB != null) { NativeMemory.AlignedFree(KvDimScratchB); KvDimScratchB = null; }
    }

    public static Tensor<float> Wrap(float* ptr, params ReadOnlySpan<int> shape)
    {
        var s = new Shape8(shape);
        var strides = Shape8.ComputeContiguousStrides(s);
        var block = new NativeMemoryBlock<float>(ptr, s.ElementCount, ownsMemory: false);
        return new Tensor<float>(block, s, strides, 0);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            FreeAll();
        }
    }
}
