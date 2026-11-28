namespace Glacier.Tune.Model;

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Tensor.Core;

#if GLACIER_TENSOR_LOCAL
/// <summary>
/// Pre-allocated unmanaged autograd scratch workspace shared across all transformer layers.
/// Delegates directly to M3.1 <see cref="Glacier.Tensor.Autograd.AutogradScratchWorkspace"/>.
/// </summary>
public sealed unsafe class AutogradScratchWorkspace : IDisposable
{
    [ThreadStatic]
    private static AutogradScratchWorkspace? t_workspace;

    private static readonly ConcurrentBag<WeakReference<AutogradScratchWorkspace>> s_instances = new();
    private Glacier.Tensor.Autograd.AutogradScratchWorkspace? _inner;
    private int _disposed;

    public float* FfnScratchA => _inner != null ? _inner.FfnScratchA : null;
    public float* FfnScratchB => _inner != null ? _inner.FfnScratchB : null;
    public float* FfnScratchC => _inner != null ? _inner.FfnScratchC : null;

    public float* HiddenScratchA => _inner != null ? _inner.HiddenScratchA : null;
    public float* HiddenScratchB => _inner != null ? _inner.HiddenScratchB : null;
    public float* HiddenScratchC => _inner != null ? _inner.HiddenScratchC : null;
    public float* HiddenScratchD => _inner != null ? _inner.HiddenScratchD : null;
    public float* HiddenScratchE => _inner != null ? _inner.HiddenScratchE : null;
    public float* HiddenScratchF => _inner != null ? _inner.HiddenScratchF : null;

    public float* QDimScratchA => _inner != null ? _inner.QDimScratchA : null;
    public float* QDimScratchB => _inner != null ? _inner.QDimScratchB : null;
    public float* KvDimScratchA => _inner != null ? _inner.KvDimScratchA : null;
    public float* KvDimScratchB => _inner != null ? _inner.KvDimScratchB : null;

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0 || (_inner != null && _inner.IsDisposed);

    private bool HasValidBuffers =>
        _inner != null &&
        !_inner.IsDisposed &&
        _inner.FfnScratchA != null &&
        _inner.FfnScratchB != null &&
        _inner.FfnScratchC != null &&
        _inner.HiddenScratchA != null &&
        _inner.HiddenScratchB != null &&
        _inner.HiddenScratchC != null &&
        _inner.HiddenScratchD != null &&
        _inner.HiddenScratchE != null &&
        _inner.HiddenScratchF != null &&
        _inner.QDimScratchA != null &&
        _inner.QDimScratchB != null &&
        _inner.KvDimScratchA != null &&
        _inner.KvDimScratchB != null;

    public static AutogradScratchWorkspace GetOrCreate(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        var ws = t_workspace;
        if (ws == null || ws.IsDisposed || !ws.HasValidBuffers)
        {
            if (ws != null)
            {
                t_workspace = null;
                ws.Dispose();
            }
            ws = new AutogradScratchWorkspace(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            t_workspace = ws;
        }
        else
        {
            try
            {
                ws.EnsureCapacity(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            }
            catch
            {
                if (!ws.HasValidBuffers || ws.IsDisposed)
                {
                    t_workspace = null;
                    ws.Dispose();
                }
                throw;
            }
        }
        return ws;
    }

    public static void DisposeAllThreadLocal()
    {
        while (s_instances.TryTake(out var weak))
        {
            if (weak.TryGetTarget(out var ws) && !ws.IsDisposed)
            {
                ws.Dispose();
            }
        }
        Glacier.Tensor.Autograd.AutogradScratchWorkspace.DisposeAllThreadLocal();
    }

    public AutogradScratchWorkspace(int maxSeqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        _inner = new Glacier.Tensor.Autograd.AutogradScratchWorkspace(maxSeqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
        s_instances.Add(new WeakReference<AutogradScratchWorkspace>(this));
    }

    public void EnsureCapacity(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ffnDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsQ);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsKv);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        _ = checked(nHeadsQ * headDim);
        _ = checked(nHeadsKv * headDim);

        if (!HasValidBuffers)
        {
            _inner?.Dispose();
            _inner = null;
            _inner = new Glacier.Tensor.Autograd.AutogradScratchWorkspace(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            return;
        }

        try
        {
            _inner!.EnsureCapacity(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
        }
        catch
        {
            if (!HasValidBuffers)
            {
                // Upstream Glacier.Tensor.Autograd.AutogradScratchWorkspace sets _maxSeqLen.._kvDim before
                // NativeMemory.AlignedAlloc and nulls all pointers in FreeAll() on OOM. Dispose and clear _inner
                // so neither this wrapper nor subsequent EnsureCapacity calls remain poisoned.
                _inner?.Dispose();
                _inner = null;
            }
            throw;
        }
    }

    public static Tensor<float> Wrap(float* ptr, params ReadOnlySpan<int> shape)
    {
        if (ptr == null)
            throw new ArgumentNullException(nameof(ptr), "Scratch pointer cannot be null.");
        var s = new Shape8(shape);
        long count = s.ElementCount;
        if (s.Rank == 0 || count <= 0 || count > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(shape), "Scratch tensor element count must be positive and within Span length.");
        return Glacier.Tensor.Autograd.AutogradScratchWorkspace.Wrap(ptr, shape);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _inner?.Dispose();
            _inner = null;
        }
        GC.SuppressFinalize(this);
    }
}
#else
/// <summary>
/// Pre-allocated unmanaged autograd scratch workspace shared across all transformer layers.
/// Uses <c>[ThreadStatic]</c> storage with a deterministic finalizer and weak-reference registry so
/// unmanaged slabs are reclaimed when worker threads exit without pinning dead <see cref="Thread"/> objects.
/// </summary>
public sealed unsafe class AutogradScratchWorkspace : IDisposable
{
    [ThreadStatic]
    private static AutogradScratchWorkspace? t_workspace;

    private static readonly ConcurrentBag<WeakReference<AutogradScratchWorkspace>> s_instances = new();

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
    private int _disposed;

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    private bool HasValidBuffers =>
        FfnScratchA != null && FfnScratchB != null && FfnScratchC != null &&
        HiddenScratchA != null && HiddenScratchB != null && HiddenScratchC != null &&
        HiddenScratchD != null && HiddenScratchE != null && HiddenScratchF != null &&
        QDimScratchA != null && QDimScratchB != null &&
        KvDimScratchA != null && KvDimScratchB != null;

    public static AutogradScratchWorkspace GetOrCreate(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        var ws = t_workspace;
        if (ws == null || ws.IsDisposed || !ws.HasValidBuffers)
        {
            if (ws != null)
            {
                t_workspace = null;
                ws.Dispose();
            }
            ws = new AutogradScratchWorkspace(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            t_workspace = ws;
        }
        else
        {
            try
            {
                ws.EnsureCapacity(seqLen, hiddenDim, ffnDim, nHeadsQ, nHeadsKv, headDim);
            }
            catch
            {
                if (!ws.HasValidBuffers || ws.IsDisposed)
                {
                    t_workspace = null;
                    ws.Dispose();
                }
                throw;
            }
        }
        return ws;
    }

    public static void DisposeAllThreadLocal()
    {
        while (s_instances.TryTake(out var weak))
        {
            if (weak.TryGetTarget(out var ws) && !ws.IsDisposed)
            {
                ws.Dispose();
            }
        }
    }

    public AutogradScratchWorkspace(int maxSeqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ffnDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsQ);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsKv);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        int qDim = checked(nHeadsQ * headDim);
        int kvDim = checked(nHeadsKv * headDim);
        Allocate(maxSeqLen, hiddenDim, ffnDim, qDim, kvDim);
        s_instances.Add(new WeakReference<AutogradScratchWorkspace>(this));
    }

    ~AutogradScratchWorkspace()
    {
        DisposeCore();
    }

    public void EnsureCapacity(int seqLen, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int headDim)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ffnDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsQ);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsKv);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        int qDim = checked(nHeadsQ * headDim);
        int kvDim = checked(nHeadsKv * headDim);

        if (seqLen > _maxSeqLen || hiddenDim > _hiddenDim || ffnDim > _ffnDim || qDim > _qDim || kvDim > _kvDim || !HasValidBuffers)
        {
            int newSeqLen = Math.Max(seqLen, _maxSeqLen);
            int newHiddenDim = Math.Max(hiddenDim, _hiddenDim);
            int newFfnDim = Math.Max(ffnDim, _ffnDim);
            int newQDim = Math.Max(qDim, _qDim);
            int newKvDim = Math.Max(kvDim, _kvDim);
            FreeAll();
            Allocate(newSeqLen, newHiddenDim, newFfnDim, newQDim, newKvDim);
        }
    }

    private void Allocate(int seqLen, int hiddenDim, int ffnDim, int qDim, int kvDim)
    {
        nuint bytesFfn = checked((nuint)seqLen * (nuint)ffnDim * sizeof(float));
        nuint bytesHidden = checked((nuint)seqLen * (nuint)hiddenDim * sizeof(float));
        nuint bytesQ = checked((nuint)seqLen * (nuint)qDim * sizeof(float));
        nuint bytesKv = checked((nuint)seqLen * (nuint)kvDim * sizeof(float));

        try
        {
            FfnScratchA = AllocCleared(bytesFfn);
            FfnScratchB = AllocCleared(bytesFfn);
            FfnScratchC = AllocCleared(bytesFfn);

            HiddenScratchA = AllocCleared(bytesHidden);
            HiddenScratchB = AllocCleared(bytesHidden);
            HiddenScratchC = AllocCleared(bytesHidden);
            HiddenScratchD = AllocCleared(bytesHidden);
            HiddenScratchE = AllocCleared(bytesHidden);
            HiddenScratchF = AllocCleared(bytesHidden);

            QDimScratchA = AllocCleared(bytesQ);
            QDimScratchB = AllocCleared(bytesQ);
            KvDimScratchA = AllocCleared(bytesKv);
            KvDimScratchB = AllocCleared(bytesKv);

            _maxSeqLen = seqLen;
            _hiddenDim = hiddenDim;
            _ffnDim = ffnDim;
            _qDim = qDim;
            _kvDim = kvDim;
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    private static float* AllocCleared(nuint bytes)
    {
        float* ptr = (float*)NativeMemory.AlignedAlloc(bytes, 64);
        NativeMemory.Clear(ptr, bytes);
        return ptr;
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

        _maxSeqLen = 0;
        _hiddenDim = 0;
        _ffnDim = 0;
        _qDim = 0;
        _kvDim = 0;
    }

    public static Tensor<float> Wrap(float* ptr, params ReadOnlySpan<int> shape)
    {
        if (ptr == null)
            throw new ArgumentNullException(nameof(ptr), "Scratch pointer cannot be null.");
        var s = new Shape8(shape);
        long count = s.ElementCount;
        if (s.Rank == 0 || count <= 0 || count > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(shape), "Scratch tensor element count must be positive and within Span length.");
        new Span<float>(ptr, (int)count).Clear();
        var strides = Shape8.ComputeContiguousStrides(s);
        var block = new NativeMemoryBlock<float>(ptr, count, ownsMemory: false);
        return new Tensor<float>(block, s, strides, 0);
    }

    private void DisposeCore()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            FreeAll();
        }
    }

    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }
}
#endif
