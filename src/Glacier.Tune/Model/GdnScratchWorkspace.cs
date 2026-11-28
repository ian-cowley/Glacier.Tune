namespace Glacier.Tune.Model;

using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Pre-allocated 64-byte-aligned unmanaged scratch workspace for <see cref="GdnBlock"/> Forward and Backward (BPTT).
/// Eliminates all per-head <c>NativeMemory.Alloc</c>/<c>Free</c> calls inside <c>Parallel.For</c> and pools
/// intermediate backward activation slabs across GDN layers.
/// Uses <c>[ThreadStatic]</c> storage with a deterministic finalizer and weak-reference registry.
/// </summary>
public sealed unsafe class GdnScratchWorkspace : IDisposable
{
    [ThreadStatic]
    private static GdnScratchWorkspace? t_workspace;

    private static readonly ConcurrentBag<WeakReference<GdnScratchWorkspace>> s_instances = new();

    private float* _fwdHeadSlab;
    private nuint _fwdHeadSlabBytes;
    private int _fwdHeads;
    private int _fwdStateDim;
    private long _fwdHeadStrideFloats;

    private float* _bwdHeadSlab;
    private nuint _bwdHeadSlabBytes;
    private int _bwdHeads;
    private int _bwdStateDim;
    private int _bwdSeqLen;
    private long _bwdHeadStrideFloats;
    private long _bwdMatAligned;
    private long _bwdVecAligned;
    private long _bwdHistMatAligned;
    private long _bwdHistVecAligned;

    private float* _bwdActSlab;
    private nuint _bwdActSlabBytes;
    private long _bwdActInnerOffset;
    private long _bwdActQPerHeadOffset;
    private long _bwdActQkGroupOffset;
    private long _bwdActHeadsOffset;
    private long _bwdActConvOffset;

    private int _disposed;

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public static GdnScratchWorkspace GetOrCreateForForward(int ssmHeads, int stateDim)
    {
        var ws = t_workspace;
        if (ws == null || ws.IsDisposed)
        {
            ws = new GdnScratchWorkspace();
            t_workspace = ws;
        }
        ws.EnsureForwardCapacity(ssmHeads, stateDim);
        return ws;
    }

    public static GdnScratchWorkspace GetOrCreateForBackward(
        int ssmHeads,
        int stateDim,
        int seqLen,
        int ssmInnerSize,
        int qkGroupFloats,
        int gdnConvChannels)
    {
        var ws = t_workspace;
        if (ws == null || ws.IsDisposed)
        {
            ws = new GdnScratchWorkspace();
            t_workspace = ws;
        }
        ws.EnsureBackwardCapacity(ssmHeads, stateDim, seqLen, ssmInnerSize, qkGroupFloats, gdnConvChannels);
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

    public GdnScratchWorkspace()
    {
        s_instances.Add(new WeakReference<GdnScratchWorkspace>(this));
    }

    ~GdnScratchWorkspace()
    {
        DisposeCore();
    }

    [MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static long Align16(long floats) => (floats + 15) & ~15L;

    public void EnsureForwardCapacity(int ssmHeads, int stateDim)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ssmHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateDim);

        long matElems = checked((long)stateDim * stateDim);
        long matAligned = Align16(matElems);
        long vecAligned = Align16(stateDim);
        long headStride = checked(matAligned + vecAligned);
        nuint neededBytes = checked((nuint)ssmHeads * (nuint)headStride * sizeof(float));

        if (_fwdHeadSlab == null || neededBytes > _fwdHeadSlabBytes || ssmHeads > _fwdHeads || stateDim > _fwdStateDim)
        {
            if (_fwdHeadSlab != null)
            {
                NativeMemory.AlignedFree(_fwdHeadSlab);
                _fwdHeadSlab = null;
                _fwdHeadSlabBytes = 0;
            }

            _fwdHeadSlab = (float*)NativeMemory.AlignedAlloc(neededBytes, 64);
            NativeMemory.Clear(_fwdHeadSlab, neededBytes);
            _fwdHeadSlabBytes = neededBytes;
            _fwdHeads = ssmHeads;
            _fwdStateDim = stateDim;
            _fwdHeadStrideFloats = headStride;
        }
    }

    public void EnsureBackwardCapacity(
        int ssmHeads,
        int stateDim,
        int seqLen,
        int ssmInnerSize,
        int qkGroupFloats,
        int gdnConvChannels)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ssmHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ssmInnerSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(qkGroupFloats);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(gdnConvChannels);

        long matElems = checked((long)stateDim * stateDim);
        long matAligned = Align16(matElems);
        long vecAligned = Align16(stateDim);
        long histMatAligned = Align16(checked((long)seqLen * matElems));
        long histVecAligned = Align16(checked((long)seqLen * stateDim));

        // 1x histMat (sPrevHist), 2x histVec (mHist, deltaHist), 5x mat (sCur, sPrime, sMat, dS, dSTot), 1x vec (dDelta)
        long headStride = checked(histMatAligned + 2 * histVecAligned + 5 * matAligned + vecAligned);
        nuint headBytes = checked((nuint)ssmHeads * (nuint)headStride * sizeof(float));

        long innerElems = Align16(checked((long)seqLen * ssmInnerSize));
        long perHeadQkElems = Align16(checked((long)ssmHeads * seqLen * stateDim));
        long qkGroupElems = Align16(checked((long)seqLen * qkGroupFloats));
        long headScalarElems = Align16(checked((long)seqLen * ssmHeads));
        long convElems = Align16(checked((long)seqLen * gdnConvChannels));

        // 5x inner (dYGated, dYNorm, dZ, dYRec, dV)
        // 2x perHeadQk (dQPerHead, dKPerHead)
        // 2x qkGroup (dQNorm, dKNorm)
        // 2x headScalar (dAlpha, dBetaRaw)
        // 3x conv (dConvOut, dConvPre, dQkv)
        long actTotalFloats = checked(
            5 * innerElems +
            2 * perHeadQkElems +
            2 * qkGroupElems +
            2 * headScalarElems +
            3 * convElems);
        nuint actBytes = checked((nuint)actTotalFloats * sizeof(float));

        try
        {
            if (_bwdHeadSlab == null || headBytes > _bwdHeadSlabBytes || ssmHeads > _bwdHeads || stateDim > _bwdStateDim || seqLen > _bwdSeqLen)
            {
                if (_bwdHeadSlab != null)
                {
                    NativeMemory.AlignedFree(_bwdHeadSlab);
                    _bwdHeadSlab = null;
                    _bwdHeadSlabBytes = 0;
                }
                _bwdHeadSlab = (float*)NativeMemory.AlignedAlloc(headBytes, 64);
                NativeMemory.Clear(_bwdHeadSlab, headBytes);
                _bwdHeadSlabBytes = headBytes;
                _bwdHeads = ssmHeads;
                _bwdStateDim = stateDim;
                _bwdSeqLen = seqLen;
            }

            _bwdHeadStrideFloats = headStride;
            _bwdMatAligned = matAligned;
            _bwdVecAligned = vecAligned;
            _bwdHistMatAligned = histMatAligned;
            _bwdHistVecAligned = histVecAligned;

            if (_bwdActSlab == null || actBytes > _bwdActSlabBytes)
            {
                if (_bwdActSlab != null)
                {
                    NativeMemory.AlignedFree(_bwdActSlab);
                    _bwdActSlab = null;
                    _bwdActSlabBytes = 0;
                }
                _bwdActSlab = (float*)NativeMemory.AlignedAlloc(actBytes, 64);
                NativeMemory.Clear(_bwdActSlab, actBytes);
                _bwdActSlabBytes = actBytes;
            }

            _bwdActInnerOffset = innerElems;
            _bwdActQPerHeadOffset = perHeadQkElems;
            _bwdActQkGroupOffset = qkGroupElems;
            _bwdActHeadsOffset = headScalarElems;
            _bwdActConvOffset = convElems;
        }
        catch
        {
            FreeBackward();
            throw;
        }
    }

    public void GetForwardHeadScratch(int headIndex, int stateDim, out float* sMatZeroed, out float* delta)
    {
        long matElems = (long)stateDim * stateDim;
        long matAligned = Align16(matElems);
        float* headBase = _fwdHeadSlab + (long)headIndex * _fwdHeadStrideFloats;
        sMatZeroed = headBase;
        NativeMemory.Clear(sMatZeroed, checked((nuint)matElems * sizeof(float)));
        delta = headBase + matAligned;
    }

    public void GetBackwardHeadScratch(
        int headIndex,
        int stateDim,
        out float* sPrevHist,
        out float* mHist,
        out float* deltaHist,
        out float* sCurZeroed,
        out float* sPrime,
        out float* sMat,
        out float* dSZeroed,
        out float* dSTot,
        out float* dDelta)
    {
        long matBytes = checked((long)stateDim * stateDim * sizeof(float));
        float* p = _bwdHeadSlab + (long)headIndex * _bwdHeadStrideFloats;

        sPrevHist = p;
        p += _bwdHistMatAligned;
        mHist = p;
        p += _bwdHistVecAligned;
        deltaHist = p;
        p += _bwdHistVecAligned;

        sCurZeroed = p;
        NativeMemory.Clear(sCurZeroed, (nuint)matBytes);
        p += _bwdMatAligned;

        sPrime = p;
        p += _bwdMatAligned;
        sMat = p;
        p += _bwdMatAligned;

        dSZeroed = p;
        NativeMemory.Clear(dSZeroed, (nuint)matBytes);
        p += _bwdMatAligned;

        dSTot = p;
        p += _bwdMatAligned;
        dDelta = p;
    }

    public void GetBackwardActivationBuffers(
        out float* dYGated,
        out float* dYNorm,
        out float* dZ,
        out float* dYRec,
        out float* dV,
        out float* dQPerHead,
        out float* dKPerHead,
        out float* dQNorm,
        out float* dKNorm,
        out float* dAlpha,
        out float* dBetaRaw,
        out float* dConvOut,
        out float* dConvPre,
        out float* dQkv)
    {
        float* p = _bwdActSlab;
        dYGated = p; p += _bwdActInnerOffset;
        dYNorm = p; p += _bwdActInnerOffset;
        dZ = p; p += _bwdActInnerOffset;
        dYRec = p; p += _bwdActInnerOffset;
        dV = p; p += _bwdActInnerOffset;

        dQPerHead = p; p += _bwdActQPerHeadOffset;
        dKPerHead = p; p += _bwdActQPerHeadOffset;

        dQNorm = p; p += _bwdActQkGroupOffset;
        dKNorm = p; p += _bwdActQkGroupOffset;

        dAlpha = p; p += _bwdActHeadsOffset;
        dBetaRaw = p; p += _bwdActHeadsOffset;

        dConvOut = p; p += _bwdActConvOffset;
        dConvPre = p; p += _bwdActConvOffset;
        dQkv = p;
    }

    private void FreeForward()
    {
        if (_fwdHeadSlab != null)
        {
            NativeMemory.AlignedFree(_fwdHeadSlab);
            _fwdHeadSlab = null;
            _fwdHeadSlabBytes = 0;
        }
    }

    private void FreeBackward()
    {
        if (_bwdHeadSlab != null)
        {
            NativeMemory.AlignedFree(_bwdHeadSlab);
            _bwdHeadSlab = null;
            _bwdHeadSlabBytes = 0;
        }
        if (_bwdActSlab != null)
        {
            NativeMemory.AlignedFree(_bwdActSlab);
            _bwdActSlab = null;
            _bwdActSlabBytes = 0;
        }
    }

    private void DisposeCore()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            FreeForward();
            FreeBackward();
        }
    }

    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }
}
