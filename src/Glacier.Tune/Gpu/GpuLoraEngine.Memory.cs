namespace Glacier.Tune.Gpu;

using System;
using System.Collections.Generic;
using Glacier.Gpu.Common;
using Glacier.Gpu.Drivers;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Core;
using Glacier.Tune.Model;
using CuDriver = Glacier.Inference.Gpu.CuDriver;

public sealed unsafe partial class GpuLoraEngine
{
    public DeviceResidentTensor<float> AllocateResident(params ReadOnlySpan<int> shape)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var s = new Shape8(shape);
        long elemCountLong = checked((long)s.ElementCount);
        int elemCount = checked((int)elemCountLong);
        nuint bytes = checked((nuint)(elemCountLong * sizeof(float)));

        _gpu.MakeCurrent();
        IntPtr dPtr = _gpu.AllocateDevice(bytes);
        var buf = new GpuBuffer<float>(
            dPtr,
            (ulong)dPtr,
            elemCount,
            GpuDeviceType.NvidiaDiscrete,
            freeAction: ptr =>
            {
                if (_disposed || _gpu.Handle == IntPtr.Zero) return;
                CuDriver.CtxSetCurrent(_gpu.Handle);
                _gpu.FreeDevice(ptr);
            });
        return new DeviceResidentTensor<float>(buf, s, Shape8.ComputeContiguousStrides(in s), 0);
    }

    public DeviceResidentTensor<float> WrapNonOwningResident(IntPtr dPtr, params ReadOnlySpan<int> shape)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var s = new Shape8(shape);
        int elemCount = checked((int)s.ElementCount);
        _gpu.MakeCurrent();
        var buf = new GpuBuffer<float>(
            dPtr,
            (ulong)dPtr,
            elemCount,
            GpuDeviceType.NvidiaDiscrete,
            freeAction: static _ => { });
        return new DeviceResidentTensor<float>(buf, s, Shape8.ComputeContiguousStrides(in s), 0);
    }

    public void ZeroResident(DeviceResidentTensor<float> tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(tensor);
        _gpu.MakeCurrent();
        nuint bytes = checked((nuint)((long)tensor.ElementCount * sizeof(float)));
        if (bytes > 0)
        {
            CuDriver.Check(CuDriver.MemsetD8(tensor.DevicePointer, 0, bytes), "MemsetD8(ZeroResident)");
        }
    }

    public void CopyHostToResident(float* pSrc, DeviceResidentTensor<float> dst)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(dst);
        nuint bytes = checked((nuint)((long)dst.ElementCount * sizeof(float)));
        if (bytes > 0)
        {
            _gpu.MakeCurrent();
            _gpu.CopyToDevice(dst.DevicePointer, (IntPtr)pSrc, bytes);
        }
    }

    public void CopyResidentToHost(DeviceResidentTensor<float> src, float* pDst)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(src);
        nuint bytes = checked((nuint)((long)src.ElementCount * sizeof(float)));
        if (bytes > 0)
        {
            _gpu.MakeCurrent();
            _gpu.CopyToHost((IntPtr)pDst, src.DevicePointer, bytes);
        }
    }

    public void BeginResidentStep(IEnumerable<LoraAdapter> adapters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _gpu.MakeCurrent();
        lock (_scratchLock)
        {
            foreach (var adapter in adapters)
            {
                if (!adapter.IsEnabled) continue;

                nuint aBytes = checked((nuint)((long)adapter.AdapterA.ElementCount * sizeof(float)));
                nuint bBytes = checked((nuint)((long)adapter.AdapterB.ElementCount * sizeof(float)));
                if (!_loraBuffers.TryGetValue(adapter, out var state))
                {
                    state = new LoraGpuBuffers
                    {
                        DA = _gpu.AllocateDevice(aBytes),
                        DB = _gpu.AllocateDevice(bBytes),
                        DGradA = _gpu.AllocateDevice(aBytes),
                        DGradB = _gpu.AllocateDevice(bBytes),
                        ABytes = aBytes,
                        BBytes = bBytes
                    };
                    _loraBuffers[adapter] = state;
                }

                _gpu.CopyToDevice(state.DA, (IntPtr)adapter.AdapterA.DataPointer, aBytes);
                _gpu.CopyToDevice(state.DB, (IntPtr)adapter.AdapterB.DataPointer, bBytes);
                CuDriver.Check(CuDriver.MemsetD8(state.DGradA, 0, aBytes), "MemsetD8(dGradA)");
                CuDriver.Check(CuDriver.MemsetD8(state.DGradB, 0, bBytes), "MemsetD8(dGradB)");
            }
        }
    }

    public void EndResidentStep(IEnumerable<LoraAdapter> adapters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _gpu.MakeCurrent();

        int maxElems = 0;
        foreach (var adapter in adapters)
        {
            if (!adapter.IsEnabled) continue;
            maxElems = Math.Max(maxElems, (int)Math.Max(adapter.AdapterA.ElementCount, adapter.AdapterB.ElementCount));
        }
        if (maxElems == 0) return;

        float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(maxElems);
        try
        {
            fixed (float* pScratch = scratch)
            {
                lock (_scratchLock)
                {
                    foreach (var adapter in adapters)
                    {
                        if (!adapter.IsEnabled) continue;
                        if (!_loraBuffers.TryGetValue(adapter, out var state)) continue;

                        if (adapter.AdapterA.Grad != null)
                        {
                            _gpu.CopyToHost((IntPtr)pScratch, state.DGradA, state.ABytes);
                            AddSpan(adapter.AdapterA.Grad.DataPointer, pScratch, (int)adapter.AdapterA.ElementCount);
                        }
                        if (adapter.AdapterB.Grad != null)
                        {
                            _gpu.CopyToHost((IntPtr)pScratch, state.DGradB, state.BBytes);
                            AddSpan(adapter.AdapterB.Grad.DataPointer, pScratch, (int)adapter.AdapterB.ElementCount);
                        }
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }

    private static void AddSpan(float* dst, float* src, int count)
    {
        for (int i = 0; i < count; i++)
        {
            dst[i] += src[i];
        }
    }

    private void EnsureLoraTempCapacity(int seqLen, int rank)
    {
        nuint needed = checked((nuint)((long)seqLen * rank * sizeof(float)));
        if (needed > _loraTempCap)
        {
            _gpu.FreeDevice(_dLoraTemp);
            _loraTempCap = needed * 2;
            _dLoraTemp = _gpu.AllocateDevice(_loraTempCap);
        }
    }

    private void EnsureCeCapacity(int validCount)
    {
        int cap = Math.Max(16, validCount);
        if (cap <= _ceTokenCap) return;

        _gpu.FreeDevice(_dCeMax);
        _gpu.FreeDevice(_dCeArgMax);
        _gpu.FreeDevice(_dCeSumOther);
        _gpu.FreeDevice(_dCeTargetLogits);
        _gpu.FreeDevice(_dCeInvDenom);
        _gpu.FreeDevice(_dCeLosses);
        _gpu.FreeDevice(_dCeTargetIds);

        _ceTokenCap = cap * 2;
        nuint c = (nuint)_ceTokenCap;
        _dCeMax = _gpu.AllocateDevice(c * sizeof(float));
        _dCeArgMax = _gpu.AllocateDevice(c * sizeof(int));
        _dCeSumOther = _gpu.AllocateDevice(c * sizeof(double));
        _dCeTargetLogits = _gpu.AllocateDevice(c * sizeof(float));
        _dCeInvDenom = _gpu.AllocateDevice(c * sizeof(double));
        _dCeLosses = _gpu.AllocateDevice(c * sizeof(float));
        _dCeTargetIds = _gpu.AllocateDevice(c * sizeof(int));
    }

    private void EnsureFfnCapacity(nuint normBytes, nuint ffnBytes)
    {
        if (normBytes > _ffnNormXCap)
        {
            _gpu.FreeDevice(_dFfnNormX);
            _gpu.FreeDevice(_dFfnDownOut);
            _ffnNormXCap = normBytes * 2;
            _ffnDownOutCap = _ffnNormXCap;
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);
            _dFfnDownOut = _gpu.AllocateDevice(_ffnNormXCap);
        }
        if (ffnBytes > _ffnGateCap)
        {
            _gpu.FreeDevice(_dFfnGate);
            _gpu.FreeDevice(_dFfnUp);
            _gpu.FreeDevice(_dFfnHidden);
            _ffnGateCap = ffnBytes * 2;
            _ffnUpCap = _ffnGateCap;
            _ffnHiddenCap = _ffnGateCap;
            _dFfnGate = _gpu.AllocateDevice(_ffnGateCap);
            _dFfnUp = _gpu.AllocateDevice(_ffnGateCap);
            _dFfnHidden = _gpu.AllocateDevice(_ffnGateCap);
        }
    }

    private void EnsureQkvCapacity(nuint normBytes, nuint qBytes, nuint kvBytes)
    {
        if (normBytes > _ffnNormXCap)
        {
            _gpu.FreeDevice(_dFfnNormX);
            _gpu.FreeDevice(_dFfnDownOut);
            _ffnNormXCap = normBytes * 2;
            _ffnDownOutCap = _ffnNormXCap;
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);
            _dFfnDownOut = _gpu.AllocateDevice(_ffnNormXCap);
        }
        if (qBytes > _qBufCap)
        {
            _gpu.FreeDevice(_dQBuf);
            _qBufCap = qBytes * 2;
            _dQBuf = _gpu.AllocateDevice(_qBufCap);
        }
        if (kvBytes > _kBufCap)
        {
            _gpu.FreeDevice(_dKBuf);
            _gpu.FreeDevice(_dVBuf);
            _kBufCap = kvBytes * 2;
            _vBufCap = _kBufCap;
            _dKBuf = _gpu.AllocateDevice(_kBufCap);
            _dVBuf = _gpu.AllocateDevice(_kBufCap);
        }
    }

    private void EnsureAttnCapacity(int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;
        nuint attnOutBytes = (nuint)((long)seqLen * qDim * sizeof(float));
        nuint lseBytes = (nuint)((long)nHeadsQ * seqLen * sizeof(float));
        nuint kvBytes = (nuint)((long)seqLen * kvDim * sizeof(float));

        if (attnOutBytes > _qBufCap)
        {
            _gpu.FreeDevice(_dQBuf);
            _qBufCap = attnOutBytes * 2;
            _dQBuf = _gpu.AllocateDevice(_qBufCap);
        }
        if (kvBytes > _kBufCap)
        {
            _gpu.FreeDevice(_dKBuf);
            _gpu.FreeDevice(_dVBuf);
            _kBufCap = kvBytes * 2;
            _vBufCap = _kBufCap;
            _dKBuf = _gpu.AllocateDevice(_kBufCap);
            _dVBuf = _gpu.AllocateDevice(_kBufCap);
        }

        if (attnOutBytes > _attnOutCap)
        {
            _gpu.FreeDevice(_dAttnOut);
            _attnOutCap = attnOutBytes * 2;
            _dAttnOut = _gpu.AllocateDevice(_attnOutCap);
        }
        if (lseBytes > _attnLseCap)
        {
            _gpu.FreeDevice(_dAttnLse);
            _gpu.FreeDevice(_dAttnD);
            _attnLseCap = lseBytes * 2;
            _attnDCap = _attnLseCap;
            _dAttnLse = _gpu.AllocateDevice(_attnLseCap);
            _dAttnD = _gpu.AllocateDevice(_attnDCap);
        }
        if (attnOutBytes > _dqCap)
        {
            _gpu.FreeDevice(_dDq);
            _dqCap = attnOutBytes * 2;
            _dDq = _gpu.AllocateDevice(_dqCap);
        }
        if (kvBytes > _dkCap)
        {
            _gpu.FreeDevice(_dDk);
            _gpu.FreeDevice(_dDv);
            _dkCap = kvBytes * 2;
            _dvCap = _dkCap;
            _dDk = _gpu.AllocateDevice(_dkCap);
            _dDv = _gpu.AllocateDevice(_dvCap);
        }
    }
}
