namespace Glacier.Tune.Gpu;

using System;

public sealed unsafe partial class GpuLoraEngine
{
    private void EnsureFfnCapacity(nuint normBytes, nuint ffnBytes)
    {
        if (normBytes > _ffnNormXCap)
        {
            _gpu.FreeDevice(_dFfnNormX);
            _gpu.FreeDevice(_dFfnDownOut);
            _ffnNormXCap = normBytes * 2;
            _dFfnNormX = _gpu.AllocateDevice(_ffnNormXCap);
            _dFfnDownOut = _gpu.AllocateDevice(_ffnNormXCap);
        }
        if (ffnBytes > _ffnGateCap)
        {
            _gpu.FreeDevice(_dFfnGate);
            _gpu.FreeDevice(_dFfnUp);
            _gpu.FreeDevice(_dFfnHidden);
            _ffnGateCap = ffnBytes * 2;
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
            _dKBuf = _gpu.AllocateDevice(_kBufCap);
            _dVBuf = _gpu.AllocateDevice(_kBufCap);
        }
    }

    private void EnsureAttnCapacity(int seqLen, int nHeadsQ, int nHeadsKv, int headDim)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;
        nuint attnOutBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));

        if (attnOutBytes > _attnOutCap)
        {
            _gpu.FreeDevice(_dAttnOut);
            _attnOutCap = attnOutBytes * 2;
            _dAttnOut = _gpu.AllocateDevice(_attnOutCap);
        }
        if (probsBytes > _attnProbsCap)
        {
            _gpu.FreeDevice(_dAttnProbs);
            _gpu.FreeDevice(_dAttnDs);
            _attnProbsCap = probsBytes * 2;
            _attnDsCap = _attnProbsCap;
            _dAttnProbs = _gpu.AllocateDevice(_attnProbsCap);
            _dAttnDs = _gpu.AllocateDevice(_attnDsCap);
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
