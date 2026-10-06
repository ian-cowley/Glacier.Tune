namespace Glacier.Tune.Gpu;

using System;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Core;

public sealed unsafe partial class GpuLoraEngine
{
    public void LaunchAttnTrainBwd(
        IntPtr dDOut, IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dProbs,
        IntPtr dDq, IntPtr dDk, IntPtr dDv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim, float scale)
    {
        uint blockSize = (uint)headDim;
        uint gridX = (uint)nHeadsQ;
        uint gridY = (uint)seqLen;
        uint sharedMemBytes = (uint)((4 + seqLen) * sizeof(float));

        IntPtr dAttnDs = _dAttnDs;
        void** pArgsDq = stackalloc void*[11];
        pArgsDq[0] = &dDOut;
        pArgsDq[1] = &dK;
        pArgsDq[2] = &dV;
        pArgsDq[3] = &dProbs;
        pArgsDq[4] = &dDq;
        pArgsDq[5] = &dAttnDs;
        pArgsDq[6] = &seqLen;
        pArgsDq[7] = &nHeadsQ;
        pArgsDq[8] = &nHeadsKv;
        pArgsDq[9] = &headDim;
        pArgsDq[10] = &scale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainBwdDqDs,
            gridX, gridY, 1,
            blockSize, 1, 1,
            sharedMemBytes, IntPtr.Zero,
            (IntPtr)pArgsDq,
            IntPtr.Zero), "LaunchKernel(attention_causal_gqa_train_bwd_dq_ds)");

        uint gridX_kv = (uint)nHeadsKv;
        uint gridY_kv = (uint)seqLen;

        void** pArgsKv = stackalloc void*[10];
        pArgsKv[0] = &dQ;
        pArgsKv[1] = &dDOut;
        pArgsKv[2] = &dProbs;
        pArgsKv[3] = &dAttnDs;
        pArgsKv[4] = &dDk;
        pArgsKv[5] = &dDv;
        pArgsKv[6] = &seqLen;
        pArgsKv[7] = &nHeadsQ;
        pArgsKv[8] = &nHeadsKv;
        pArgsKv[9] = &headDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnTrainBwdDkDv,
            gridX_kv, gridY_kv, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgsKv,
            IntPtr.Zero), "LaunchKernel(attention_causal_gqa_train_bwd_dk_dv)");
    }

    public void BackwardAttention(
        int layerIndex,
        Tensor<float> dAttnOut,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float> attnProbs,
        Tensor<float> dq,
        Tensor<float> dk,
        Tensor<float> dv,
        int seqLen,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        float ropeFreqBase)
    {
        int qDim = nHeadsQ * headDim;
        int kvDim = nHeadsKv * headDim;
        nuint qBytes = (nuint)(seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)(seqLen * kvDim * sizeof(float));
        nuint probsBytes = (nuint)((long)nHeadsQ * seqLen * seqLen * sizeof(float));
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            // Copy inputs to VRAM
            _gpu.CopyToDevice(_dAttnOut, (IntPtr)dAttnOut.DataPointer, qBytes);
            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dAttnProbs, (IntPtr)attnProbs.DataPointer, probsBytes);

            // 1. Attention backward in VRAM -> _dDq, _dDk, _dDv
            LaunchAttnTrainBwd(
                _dAttnOut, _dQBuf, _dKBuf, _dVBuf, _dAttnProbs,
                _dDq, _dDk, _dDv,
                seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            // 2. RoPE backward in VRAM (inverse rotation with invSign = -1.0f)
            LaunchRopeBatch(_dDq, _dDk, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, -1.0f);

            // Copy gradients back to host
            _gpu.CopyToHost((IntPtr)dq.DataPointer, _dDq, qBytes);
            _gpu.CopyToHost((IntPtr)dk.DataPointer, _dDk, kvBytes);
            _gpu.CopyToHost((IntPtr)dv.DataPointer, _dDv, kvBytes);
        }
    }
}
