namespace Glacier.Tune.Gpu;

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Tensor.Core;
using Glacier.Tune.Model;

public sealed unsafe partial class GpuLoraEngine
{
    public bool SupportsBackwardGemm(GgufType type) =>
        type is GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.Q4_0 or GgufType.F32;

    public void LaunchGemmBwd(
        GgufType type,
        IntPtr dDX,
        IntPtr dDY,
        IntPtr dW,
        int kCols,
        int mRows,
        int batchSize,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBwd,
            GgufType.Q6_K => _fnGemmQ6KBwd,
            GgufType.Q8_0 => _fnGemmQ8_0Bwd,
            GgufType.Q4_0 => _fnGemmQ4_0Bwd,
            GgufType.F32 => _fnGemmF32Bwd,
            _ => throw new NotSupportedException($"GPU backward GEMM does not support type {type}")
        };

        bool isSuperBlock = type is GgufType.Q4_K or GgufType.Q6_K;
        uint blockSize = isSuperBlock ? 256u : 128u;
        uint gridX = (uint)((kCols + (int)blockSize - 1) / (int)blockSize);
        uint gridY = (uint)((batchSize + 7) / 8);

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dDX;
        pArgs[1] = &dDY;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_bwd)");
    }

    public void LaunchSwigluBwd(IntPtr dDHidden, IntPtr dGate, IntPtr dUp, IntPtr dDGate, IntPtr dDUp, int size)
    {
        if (size <= 0) return;
        uint blockSize = 256;
        uint gridX = (uint)((size + 255) / 256);

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dDHidden;
        pArgs[1] = &dGate;
        pArgs[2] = &dUp;
        pArgs[3] = &dDGate;
        pArgs[4] = &dDUp;
        pArgs[5] = &size;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwigluBwd,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(swiglu_bwd_kernel)");
    }

    public void LaunchRmsNormBwd(IntPtr dDOut, IntPtr dX, IntPtr dWeight, IntPtr dDX, int seqLen, int dim, float eps)
    {
        if (seqLen <= 0 || dim <= 0) return;
        uint blockSize = 256;
        uint gridX = (uint)seqLen;

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dDOut;
        pArgs[1] = &dX;
        pArgs[2] = &dWeight;
        pArgs[3] = &dDX;
        pArgs[4] = &seqLen;
        pArgs[5] = &dim;
        pArgs[6] = &eps;

        CuDriver.CtxSetCurrent(_gpu.Handle);
        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBwd,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(tune_rmsnorm_bwd)");
    }

    public void LaunchLoraBackward(LoraAdapter adapter, IntPtr dX, IntPtr dDY, IntPtr dDXOut, int seqLen)
    {
        if (!adapter.IsEnabled) return;

        int inFeatures = adapter.InFeatures;
        int outFeatures = adapter.OutFeatures;
        int rank = adapter.Rank;
        float scaling = adapter.Scaling;

        lock (_scratchLock)
        {
            if (!_loraBuffers.TryGetValue(adapter, out var state))
            {
                BeginResidentStep([adapter]);
                state = _loraBuffers[adapter];
            }

            EnsureLoraTempCapacity(seqLen, rank);
            IntPtr dA = state.DA;
            IntPtr dB = state.DB;
            IntPtr dGradA = state.DGradA;
            IntPtr dGradB = state.DGradB;
            IntPtr dLr = _dLoraTemp;

            CuDriver.CtxSetCurrent(_gpu.Handle);

            // 1. Recompute lowRank = X * A
            void** pProjArgs = stackalloc void*[6];
            pProjArgs[0] = &dX;
            pProjArgs[1] = &dA;
            pProjArgs[2] = &dLr;
            pProjArgs[3] = &seqLen;
            pProjArgs[4] = &inFeatures;
            pProjArgs[5] = &rank;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnLoraProjLr,
                (uint)rank, (uint)seqLen, 1,
                256, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pProjArgs,
                IntPtr.Zero), "LaunchKernel(tune_lora_proj_lr_bwd)");

            // 2. gradB += scaling * lowRank^T * dY
            void** pDbArgs = stackalloc void*[7];
            pDbArgs[0] = &dLr;
            pDbArgs[1] = &dDY;
            pDbArgs[2] = &dGradB;
            pDbArgs[3] = &seqLen;
            pDbArgs[4] = &outFeatures;
            pDbArgs[5] = &rank;
            pDbArgs[6] = &scaling;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnLoraBwdDb,
                (uint)((outFeatures + 255) / 256), (uint)rank, 1,
                256, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pDbArgs,
                IntPtr.Zero), "LaunchKernel(tune_lora_bwd_db)");

            // 3. dLR = scaling * dY * B^T
            void** pDlrArgs = stackalloc void*[7];
            pDlrArgs[0] = &dDY;
            pDlrArgs[1] = &dB;
            pDlrArgs[2] = &dLr;
            pDlrArgs[3] = &seqLen;
            pDlrArgs[4] = &outFeatures;
            pDlrArgs[5] = &rank;
            pDlrArgs[6] = &scaling;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnLoraBwdDlr,
                (uint)rank, (uint)seqLen, 1,
                256, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pDlrArgs,
                IntPtr.Zero), "LaunchKernel(tune_lora_bwd_dlr)");

            // 4. gradA += X^T * dLR
            int totalA = checked(inFeatures * rank);
            void** pDaArgs = stackalloc void*[6];
            pDaArgs[0] = &dX;
            pDaArgs[1] = &dLr;
            pDaArgs[2] = &dGradA;
            pDaArgs[3] = &seqLen;
            pDaArgs[4] = &inFeatures;
            pDaArgs[5] = &rank;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnLoraBwdDa,
                (uint)((totalA + 255) / 256), 1, 1,
                256, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pDaArgs,
                IntPtr.Zero), "LaunchKernel(tune_lora_bwd_da)");

            // 5. dX += dLR * A^T
            if (dDXOut != IntPtr.Zero)
            {
                void** pDxArgs = stackalloc void*[6];
                pDxArgs[0] = &dLr;
                pDxArgs[1] = &dA;
                pDxArgs[2] = &dDXOut;
                pDxArgs[3] = &seqLen;
                pDxArgs[4] = &inFeatures;
                pDxArgs[5] = &rank;
                CuDriver.Check(CuDriver.LaunchKernel(
                    _fnLoraBwdDx,
                    (uint)((inFeatures + 255) / 256), (uint)seqLen, 1,
                    256, 1, 1,
                    0, IntPtr.Zero,
                    (IntPtr)pDxArgs,
                    IntPtr.Zero), "LaunchKernel(tune_lora_bwd_dx)");
            }
        }
    }

    public void BackwardProjection(
        GgufType type,
        IntPtr dW,
        Tensor<float> dY,
        Tensor<float> dXInput,
        int inFeatures,
        int outFeatures,
        int seqLen)
    {
        nuint bytesDX = (nuint)((long)seqLen * inFeatures * sizeof(float));
        nuint bytesDY = (nuint)((long)seqLen * outFeatures * sizeof(float));

        lock (_scratchLock)
        {
            if (bytesDX > _scratchXCap)
            {
                _gpu.FreeDevice(_dScratchX);
                _dScratchX = _gpu.AllocateDevice(bytesDX * 2);
                _scratchXCap = bytesDX * 2;
            }
            if (bytesDY > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(bytesDY * 2);
                _scratchYCap = bytesDY * 2;
            }

            _gpu.CopyToDevice(_dScratchX, (IntPtr)dXInput.DataPointer, bytesDX);
            _gpu.CopyToDevice(_dScratchY, (IntPtr)dY.DataPointer, bytesDY);

            LaunchGemmBwd(type, _dScratchX, _dScratchY, dW, inFeatures, outFeatures, seqLen);

            _gpu.CopyToHost((IntPtr)dXInput.DataPointer, _dScratchX, bytesDX);
        }
    }

    /// <summary>
    /// Computes exact tiled online streaming Cross-Entropy loss in GPU VRAM without downloading logits to CPU.
    /// Keeps peak logit VRAM buffer bounded to [validCount x vocabTileSize].
    /// </summary>
    public float ComputeTiledLmHeadLoss(
        GgufType lmHeadType,
        float* pValidHidden,
        int* validTargetIds,
        int hiddenDim,
        int vocabSize,
        int validCount,
        int vocabTileSize)
    {
        if (validCount <= 0) return 0f;

        nuint hiddenBytes = (nuint)((long)validCount * hiddenDim * sizeof(float));
        nuint tileLogitBytes = (nuint)((long)validCount * Math.Min(vocabTileSize, vocabSize) * sizeof(float));

        lock (_scratchLock)
        {
            EnsureFfnCapacity(hiddenBytes, tileLogitBytes);
            EnsureCeCapacity(validCount);
            if (tileLogitBytes > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(tileLogitBytes * 2);
                _scratchYCap = tileLogitBytes * 2;
            }

            _gpu.CopyToDevice(_dFfnNormX, (IntPtr)pValidHidden, hiddenBytes);
            _gpu.CopyToDevice(_dCeTargetIds, (IntPtr)validTargetIds, (nuint)validCount * sizeof(int));

            return RunVramCrossEntropyCore(
                lmHeadType, hiddenDim, vocabSize, validCount, vocabTileSize,
                computeGradients: false, gradScale: 0f);
        }
    }

    /// <summary>
    /// Computes exact tiled online streaming Cross-Entropy loss and hidden-state gradients in GPU VRAM
    /// without downloading [validCount x vocabTileSize] logits to host memory.
    /// </summary>
    public float ComputeTiledLmHeadLossAndGradients(
        GgufType lmHeadType,
        float* pValidHidden,
        float* pValidGradOut,
        int* validTargetIds,
        int hiddenDim,
        int vocabSize,
        int validCount,
        int vocabTileSize)
    {
        if (validCount <= 0) return 0f;

        nuint hiddenBytes = (nuint)((long)validCount * hiddenDim * sizeof(float));
        nuint tileLogitBytes = (nuint)((long)validCount * Math.Min(vocabTileSize, vocabSize) * sizeof(float));
        float invValid = 1.0f / validCount;

        lock (_scratchLock)
        {
            EnsureFfnCapacity(hiddenBytes, tileLogitBytes);
            EnsureCeCapacity(validCount);
            if (tileLogitBytes > _scratchYCap)
            {
                _gpu.FreeDevice(_dScratchY);
                _dScratchY = _gpu.AllocateDevice(tileLogitBytes * 2);
                _scratchYCap = tileLogitBytes * 2;
            }

            _gpu.CopyToDevice(_dFfnNormX, (IntPtr)pValidHidden, hiddenBytes);
            _gpu.CopyToDevice(_dCeTargetIds, (IntPtr)validTargetIds, (nuint)validCount * sizeof(int));
            CuDriver.Check(CuDriver.MemsetD8(_dFfnDownOut, 0, hiddenBytes), "MemsetD8(ceGradOut)");

            float loss = RunVramCrossEntropyCore(
                lmHeadType, hiddenDim, vocabSize, validCount, vocabTileSize,
                computeGradients: true, gradScale: invValid);

            _gpu.CopyToHost((IntPtr)pValidGradOut, _dFfnDownOut, hiddenBytes);
            return loss;
        }
    }

    public float ComputeTiledLmHeadLossResident(
        GgufType lmHeadType,
        DeviceResidentTensor<float> finalNormOut,
        int[] targetTokens,
        int hiddenDim,
        int vocabSize,
        int seqLen,
        int vocabTileSize,
        out int validCount)
    {
        int[] validIndices = ArrayPool<int>.Shared.Rent(seqLen);
        int[] validTargets = ArrayPool<int>.Shared.Rent(seqLen);
        try
        {
            validCount = CollectValidTargets(targetTokens, seqLen, vocabSize, validIndices, validTargets);
            if (validCount == 0) return 0f;

            nuint hiddenBytes = (nuint)((long)validCount * hiddenDim * sizeof(float));
            nuint tileLogitBytes = (nuint)((long)validCount * Math.Min(vocabTileSize, vocabSize) * sizeof(float));

            lock (_scratchLock)
            {
                EnsureFfnCapacity(hiddenBytes, tileLogitBytes);
                EnsureCeCapacity(validCount);
                if (tileLogitBytes > _scratchYCap)
                {
                    _gpu.FreeDevice(_dScratchY);
                    _dScratchY = _gpu.AllocateDevice(tileLogitBytes * 2);
                    _scratchYCap = tileLogitBytes * 2;
                }

                GatherRowsDtoD(_dFfnNormX, finalNormOut.DevicePointer, validIndices, validCount, hiddenDim);
                fixed (int* pTargets = validTargets)
                {
                    _gpu.CopyToDevice(_dCeTargetIds, (IntPtr)pTargets, (nuint)validCount * sizeof(int));
                }

                return RunVramCrossEntropyCore(
                    lmHeadType, hiddenDim, vocabSize, validCount, vocabTileSize,
                    computeGradients: false, gradScale: 0f);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(validIndices);
            ArrayPool<int>.Shared.Return(validTargets);
        }
    }

    public float ComputeTiledLmHeadLossAndGradientsResident(
        GgufType lmHeadType,
        DeviceResidentTensor<float> finalNormOut,
        DeviceResidentTensor<float> dFinalNormOut,
        int[] targetTokens,
        int hiddenDim,
        int vocabSize,
        int seqLen,
        int vocabTileSize,
        float lossScale = 1.0f)
    {
        ZeroResident(dFinalNormOut);

        int[] validIndices = ArrayPool<int>.Shared.Rent(seqLen);
        int[] validTargets = ArrayPool<int>.Shared.Rent(seqLen);
        try
        {
            int validCount = CollectValidTargets(targetTokens, seqLen, vocabSize, validIndices, validTargets);
            if (validCount == 0) return 0f;

            nuint hiddenBytes = (nuint)((long)validCount * hiddenDim * sizeof(float));
            nuint tileLogitBytes = (nuint)((long)validCount * Math.Min(vocabTileSize, vocabSize) * sizeof(float));
            float gradScale = lossScale / validCount;

            lock (_scratchLock)
            {
                EnsureFfnCapacity(hiddenBytes, tileLogitBytes);
                EnsureCeCapacity(validCount);
                if (tileLogitBytes > _scratchYCap)
                {
                    _gpu.FreeDevice(_dScratchY);
                    _dScratchY = _gpu.AllocateDevice(tileLogitBytes * 2);
                    _scratchYCap = tileLogitBytes * 2;
                }

                GatherRowsDtoD(_dFfnNormX, finalNormOut.DevicePointer, validIndices, validCount, hiddenDim);
                fixed (int* pTargets = validTargets)
                {
                    _gpu.CopyToDevice(_dCeTargetIds, (IntPtr)pTargets, (nuint)validCount * sizeof(int));
                }
                CuDriver.Check(CuDriver.MemsetD8(_dFfnDownOut, 0, hiddenBytes), "MemsetD8(ceGradOut)");

                float loss = RunVramCrossEntropyCore(
                    lmHeadType, hiddenDim, vocabSize, validCount, vocabTileSize,
                    computeGradients: true, gradScale: gradScale);

                ScatterRowsDtoD(dFinalNormOut.DevicePointer, _dFfnDownOut, validIndices, validCount, hiddenDim);
                return loss;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(validIndices);
            ArrayPool<int>.Shared.Return(validTargets);
        }
    }

    private static int CollectValidTargets(int[] targetTokens, int seqLen, int vocabSize, int[] validIndices, int[] validTargets)
    {
        int validCount = 0;
        for (int t = 0; t < seqLen && t < targetTokens.Length; t++)
        {
            int target = targetTokens[t];
            if (target >= 0 && target < vocabSize)
            {
                validIndices[validCount] = t;
                validTargets[validCount] = target;
                validCount++;
            }
        }
        return validCount;
    }

    private void GatherRowsDtoD(IntPtr dDst, IntPtr dSrc, int[] indices, int validCount, int hiddenDim)
    {
        CuDriver.CtxSetCurrent(_gpu.Handle);
        nuint rowBytes = (nuint)((long)hiddenDim * sizeof(float));
        bool contiguous = validCount > 0 && indices[0] == 0 && indices[validCount - 1] == validCount - 1;
        if (contiguous)
        {
            CuDriver.Check(CuDriver.MemcpyDtoD(dDst, dSrc, (nuint)validCount * rowBytes), "MemcpyDtoD(gather)");
            return;
        }
        for (int i = 0; i < validCount; i++)
        {
            IntPtr srcRow = dSrc + (nint)((long)indices[i] * hiddenDim * sizeof(float));
            IntPtr dstRow = dDst + (nint)((long)i * hiddenDim * sizeof(float));
            CuDriver.Check(CuDriver.MemcpyDtoD(dstRow, srcRow, rowBytes), "MemcpyDtoD(gatherRow)");
        }
    }

    private void ScatterRowsDtoD(IntPtr dDst, IntPtr dSrc, int[] indices, int validCount, int hiddenDim)
    {
        CuDriver.CtxSetCurrent(_gpu.Handle);
        nuint rowBytes = (nuint)((long)hiddenDim * sizeof(float));
        bool contiguous = validCount > 0 && indices[0] == 0 && indices[validCount - 1] == validCount - 1;
        if (contiguous)
        {
            CuDriver.Check(CuDriver.MemcpyDtoD(dDst, dSrc, (nuint)validCount * rowBytes), "MemcpyDtoD(scatter)");
            return;
        }
        for (int i = 0; i < validCount; i++)
        {
            IntPtr srcRow = dSrc + (nint)((long)i * hiddenDim * sizeof(float));
            IntPtr dstRow = dDst + (nint)((long)indices[i] * hiddenDim * sizeof(float));
            CuDriver.Check(CuDriver.MemcpyDtoD(dstRow, srcRow, rowBytes), "MemcpyDtoD(scatterRow)");
        }
    }

    private float RunVramCrossEntropyCore(
        GgufType lmHeadType,
        int hiddenDim,
        int vocabSize,
        int validCount,
        int vocabTileSize,
        bool computeGradients,
        float gradScale)
    {
        int actualTileSize = Math.Min(vocabTileSize, vocabSize);
        int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
        int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

        CuDriver.CtxSetCurrent(_gpu.Handle);

        for (int tile = 0; tile < numTiles; tile++)
        {
            int vStart = tile * actualTileSize;
            int vCount = Math.Min(actualTileSize, vocabSize - vStart);
            IntPtr dTileW = _dLmHeadWeight + (nint)((long)vStart * rowBytes);

            LaunchGemmBatch(lmHeadType, _dScratchY, _dFfnNormX, dTileW, hiddenDim, vCount, validCount);
            LaunchCeTileStats(_dScratchY, _dCeTargetIds, vStart, vCount, validCount);
        }

        LaunchCeFinalizeLoss(validCount);

        float[] hostLosses = ArrayPool<float>.Shared.Rent(validCount);
        double totalLoss = 0.0;
        try
        {
            fixed (float* pLosses = hostLosses)
            {
                _gpu.CopyToHost((IntPtr)pLosses, _dCeLosses, (nuint)validCount * sizeof(float));
            }
            for (int i = 0; i < validCount; i++)
            {
                totalLoss += hostLosses[i];
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hostLosses);
        }

        if (computeGradients)
        {
            for (int tile = 0; tile < numTiles; tile++)
            {
                int vStart = tile * actualTileSize;
                int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                IntPtr dTileW = _dLmHeadWeight + (nint)((long)vStart * rowBytes);

                if (numTiles > 1)
                {
                    LaunchGemmBatch(lmHeadType, _dScratchY, _dFfnNormX, dTileW, hiddenDim, vCount, validCount);
                }

                LaunchCeWriteGrad(_dScratchY, _dCeTargetIds, vStart, vCount, validCount, gradScale);
                LaunchGemmBwd(lmHeadType, _dFfnDownOut, _dScratchY, dTileW, hiddenDim, vCount, validCount);
            }
        }

        return (float)(totalLoss / validCount);
    }

    private void LaunchCeTileStats(IntPtr dTileLogits, IntPtr dTargetIds, int vStart, int vCount, int validCount)
    {
        IntPtr dMax = _dCeMax;
        IntPtr dArgMax = _dCeArgMax;
        IntPtr dSumOther = _dCeSumOther;
        IntPtr dTargetLogits = _dCeTargetLogits;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dTileLogits;
        pArgs[1] = &dTargetIds;
        pArgs[2] = &vStart;
        pArgs[3] = &vCount;
        pArgs[4] = &validCount;
        pArgs[5] = &dMax;
        pArgs[6] = &dArgMax;
        pArgs[7] = &dSumOther;
        pArgs[8] = &dTargetLogits;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnCeTileStats,
            (uint)validCount, 1, 1,
            256, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(tune_ce_tile_stats)");
    }

    private void LaunchCeFinalizeLoss(int validCount)
    {
        IntPtr dMax = _dCeMax;
        IntPtr dSumOther = _dCeSumOther;
        IntPtr dTargetLogits = _dCeTargetLogits;
        IntPtr dInvDenom = _dCeInvDenom;
        IntPtr dLosses = _dCeLosses;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dMax;
        pArgs[1] = &dSumOther;
        pArgs[2] = &dTargetLogits;
        pArgs[3] = &dInvDenom;
        pArgs[4] = &dLosses;
        pArgs[5] = &validCount;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnCeFinalizeLoss,
            (uint)((validCount + 255) / 256), 1, 1,
            256, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(tune_ce_finalize_loss)");
    }

    private void LaunchCeWriteGrad(IntPtr dTileLogits, IntPtr dTargetIds, int vStart, int vCount, int validCount, float gradScale)
    {
        IntPtr dMax = _dCeMax;
        IntPtr dArgMax = _dCeArgMax;
        IntPtr dSumOther = _dCeSumOther;
        IntPtr dInvDenom = _dCeInvDenom;

        void** pArgs = stackalloc void*[10];
        pArgs[0] = &dTileLogits;
        pArgs[1] = &dTargetIds;
        pArgs[2] = &vStart;
        pArgs[3] = &vCount;
        pArgs[4] = &validCount;
        pArgs[5] = &dMax;
        pArgs[6] = &dArgMax;
        pArgs[7] = &dSumOther;
        pArgs[8] = &dInvDenom;
        pArgs[9] = &gradScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnCeWriteGrad,
            (uint)((vCount + 255) / 256), (uint)validCount, 1,
            256, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(tune_ce_write_grad)");
    }

    public void LaunchAttnTrainBwd(
        IntPtr dDOut, IntPtr dAttnOut, IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dLse,
        IntPtr dDq, IntPtr dDk, IntPtr dDv,
        int seqLen, int nHeadsQ, int nHeadsKv, int headDim, float scale)
    {
        if (headDim <= 0 || headDim > 1024)
            throw new ArgumentOutOfRangeException(nameof(headDim), $"GPU training attention requires 1 <= headDim <= 1024, got {headDim}.");

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);

            uint blockSize = (uint)(((headDim + 31) / 32) * 32);
            int bc = AttnTileBc(headDim);
            uint sharedMemBytes = (uint)(2 * bc * headDim * sizeof(float));
            IntPtr dAttnD = _dAttnD;

            CuDriver.CtxSetCurrent(_gpu.Handle);

            void** pPreArgs = stackalloc void*[6];
            pPreArgs[0] = &dDOut;
            pPreArgs[1] = &dAttnOut;
            pPreArgs[2] = &dAttnD;
            pPreArgs[3] = &seqLen;
            pPreArgs[4] = &nHeadsQ;
            pPreArgs[5] = &headDim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBwdPreprocessD,
                (uint)nHeadsQ, (uint)seqLen, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pPreArgs,
                IntPtr.Zero), "LaunchKernel(tune_attn_bwd_preprocess_d)");

            void** pArgsDq = stackalloc void*[12];
            pArgsDq[0] = &dQ;
            pArgsDq[1] = &dK;
            pArgsDq[2] = &dV;
            pArgsDq[3] = &dDOut;
            pArgsDq[4] = &dLse;
            pArgsDq[5] = &dAttnD;
            pArgsDq[6] = &dDq;
            pArgsDq[7] = &seqLen;
            pArgsDq[8] = &nHeadsQ;
            pArgsDq[9] = &nHeadsKv;
            pArgsDq[10] = &headDim;
            pArgsDq[11] = &scale;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnTrainBwdDq,
                (uint)nHeadsQ, (uint)seqLen, 1,
                blockSize, 1, 1,
                sharedMemBytes, IntPtr.Zero,
                (IntPtr)pArgsDq,
                IntPtr.Zero), "LaunchKernel(tune_attn_causal_gqa_bwd_dq)");

            void** pArgsKv = stackalloc void*[13];
            pArgsKv[0] = &dQ;
            pArgsKv[1] = &dK;
            pArgsKv[2] = &dV;
            pArgsKv[3] = &dDOut;
            pArgsKv[4] = &dLse;
            pArgsKv[5] = &dAttnD;
            pArgsKv[6] = &dDk;
            pArgsKv[7] = &dDv;
            pArgsKv[8] = &seqLen;
            pArgsKv[9] = &nHeadsQ;
            pArgsKv[10] = &nHeadsKv;
            pArgsKv[11] = &headDim;
            pArgsKv[12] = &scale;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnTrainBwdDkDv,
                (uint)nHeadsKv, (uint)seqLen, 1,
                blockSize, 1, 1,
                sharedMemBytes, IntPtr.Zero,
                (IntPtr)pArgsKv,
                IntPtr.Zero), "LaunchKernel(tune_attn_causal_gqa_bwd_dk_dv)");
        }
    }

    public void BackwardAttention(
        int layerIndex,
        Tensor<float> dAttnOut,
        Tensor<float> attnOut,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float> attnLse,
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
        nuint qBytes = (nuint)((long)seqLen * qDim * sizeof(float));
        nuint kvBytes = (nuint)((long)seqLen * kvDim * sizeof(float));
        nuint lseBytes = (nuint)((long)nHeadsQ * seqLen * sizeof(float));
        float scale = 1.0f / MathF.Sqrt(headDim);

        lock (_scratchLock)
        {
            EnsureAttnCapacity(seqLen, nHeadsQ, nHeadsKv, headDim);
            if (qBytes > _scratchXCap)
            {
                _gpu.FreeDevice(_dScratchX);
                _scratchXCap = qBytes * 2;
                _dScratchX = _gpu.AllocateDevice(_scratchXCap);
            }

            _gpu.CopyToDevice(_dAttnOut, (IntPtr)attnOut.DataPointer, qBytes);
            _gpu.CopyToDevice(_dScratchX, (IntPtr)dAttnOut.DataPointer, qBytes);
            _gpu.CopyToDevice(_dQBuf, (IntPtr)q.DataPointer, qBytes);
            _gpu.CopyToDevice(_dKBuf, (IntPtr)k.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dVBuf, (IntPtr)v.DataPointer, kvBytes);
            _gpu.CopyToDevice(_dAttnLse, (IntPtr)attnLse.DataPointer, lseBytes);

            LaunchAttnTrainBwd(
                _dScratchX, _dAttnOut, _dQBuf, _dKBuf, _dVBuf, _dAttnLse,
                _dDq, _dDk, _dDv,
                seqLen, nHeadsQ, nHeadsKv, headDim, scale);

            LaunchRopeBatch(_dDq, _dDk, nHeadsQ, nHeadsKv, headDim, seqLen, ropeFreqBase, -1.0f);

            _gpu.CopyToHost((IntPtr)dq.DataPointer, _dDq, qBytes);
            _gpu.CopyToHost((IntPtr)dk.DataPointer, _dDk, kvBytes);
            _gpu.CopyToHost((IntPtr)dv.DataPointer, _dDv, kvBytes);
        }
    }
}
