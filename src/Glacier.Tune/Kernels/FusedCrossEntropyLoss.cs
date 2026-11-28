using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Online streaming Fused Cross-Entropy Loss over large vocabularies (e.g. 152,064 tokens).
/// Computes online Log-Sum-Exp and directly backpropagates exact unbiased dLogits into hidden-state gradients (dX)
/// via tiled GPU VRAM or SIMD CPU transpose-GEMMs without allocating full [T x Vocab] tensors.
/// Delegates FP32 CPU and online-softmax primitives to <c>Glacier.Tensor.Losses.FusedCrossEntropyLoss</c>
/// when compiled against local M3.1 <c>Glacier.Tensor</c>.
/// </summary>
public static unsafe class FusedCrossEntropyLoss
{
    internal const int VocabTileSize = 4096;

    /// <summary>
    /// Computes forward-only online streaming Cross-Entropy loss without allocating or mutating gradient buffers.
    /// </summary>
    public static float ComputeLoss(
        float* finalHidden,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight,
        int seqLen,
        int hiddenDim,
        int vocabSize,
        Glacier.Tune.Gpu.GpuLoraEngine? gpuEngine = null)
        => ComputeLoss(finalHidden, targetTokens, lmHeadType, lmHeadWeight, seqLen, hiddenDim, vocabSize, out _, gpuEngine);

    /// <summary>
    /// Computes forward-only online streaming Cross-Entropy loss without allocating or mutating gradient buffers,
    /// returning the count of valid supervised tokens via <paramref name="validTokenCount"/>.
    /// </summary>
    public static float ComputeLoss(
        float* finalHidden,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight,
        int seqLen,
        int hiddenDim,
        int vocabSize,
        out int validTokenCount,
        Glacier.Tune.Gpu.GpuLoraEngine? gpuEngine = null)
    {
        ArgumentNullException.ThrowIfNull(targetTokens);
        if (seqLen <= 0 || hiddenDim <= 0 || vocabSize <= 0)
        {
            validTokenCount = 0;
            return 0.0f;
        }

        var gpu = gpuEngine;
        bool canUseGpu = gpu != null &&
                         gpu.LmHeadWeight != IntPtr.Zero &&
                         gpu.SupportsForwardGemm(lmHeadType);

#if GLACIER_TENSOR_LOCAL
        if (!canUseGpu && lmHeadType == GgufType.F32)
        {
            return Glacier.Tensor.Losses.FusedCrossEntropyLoss.ComputeLoss(
                finalHidden,
                targetTokens.AsSpan(),
                (float*)lmHeadWeight,
                seqLen,
                hiddenDim,
                vocabSize,
                out validTokenCount);
        }
#endif

        int[] validSeqIndices = new int[seqLen];
        int validCount = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (t < targetTokens.Length && targetTokens[t] >= 0 && targetTokens[t] < vocabSize)
            {
                validSeqIndices[validCount++] = t;
            }
        }

        validTokenCount = validCount;
        if (validCount == 0) return 0.0f;

        float invValid = 1.0f / validCount;
        nuint validHiddenBytes = checked((nuint)validCount * (nuint)hiddenDim * sizeof(float));
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        int* validTargetIds = (int*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(int));

        try
        {
            nuint rowCopyBytes = checked((nuint)hiddenDim * sizeof(float));
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                float* src = finalHidden + (long)t * hiddenDim;
                float* dst = pValidHidden + (long)i * hiddenDim;
                Buffer.MemoryCopy(src, dst, rowCopyBytes, rowCopyBytes);
            }

            if (canUseGpu)
            {
                return gpu!.ComputeTiledLmHeadLoss(
                    lmHeadType,
                    pValidHidden,
                    validTargetIds,
                    hiddenDim,
                    vocabSize,
                    validCount,
                    VocabTileSize);
            }

            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
            int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc(checked((nuint)validCount * (nuint)actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(float));

            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    runningMax[i] = float.NegativeInfinity;
                    runningArgMax[i] = -1;
                    runningSumOther[i] = 0.0;
                    targetLogits[i] = 0.0f;
                }

                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    byte* tileWeight = lmHeadWeight + (long)vStart * rowBytes;

                    QuantKernels.MatMulBatch(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount);

                    for (int i = 0; i < validCount; i++)
                    {
                        float* logits = pTileLogits + (long)i * vCount;
                        int targetId = validTargetIds[i];
                        if (targetId >= vStart && targetId < vStart + vCount)
                        {
                            targetLogits[i] = logits[targetId - vStart];
                        }

                        UpdateOnlineTileStats(logits, vStart, vCount, ref runningMax[i], ref runningArgMax[i], ref runningSumOther[i]);
                    }
                }

                double totalLoss = 0.0;
                for (int i = 0; i < validCount; i++)
                {
                    totalLoss += Log1p(runningSumOther[i]) + ((double)runningMax[i] - (double)targetLogits[i]);
                }

                return (float)(totalLoss * invValid);
            }
            finally
            {
                NativeMemory.AlignedFree(pTileLogits);
                NativeMemory.Free(runningMax);
                NativeMemory.Free(runningArgMax);
                NativeMemory.Free(runningSumOther);
                NativeMemory.Free(targetLogits);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pValidHidden);
            NativeMemory.Free(validTargetIds);
        }
    }

    public static float ComputeLossAndGradients(
        float* finalHidden, // [seqLen, hiddenDim]
        int[] inputTokens,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight, // [vocabSize, hiddenDim]
        float* dFinalHidden, // [seqLen, hiddenDim] - output gradient
        int seqLen,
        int hiddenDim,
        int vocabSize,
        Glacier.Tune.Gpu.GpuLoraEngine? gpuEngine = null)
    {
        ArgumentNullException.ThrowIfNull(targetTokens);
        _ = inputTokens;
        if (seqLen <= 0 || hiddenDim <= 0 || vocabSize <= 0)
        {
            return 0.0f;
        }

        var gpu = gpuEngine;
        bool canUseGpu = gpu != null &&
                         gpu.LmHeadWeight != IntPtr.Zero &&
                         gpu.SupportsForwardGemm(lmHeadType) &&
                         gpu.SupportsBackwardGemm(lmHeadType);

#if GLACIER_TENSOR_LOCAL
        if (!canUseGpu && lmHeadType == GgufType.F32)
        {
            return Glacier.Tensor.Losses.FusedCrossEntropyLoss.ComputeLossAndGradients(
                finalHidden,
                targetTokens.AsSpan(),
                (float*)lmHeadWeight,
                dFinalHidden,
                seqLen,
                hiddenDim,
                vocabSize);
        }
#endif

        // Clear hidden state gradients
        long totalHiddenElems = checked((long)seqLen * hiddenDim);
        NativeMemory.Clear(dFinalHidden, checked((nuint)totalHiddenElems * sizeof(float)));

        // 1. Identify valid target tokens (where target >= 0 and target < vocabSize)
        int[] validSeqIndices = new int[seqLen];
        int validCount = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (t < targetTokens.Length && targetTokens[t] >= 0 && targetTokens[t] < vocabSize)
            {
                validSeqIndices[validCount++] = t;
            }
        }

        if (validCount == 0) return 0.0f;

        float invValid = 1.0f / validCount;

        // 2. Pack valid hidden states [validCount, hiddenDim] and valid target IDs
        nuint validHiddenBytes = checked((nuint)validCount * (nuint)hiddenDim * sizeof(float));
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        float* pValidGrad = (float*)NativeMemory.AlignedAlloc(validHiddenBytes, 64);
        int* validTargetIds = (int*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(int));

        try
        {
            NativeMemory.Clear(pValidGrad, validHiddenBytes);

            nuint rowCopyBytes = checked((nuint)hiddenDim * sizeof(float));
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                float* src = finalHidden + (long)t * hiddenDim;
                float* dst = pValidHidden + (long)i * hiddenDim;
                Buffer.MemoryCopy(src, dst, rowCopyBytes, rowCopyBytes);
            }

            // 3. Tiled GPU VRAM execution branch (bounded tile logit memory, zero CPU logit downloads)
            if (canUseGpu)
            {
                float gpuLoss = gpu!.ComputeTiledLmHeadLossAndGradients(
                    lmHeadType,
                    pValidHidden,
                    pValidGrad,
                    validTargetIds,
                    hiddenDim,
                    vocabSize,
                    validCount,
                    VocabTileSize);

                ScatterValidGradients(pValidGrad, dFinalHidden, validSeqIndices, validCount, hiddenDim);
                return gpuLoss;
            }

            // 4. 2D Tiled Online Streaming Softmax & Exact Unbiased Transpose-GEMM on CPU
            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
            int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc(checked((nuint)validCount * (nuint)actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(float));
            double* invDenom = (double*)NativeMemory.Alloc(checked((nuint)validCount), sizeof(double));

            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    runningMax[i] = float.NegativeInfinity;
                    runningArgMax[i] = -1;
                    runningSumOther[i] = 0.0;
                    targetLogits[i] = 0.0f;
                }

                // PASS 1: Online Log-Sum-Exp computation streamed in tiles using Barron's Log1p argmax-excluded split sum
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    byte* tileWeight = lmHeadWeight + (long)vStart * rowBytes;

                    QuantKernels.MatMulBatch(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount);

                    for (int i = 0; i < validCount; i++)
                    {
                        float* logits = pTileLogits + (long)i * vCount;
                        int targetId = validTargetIds[i];
                        if (targetId >= vStart && targetId < vStart + vCount)
                        {
                            targetLogits[i] = logits[targetId - vStart];
                        }

                        UpdateOnlineTileStats(logits, vStart, vCount, ref runningMax[i], ref runningArgMax[i], ref runningSumOther[i]);
                    }
                }

                double totalLoss = 0.0;
                for (int i = 0; i < validCount; i++)
                {
                    invDenom[i] = 1.0 / (1.0 + runningSumOther[i]);
                    totalLoss += Log1p(runningSumOther[i]) + ((double)runningMax[i] - (double)targetLogits[i]);
                }

                // PASS 2: Exact unbiased analytical gradient backpropagation via tiled BaseBackward
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    byte* tileWeight = lmHeadWeight + (long)vStart * rowBytes;

                    if (numTiles > 1)
                    {
                        QuantKernels.MatMulBatch(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount);
                    }

                    int vStartSnap = vStart;
                    int vCountSnap = vCount;
                    Parallel.For(0, validCount, i =>
                    {
                        WriteTiledSoftmaxCrossEntropyGradients(
                            pTileLogits + (long)i * vCountSnap,
                            vStartSnap,
                            vCountSnap,
                            validTargetIds[i],
                            runningMax[i],
                            runningArgMax[i],
                            runningSumOther[i],
                            invDenom[i],
                            invValid);
                    });

                    LoraKernels.AccumulateBaseBackward(
                        lmHeadType,
                        tileWeight,
                        pTileLogits,
                        pValidGrad,
                        validCount,
                        hiddenDim,
                        vCount);
                }

                ScatterValidGradients(pValidGrad, dFinalHidden, validSeqIndices, validCount, hiddenDim);
                return (float)(totalLoss * invValid);
            }
            finally
            {
                NativeMemory.AlignedFree(pTileLogits);
                NativeMemory.Free(runningMax);
                NativeMemory.Free(runningArgMax);
                NativeMemory.Free(runningSumOther);
                NativeMemory.Free(targetLogits);
                NativeMemory.Free(invDenom);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pValidHidden);
            NativeMemory.AlignedFree(pValidGrad);
            NativeMemory.Free(validTargetIds);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScatterValidGradients(float* pValidGrad, float* dFinalHidden, int[] validSeqIndices, int validCount, int hiddenDim)
    {
        ulong rowBytes = (ulong)(hiddenDim * sizeof(float));
        for (int i = 0; i < validCount; i++)
        {
            int t = validSeqIndices[i];
            Buffer.MemoryCopy(
                pValidGrad + (long)i * hiddenDim,
                dFinalHidden + (long)t * hiddenDim,
                rowBytes,
                rowBytes);
        }
    }

#if GLACIER_TENSOR_LOCAL
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Log1p(double x) => Glacier.Tensor.Losses.FusedCrossEntropyLoss.Log1p(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VectorizedMax(float* data, int count) => Glacier.Tensor.Losses.FusedCrossEntropyLoss.VectorizedMax(data, count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double VectorizedSumExp(float* data, int count, float maxVal) => Glacier.Tensor.Losses.FusedCrossEntropyLoss.VectorizedSumExp(data, count, maxVal);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void UpdateOnlineTileStats(
        float* logits,
        int vStart,
        int vCount,
        ref float runningMax,
        ref int runningArgMax,
        ref double runningSumOther)
        => Glacier.Tensor.Losses.FusedCrossEntropyLoss.UpdateOnlineTileStats(logits, vStart, vCount, ref runningMax, ref runningArgMax, ref runningSumOther);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double SumExpExcluding(float* data, int count, float maxVal, int excludeLocalIdx)
        => Glacier.Tensor.Losses.FusedCrossEntropyLoss.SumExpExcluding(data, count, maxVal, excludeLocalIdx);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void WriteTiledSoftmaxCrossEntropyGradients(
        float* logits,
        int vStart,
        int vCount,
        int targetId,
        float maxLogit,
        int globalArgMax,
        double sumOther,
        double invDenom,
        float invValid)
        => Glacier.Tensor.Losses.FusedCrossEntropyLoss.WriteTiledSoftmaxCrossEntropyGradients(
            logits, vStart, vCount, targetId, maxLogit, globalArgMax, sumOther, invDenom, invValid);
#else
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float VectorizedMax(float* data, int count)
    {
        int i = 0;
        float maxVal = float.NegativeInfinity;
        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(float.NegativeInfinity);
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector512.Max(vMax, Vector512.Load(data + i));
            }
            for (int k = 0; k < Vector512<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }
        if (Vector256.IsHardwareAccelerated && count - i >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(float.NegativeInfinity);
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector256.Max(vMax, Vector256.Load(data + i));
            }
            for (int k = 0; k < Vector256<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }
        if (Vector128.IsHardwareAccelerated && count - i >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(float.NegativeInfinity);
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                vMax = Vector128.Max(vMax, Vector128.Load(data + i));
            }
            for (int k = 0; k < Vector128<float>.Count; k++)
            {
                if (vMax[k] > maxVal) maxVal = vMax[k];
            }
        }
        for (; i < count; i++)
        {
            if (data[i] > maxVal) maxVal = data[i];
        }
        return maxVal;
    }

    internal static void UpdateOnlineTileStats(
        float* logits,
        int vStart,
        int vCount,
        ref float runningMax,
        ref int runningArgMax,
        ref double runningSumOther)
    {
        float tileMax = float.NegativeInfinity;
        int tileLocalArgMax = 0;
        for (int v = 0; v < vCount; v++)
        {
            float val = logits[v];
            if (v == 0 || val > tileMax)
            {
                tileMax = val;
                tileLocalArgMax = v;
            }
        }

        if (runningArgMax < 0)
        {
            runningMax = tileMax;
            runningArgMax = vStart + tileLocalArgMax;
            runningSumOther = SumExpExcluding(logits, vCount, tileMax, tileLocalArgMax);
        }
        else if (tileMax > runningMax)
        {
            double factor = float.IsNegativeInfinity(runningMax) ? 0.0 : Math.Exp((double)runningMax - tileMax);
            double tileSumOther = SumExpExcluding(logits, vCount, tileMax, tileLocalArgMax);
            runningSumOther = (1.0 + runningSumOther) * factor + tileSumOther;
            runningMax = tileMax;
            runningArgMax = vStart + tileLocalArgMax;
        }
        else
        {
            runningSumOther += SumExpExcluding(logits, vCount, runningMax, excludeLocalIdx: -1);
        }
    }

    internal static double SumExpExcluding(float* data, int count, float maxVal, int excludeLocalIdx)
    {
        if (float.IsNegativeInfinity(maxVal) || count <= 0)
            return 0.0;

        if (excludeLocalIdx < 0 || excludeLocalIdx >= count)
            return VectorizedSumExp(data, count, maxVal);

        double left = excludeLocalIdx > 0 ? VectorizedSumExp(data, excludeLocalIdx, maxVal) : 0.0;
        int rightCount = count - excludeLocalIdx - 1;
        double right = rightCount > 0 ? VectorizedSumExp(data + excludeLocalIdx + 1, rightCount, maxVal) : 0.0;
        return left + right;
    }

    internal static void WriteTiledSoftmaxCrossEntropyGradients(
        float* logits,
        int vStart,
        int vCount,
        int targetId,
        float maxLogit,
        int globalArgMax,
        double sumOther,
        double invDenom,
        float invValid)
    {
        float scale = (float)(invDenom * invValid);
        int v = 0;

        if (Vector512.IsHardwareAccelerated && vCount >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(maxLogit);
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAvx512(Vector512.Subtract(Vector512.Load(logits + v), vMax));
                Vector512.Multiply(exp, vScale).Store(logits + v);
            }
        }
        if (Vector256.IsHardwareAccelerated && vCount - v >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(maxLogit);
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAvx2(Vector256.Subtract(Vector256.Load(logits + v), vMax));
                Vector256.Multiply(exp, vScale).Store(logits + v);
            }
        }
        if (Vector128.IsHardwareAccelerated && vCount - v >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(maxLogit);
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            int limit = vCount - step;
            for (; v <= limit; v += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAdvSimd(Vector128.Subtract(Vector128.Load(logits + v), vMax));
                Vector128.Multiply(exp, vScale).Store(logits + v);
            }
        }

        for (; v < vCount; v++)
        {
            logits[v] = (float)(Math.Exp((double)logits[v] - maxLogit) * invDenom * invValid);
        }

        int localArgMax = globalArgMax - vStart;
        if ((uint)localArgMax < (uint)vCount)
        {
            logits[localArgMax] = (globalArgMax == targetId)
                ? (float)(-sumOther * invDenom * invValid)
                : (float)(invDenom * invValid);
        }

        int localTarget = targetId - vStart;
        if ((uint)localTarget < (uint)vCount && localTarget != localArgMax)
        {
            logits[localTarget] -= invValid;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static double VectorizedSumExp(float* data, int count, float maxVal)
    {
        if (float.IsNegativeInfinity(maxVal) || count <= 0)
            return 0.0;

        int i = 0;
        double sum = 0.0;

        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vMax = Vector512.Create(maxVal);
            var accLo = Vector512<double>.Zero;
            var accHi = Vector512<double>.Zero;
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAvx512(Vector512.Subtract(Vector512.Load(data + i), vMax));
                var (lo, hi) = Vector512.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector512.Sum(accLo + accHi);
        }
        if (Vector256.IsHardwareAccelerated && count - i >= Vector256<float>.Count)
        {
            var vMax = Vector256.Create(maxVal);
            var accLo = Vector256<double>.Zero;
            var accHi = Vector256<double>.Zero;
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAvx2(Vector256.Subtract(Vector256.Load(data + i), vMax));
                var (lo, hi) = Vector256.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector256.Sum(accLo + accHi);
        }
        if (Vector128.IsHardwareAccelerated && count - i >= Vector128<float>.Count)
        {
            var vMax = Vector128.Create(maxVal);
            var accLo = Vector128<double>.Zero;
            var accHi = Vector128<double>.Zero;
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var exp = TuneSimdMath.FastExpNonPositiveAdvSimd(Vector128.Subtract(Vector128.Load(data + i), vMax));
                var (lo, hi) = Vector128.Widen(exp);
                accLo += lo;
                accHi += hi;
            }
            sum += Vector128.Sum(accLo + accHi);
        }

        for (; i < count; i++)
        {
            sum += Math.Exp((double)data[i] - maxVal);
        }
        return sum;
    }
#endif
}
