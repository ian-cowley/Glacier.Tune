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
/// </summary>
public static unsafe class FusedCrossEntropyLoss
{
    private const int VocabTileSize = 4096;

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
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc((nuint)((long)validCount * hiddenDim * sizeof(float)), 64);
        int* validTargetIds = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));

        try
        {
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                float* src = finalHidden + (long)t * hiddenDim;
                float* dst = pValidHidden + (long)i * hiddenDim;
                Buffer.MemoryCopy(src, dst, (ulong)(hiddenDim * sizeof(float)), (ulong)(hiddenDim * sizeof(float)));
            }

            var gpu = gpuEngine;
            if (gpu != null &&
                gpu.LmHeadWeight != IntPtr.Zero &&
                gpu.SupportsForwardGemm(lmHeadType))
            {
                return gpu.ComputeTiledLmHeadLoss(
                    lmHeadType,
                    pValidHidden,
                    validTargetIds,
                    hiddenDim,
                    vocabSize,
                    validCount,
                    VocabTileSize,
                    &VectorizedMax,
                    &VectorizedSumExp);
            }

            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
            int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc((nuint)((long)validCount * actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));

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
        // Clear hidden state gradients
        new Span<float>(dFinalHidden, seqLen * hiddenDim).Clear();

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
        float* pValidHidden = (float*)NativeMemory.AlignedAlloc((nuint)((long)validCount * hiddenDim * sizeof(float)), 64);
        float* pValidGrad = (float*)NativeMemory.AlignedAlloc((nuint)((long)validCount * hiddenDim * sizeof(float)), 64);
        int* validTargetIds = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));

        try
        {
            new Span<float>(pValidGrad, validCount * hiddenDim).Clear();

            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                validTargetIds[i] = targetTokens[t];
                float* src = finalHidden + (long)t * hiddenDim;
                float* dst = pValidHidden + (long)i * hiddenDim;
                Buffer.MemoryCopy(src, dst, (ulong)(hiddenDim * sizeof(float)), (ulong)(hiddenDim * sizeof(float)));
            }

            // 3. Tiled GPU VRAM execution branch (bounded < 16 MB tile logit memory, zero CPU dequantization)
            var gpu = gpuEngine;
            if (gpu != null &&
                gpu.LmHeadWeight != IntPtr.Zero &&
                gpu.SupportsForwardGemm(lmHeadType) &&
                gpu.SupportsBackwardGemm(lmHeadType))
            {
                float gpuLoss = gpu.ComputeTiledLmHeadLossAndGradients(
                    lmHeadType,
                    pValidHidden,
                    pValidGrad,
                    validTargetIds,
                    hiddenDim,
                    vocabSize,
                    validCount,
                    VocabTileSize,
                    &VectorizedMax,
                    &VectorizedSumExp);

                ScatterValidGradients(pValidGrad, dFinalHidden, validSeqIndices, validCount, hiddenDim);
                return gpuLoss;
            }

            // 4. 2D Tiled Online Streaming Softmax & Exact Unbiased Transpose-GEMM on CPU
            int actualTileSize = Math.Min(VocabTileSize, vocabSize);
            int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
            int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

            float* pTileLogits = (float*)NativeMemory.AlignedAlloc((nuint)((long)validCount * actualTileSize * sizeof(float)), 64);
            float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            int* runningArgMax = (int*)NativeMemory.Alloc((nuint)validCount, sizeof(int));
            double* runningSumOther = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            double* invDenom = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));

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

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static float VectorizedMax(float* data, int count)
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
        else if (Vector256.IsHardwareAccelerated && count >= Vector256<float>.Count)
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
        else if (Vector128.IsHardwareAccelerated && count >= Vector128<float>.Count)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
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
        if (float.IsNegativeInfinity(maxVal))
            return 0.0;

        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            if (i != excludeLocalIdx)
                sum += Math.Exp((double)data[i] - maxVal);
        }
        return sum;
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
        for (int v = 0; v < vCount; v++)
        {
            int vGlobal = vStart + v;
            if (vGlobal == targetId && vGlobal == globalArgMax)
                logits[v] = (float)(-sumOther * invDenom * invValid);
            else if (vGlobal == targetId)
                logits[v] = (float)((Math.Exp((double)logits[v] - maxLogit) * invDenom - 1.0) * invValid);
            else if (vGlobal == globalArgMax)
                logits[v] = (float)(invDenom * invValid);
            else
                logits[v] = (float)(Math.Exp((double)logits[v] - maxLogit) * invDenom * invValid);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static double VectorizedSumExp(float* data, int count, float maxVal)
    {
        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            sum += MathF.Exp(data[i] - maxVal);
        }
        return sum;
    }
}
