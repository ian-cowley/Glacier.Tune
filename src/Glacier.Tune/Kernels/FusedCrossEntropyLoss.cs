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
            double* runningSumExp = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));
            float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
            double* logSumExp = (double*)NativeMemory.Alloc((nuint)validCount, sizeof(double));

            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    runningMax[i] = float.NegativeInfinity;
                    runningSumExp[i] = 0.0;
                    targetLogits[i] = 0.0f;
                }

                // PASS 1: Online Log-Sum-Exp computation streamed in tiles
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

                        float tileMax = VectorizedMax(logits, vCount);
                        if (float.IsNegativeInfinity(runningMax[i]))
                        {
                            runningMax[i] = tileMax;
                            runningSumExp[i] = VectorizedSumExp(logits, vCount, tileMax);
                        }
                        else
                        {
                            float mNew = MathF.Max(runningMax[i], tileMax);
                            double factor = Math.Exp((double)runningMax[i] - mNew);
                            double tileSum = VectorizedSumExp(logits, vCount, mNew);
                            runningSumExp[i] = runningSumExp[i] * factor + tileSum;
                            runningMax[i] = mNew;
                        }
                    }
                }

                double totalLoss = 0.0;
                for (int i = 0; i < validCount; i++)
                {
                    double lse = runningMax[i] + Math.Log(runningSumExp[i]);
                    logSumExp[i] = lse;
                    totalLoss += lse - targetLogits[i];
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
                        int targetId = validTargetIds[i];
                        double lse = logSumExp[i];
                        float* logits = pTileLogits + (long)i * vCountSnap;
                        for (int v = 0; v < vCountSnap; v++)
                        {
                            if (vStartSnap + v == targetId)
                            {
                                logits[v] = (float)(Math.Exp(logits[v] - lse) - 1.0) * invValid;
                            }
                            else
                            {
                                logits[v] = MathF.Exp((float)(logits[v] - lse)) * invValid;
                            }
                        }
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
                NativeMemory.Free(runningSumExp);
                NativeMemory.Free(targetLogits);
                NativeMemory.Free(logSumExp);
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
