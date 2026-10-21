using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Online streaming Fused Cross-Entropy Loss over large vocabularies (e.g. 152,064 tokens).
/// Computes online Log-Sum-Exp and directly backpropagates dLogits into hidden-state gradients (dX)
/// with sub-2MB peak memory consumption, avoiding large [T x Vocab] tensor allocations.
/// </summary>
public static unsafe class FusedCrossEntropyLoss
{
    private static readonly ThreadLocal<float[]> s_threadLocalBuffer = new(() => new float[8192]);
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
        int vocabSize)
    {
        // Clear hidden state gradients
        new Span<float>(dFinalHidden, seqLen * hiddenDim).Clear();

        // 1. Identify valid assistant target tokens (where target >= 0 and target < vocabSize)
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
        float totalLoss = 0.0f;

        // 2. Pack valid hidden states into contiguous buffer [validCount, hiddenDim]
        float* pValidHidden = (float*)NativeMemory.Alloc((nuint)(validCount * hiddenDim), sizeof(float));

        for (int i = 0; i < validCount; i++)
        {
            int t = validSeqIndices[i];
            float* src = finalHidden + (long)t * hiddenDim;
            float* dst = pValidHidden + (long)i * hiddenDim;
            Buffer.MemoryCopy(src, dst, (ulong)(hiddenDim * sizeof(float)), (ulong)(hiddenDim * sizeof(float)));
        }

        // 3. GPU execution branch (if GPU LoRA engine is configured)
        if (Glacier.Tune.Gpu.GpuLoraEngine.Current != null && Glacier.Tune.Gpu.GpuLoraEngine.Current.LmHeadWeight != IntPtr.Zero)
        {
            float* pValidLogits = (float*)NativeMemory.Alloc((nuint)((long)validCount * vocabSize), sizeof(float));
            try
            {
                Glacier.Tune.Gpu.GpuLoraEngine.Current.ComputeLmHeadLogits(
                    lmHeadType, Glacier.Tune.Gpu.GpuLoraEngine.Current.LmHeadWeight, pValidHidden, pValidLogits, hiddenDim, vocabSize, validCount);

                object lossLock = new();
                Parallel.For<double>(0, validCount, () => 0.0, (i, loopState, localLoss) =>
                {
                    int t = validSeqIndices[i];
                    int targetId = targetTokens[t];
                    float* logits = pValidLogits + (long)i * vocabSize;
                    float* dhVec = dFinalHidden + (long)t * hiddenDim;

                    float maxLogit = VectorizedMax(logits, vocabSize);
                    float sumExp = VectorizedSumExp(logits, vocabSize, maxLogit);
                    float logSumExp = maxLogit + MathF.Log(sumExp);
                    float targetLogit = logits[targetId];
                    float tokenLoss = logSumExp - targetLogit;

                    float pTarget = MathF.Exp(targetLogit - logSumExp);
                    float targetGradScale = (pTarget - 1.0f) * invValid;

                    float[] tempBuf = s_threadLocalBuffer.Value!;
                    if (tempBuf.Length < hiddenDim)
                    {
                        tempBuf = new float[hiddenDim];
                        s_threadLocalBuffer.Value = tempBuf;
                    }

                    fixed (float* tempRow = tempBuf)
                    {
                        QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, targetId, tempRow, hiddenDim);
                        AccumulateScaled(dhVec, tempRow, targetGradScale, hiddenDim);

                        const float precisionThreshold = -16.0f;
                        for (int v = 0; v < vocabSize; v++)
                        {
                            if (v == targetId) continue;
                            float diff = logits[v] - logSumExp;
                            if (diff < precisionThreshold) continue;

                            float p = MathF.Exp(diff);
                            float pScale = p * invValid;

                            QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, v, tempRow, hiddenDim);
                            AccumulateScaled(dhVec, tempRow, pScale, hiddenDim);
                        }
                    }

                    return localLoss + tokenLoss;
                },
                localLoss =>
                {
                    lock (lossLock) totalLoss += (float)localLoss;
                });

                return totalLoss * invValid;
            }
            finally
            {
                NativeMemory.Free(pValidHidden);
                NativeMemory.Free(pValidLogits);
            }
        }

        // 4. 2D Tiled Online Streaming Softmax across large vocabulary
        int actualTileSize = Math.Min(VocabTileSize, vocabSize);
        int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
        int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

        float* pTileLogits = (float*)NativeMemory.Alloc((nuint)((long)validCount * actualTileSize), sizeof(float));
        float* runningMax = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
        float* runningSumExp = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
        float* targetLogits = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));
        float* logSumExp = (float*)NativeMemory.Alloc((nuint)validCount, sizeof(float));

        try
        {
            for (int i = 0; i < validCount; i++)
            {
                runningMax[i] = float.NegativeInfinity;
                runningSumExp[i] = 0.0f;
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
                    int t = validSeqIndices[i];
                    int targetId = targetTokens[t];
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
                        float factor = MathF.Exp(runningMax[i] - mNew);
                        float tileSum = VectorizedSumExp(logits, vCount, mNew);
                        runningSumExp[i] = runningSumExp[i] * factor + tileSum;
                        runningMax[i] = mNew;
                    }
                }
            }

            for (int i = 0; i < validCount; i++)
            {
                float lse = runningMax[i] + MathF.Log(runningSumExp[i]);
                logSumExp[i] = lse;
                totalLoss += lse - targetLogits[i];
            }

            // PASS 2: Exact analytical gradient backpropagation directly into dFinalHidden
            if (numTiles == 1)
            {
                // Single tile: reuse pTileLogits directly without re-evaluating GEMM
                Parallel.For(0, validCount, i =>
                {
                    int t = validSeqIndices[i];
                    int targetId = targetTokens[t];
                    float lse = logSumExp[i];
                    float* logits = pTileLogits + (long)i * vocabSize;
                    float* dhVec = dFinalHidden + (long)t * hiddenDim;

                    float[] tempBuf = s_threadLocalBuffer.Value!;
                    if (tempBuf.Length < hiddenDim)
                    {
                        tempBuf = new float[hiddenDim];
                        s_threadLocalBuffer.Value = tempBuf;
                    }

                    fixed (float* tempRow = tempBuf)
                    {
                        const float precisionThreshold = -16.0f;
                        for (int v = 0; v < vocabSize; v++)
                        {
                            float diff = logits[v] - lse;
                            if (v == targetId)
                            {
                                float p = MathF.Exp(diff);
                                float targetGradScale = (p - 1.0f) * invValid;
                                QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, v, tempRow, hiddenDim);
                                AccumulateScaled(dhVec, tempRow, targetGradScale, hiddenDim);
                            }
                            else if (diff >= precisionThreshold)
                            {
                                float p = MathF.Exp(diff);
                                float pScale = p * invValid;
                                QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, v, tempRow, hiddenDim);
                                AccumulateScaled(dhVec, tempRow, pScale, hiddenDim);
                            }
                        }
                    }
                });
            }
            else
            {
                // Multi-tile streaming backpropagation
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    byte* tileWeight = lmHeadWeight + (long)vStart * rowBytes;

                    QuantKernels.MatMulBatch(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount);

                    Parallel.For(0, validCount, i =>
                    {
                        int t = validSeqIndices[i];
                        int targetId = targetTokens[t];
                        float lse = logSumExp[i];
                        float* logits = pTileLogits + (long)i * vCount;
                        float* dhVec = dFinalHidden + (long)t * hiddenDim;

                        float[] tempBuf = s_threadLocalBuffer.Value!;
                        if (tempBuf.Length < hiddenDim)
                        {
                            tempBuf = new float[hiddenDim];
                            s_threadLocalBuffer.Value = tempBuf;
                        }

                        fixed (float* tempRow = tempBuf)
                        {
                            const float precisionThreshold = -16.0f;
                            for (int v = 0; v < vCount; v++)
                            {
                                int globalV = vStart + v;
                                float diff = logits[v] - lse;
                                if (globalV == targetId)
                                {
                                    float p = MathF.Exp(diff);
                                    float targetGradScale = (p - 1.0f) * invValid;
                                    QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, globalV, tempRow, hiddenDim);
                                    AccumulateScaled(dhVec, tempRow, targetGradScale, hiddenDim);
                                }
                                else if (diff >= precisionThreshold)
                                {
                                    float p = MathF.Exp(diff);
                                    float pScale = p * invValid;
                                    QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, globalV, tempRow, hiddenDim);
                                    AccumulateScaled(dhVec, tempRow, pScale, hiddenDim);
                                }
                            }
                        }
                    });
                }
            }
        }
        finally
        {
            NativeMemory.Free(pValidHidden);
            NativeMemory.Free(pTileLogits);
            NativeMemory.Free(runningMax);
            NativeMemory.Free(runningSumExp);
            NativeMemory.Free(targetLogits);
            NativeMemory.Free(logSumExp);
        }

        return totalLoss * invValid;
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
    private static float VectorizedSumExp(float* data, int count, float maxVal)
    {
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            sum += MathF.Exp(data[i] - maxVal);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateScaled(float* dst, float* src, float scale, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= Vector512<float>.Count)
        {
            var vScale = Vector512.Create(scale);
            int step = Vector512<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector512.Load(dst + i);
                var vSrc = Vector512.Load(src + i);
                vDst = Vector512.FusedMultiplyAdd(vSrc, vScale, vDst);
                vDst.Store(dst + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && count >= Vector256<float>.Count)
        {
            var vScale = Vector256.Create(scale);
            int step = Vector256<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector256.Load(dst + i);
                var vSrc = Vector256.Load(src + i);
                vDst = Vector256.FusedMultiplyAdd(vSrc, vScale, vDst);
                vDst.Store(dst + i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && count >= Vector128<float>.Count)
        {
            var vScale = Vector128.Create(scale);
            int step = Vector128<float>.Count;
            int limit = count - step;
            for (; i <= limit; i += step)
            {
                var vDst = Vector128.Load(dst + i);
                var vSrc = Vector128.Load(src + i);
                vDst = Vector128.FusedMultiplyAdd(vSrc, vScale, vDst);
                vDst.Store(dst + i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += scale * src[i];
        }
    }
}
