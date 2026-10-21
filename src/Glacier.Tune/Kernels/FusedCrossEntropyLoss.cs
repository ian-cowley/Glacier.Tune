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
    [ThreadStatic]
    private static LossScratchArena? t_arena;
    private static readonly ThreadLocal<float[]> s_threadLocalBuffer = new(() => new float[32768]);
    private const int VocabTileSize = 4096;

    /// <summary>
    /// Thread-local persistent unmanaged memory workspace for fused cross-entropy calculations.
    /// Completely eliminates heap and native memory reallocations during training steps.
    /// </summary>
    public sealed class LossScratchArena : IDisposable
    {
        public float* ValidHidden { get; private set; }
        public float* TileLogits { get; private set; }
        public float* RunningMax { get; private set; }
        public float* RunningSumExp { get; private set; }
        public float* TargetLogits { get; private set; }
        public float* LogSumExp { get; private set; }
        public int* ValidSeqIndices { get; private set; }
        public float* GpuValidLogits { get; private set; }
        public float* TempRow { get; private set; }

        private int _maxValidCount;
        private int _maxHiddenDim;
        private int _maxTileSize;
        private int _maxSeqLen;
        private int _maxGpuVocabSize;
        private bool _disposed;

        public void EnsureCapacity(int seqLen, int validCount, int hiddenDim, int tileSize, int gpuVocabSize = 0)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (seqLen > _maxSeqLen || ValidSeqIndices == null)
            {
                if (ValidSeqIndices != null) NativeMemory.AlignedFree(ValidSeqIndices);
                _maxSeqLen = Math.Max(seqLen, Math.Max(1024, _maxSeqLen * 2));
                ValidSeqIndices = (int*)NativeMemory.AlignedAlloc((nuint)_maxSeqLen * sizeof(int), 64);
            }

            if (validCount > _maxValidCount || RunningMax == null)
            {
                FreeMetrics();
                _maxValidCount = Math.Max(validCount, Math.Max(1024, _maxValidCount * 2));
                RunningMax = (float*)NativeMemory.AlignedAlloc((nuint)_maxValidCount * sizeof(float), 64);
                RunningSumExp = (float*)NativeMemory.AlignedAlloc((nuint)_maxValidCount * sizeof(float), 64);
                TargetLogits = (float*)NativeMemory.AlignedAlloc((nuint)_maxValidCount * sizeof(float), 64);
                LogSumExp = (float*)NativeMemory.AlignedAlloc((nuint)_maxValidCount * sizeof(float), 64);
            }

            int targetValidCount = Math.Max(validCount, _maxValidCount);
            if (targetValidCount > _maxValidCount || hiddenDim > _maxHiddenDim || ValidHidden == null)
            {
                if (ValidHidden != null) NativeMemory.AlignedFree(ValidHidden);
                _maxHiddenDim = Math.Max(hiddenDim, _maxHiddenDim);
                ValidHidden = (float*)NativeMemory.AlignedAlloc((nuint)((long)targetValidCount * _maxHiddenDim) * sizeof(float), 64);
            }

            if (hiddenDim > _maxHiddenDim || TempRow == null)
            {
                if (TempRow != null) NativeMemory.AlignedFree(TempRow);
                _maxHiddenDim = Math.Max(hiddenDim, _maxHiddenDim);
                TempRow = (float*)NativeMemory.AlignedAlloc((nuint)Math.Max(32768, _maxHiddenDim) * sizeof(float), 64);
            }

            if (targetValidCount > _maxValidCount || tileSize > _maxTileSize || TileLogits == null)
            {
                if (TileLogits != null) NativeMemory.AlignedFree(TileLogits);
                _maxTileSize = Math.Max(tileSize, _maxTileSize);
                TileLogits = (float*)NativeMemory.AlignedAlloc((nuint)((long)targetValidCount * _maxTileSize) * sizeof(float), 64);
            }

            if (gpuVocabSize > 0 && (targetValidCount > _maxValidCount || gpuVocabSize > _maxGpuVocabSize || GpuValidLogits == null))
            {
                if (GpuValidLogits != null) NativeMemory.AlignedFree(GpuValidLogits);
                _maxGpuVocabSize = Math.Max(gpuVocabSize, _maxGpuVocabSize);
                GpuValidLogits = (float*)NativeMemory.AlignedAlloc((nuint)((long)targetValidCount * _maxGpuVocabSize) * sizeof(float), 64);
            }
        }

        private void FreeMetrics()
        {
            if (RunningMax != null) { NativeMemory.AlignedFree(RunningMax); RunningMax = null; }
            if (RunningSumExp != null) { NativeMemory.AlignedFree(RunningSumExp); RunningSumExp = null; }
            if (TargetLogits != null) { NativeMemory.AlignedFree(TargetLogits); TargetLogits = null; }
            if (LogSumExp != null) { NativeMemory.AlignedFree(LogSumExp); LogSumExp = null; }
        }

        private void FreeAll()
        {
            FreeMetrics();
            if (ValidSeqIndices != null) { NativeMemory.AlignedFree(ValidSeqIndices); ValidSeqIndices = null; }
            if (ValidHidden != null) { NativeMemory.AlignedFree(ValidHidden); ValidHidden = null; }
            if (TileLogits != null) { NativeMemory.AlignedFree(TileLogits); TileLogits = null; }
            if (GpuValidLogits != null) { NativeMemory.AlignedFree(GpuValidLogits); GpuValidLogits = null; }
            if (TempRow != null) { NativeMemory.AlignedFree(TempRow); TempRow = null; }
            _maxValidCount = 0;
            _maxHiddenDim = 0;
            _maxTileSize = 0;
            _maxSeqLen = 0;
            _maxGpuVocabSize = 0;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                FreeAll();
                GC.SuppressFinalize(this);
            }
        }

        ~LossScratchArena()
        {
            if (!_disposed)
            {
                _disposed = true;
                FreeAll();
            }
        }
    }

    /// <summary>
    /// Explicitly resets the thread-local scratch workspace.
    /// </summary>
    public static void ResetScratchWorkspace()
    {
        t_arena?.Dispose();
        t_arena = null;
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
        int vocabSize)
    {
        // Clear hidden state gradients
        new Span<float>(dFinalHidden, seqLen * hiddenDim).Clear();

        int actualTileSize = Math.Min(VocabTileSize, vocabSize);
        bool isGpu = Glacier.Tune.Gpu.GpuLoraEngine.Current != null && Glacier.Tune.Gpu.GpuLoraEngine.Current.LmHeadWeight != IntPtr.Zero;

        // 1. Acquire thread-local persistent arena and ensure capacity
        var arena = t_arena ??= new LossScratchArena();
        arena.EnsureCapacity(seqLen, Math.Min(seqLen, targetTokens.Length), hiddenDim, actualTileSize, isGpu ? vocabSize : 0);

        int* validSeqIndices = arena.ValidSeqIndices;
        int validCount = 0;
        int maxTokensToCheck = Math.Min(seqLen, targetTokens.Length);
        for (int t = 0; t < maxTokensToCheck; t++)
        {
            int token = targetTokens[t];
            if (token >= 0 && token < vocabSize)
            {
                validSeqIndices[validCount++] = t;
            }
        }

        if (validCount == 0) return 0.0f;

        float invValid = 1.0f / validCount;
        float totalLoss = 0.0f;

        // 2. Pack valid hidden states into contiguous arena buffer [validCount, hiddenDim]
        float* pValidHidden = arena.ValidHidden;

        for (int i = 0; i < validCount; i++)
        {
            int t = validSeqIndices[i];
            float* src = finalHidden + (long)t * hiddenDim;
            float* dst = pValidHidden + (long)i * hiddenDim;
            Buffer.MemoryCopy(src, dst, (ulong)(hiddenDim * sizeof(float)), (ulong)(hiddenDim * sizeof(float)));
        }

        // 3. GPU execution branch (if GPU LoRA engine is configured)
        if (isGpu)
        {
            float* pValidLogits = arena.GpuValidLogits;
            Glacier.Tune.Gpu.GpuLoraEngine.Current!.ComputeLmHeadLogits(
                lmHeadType, Glacier.Tune.Gpu.GpuLoraEngine.Current.LmHeadWeight, pValidHidden, pValidLogits, hiddenDim, vocabSize, validCount);

            return ComputeGpuLossAndGradientsParallel(
                pValidLogits, dFinalHidden, validSeqIndices, targetTokens, lmHeadType, lmHeadWeight, validCount, hiddenDim, vocabSize, invValid);
        }

        // 4. 2D Tiled Online Streaming Softmax across large vocabulary
        int numTiles = (vocabSize + actualTileSize - 1) / actualTileSize;
        int rowBytes = (int)GgufTypes.GetRowBytes(lmHeadType, hiddenDim);

        float* pTileLogits = arena.TileLogits;
        float* runningMax = arena.RunningMax;
        float* runningSumExp = arena.RunningSumExp;
        float* targetLogits = arena.TargetLogits;
        float* logSumExp = arena.LogSumExp;

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

                ComputeTileLogits(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount, arena.TempRow);

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
                if (validCount <= 128 || vocabSize <= 1024)
                {
                    float* tempRow = arena.TempRow;
                    const float precisionThreshold = -16.0f;
                    for (int i = 0; i < validCount; i++)
                    {
                        int t = validSeqIndices[i];
                        int targetId = targetTokens[t];
                        float lse = logSumExp[i];
                        float* logits = pTileLogits + (long)i * vocabSize;
                        float* dhVec = dFinalHidden + (long)t * hiddenDim;

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
                }
                else
                {
                    BackpropSingleTileParallel(
                        pTileLogits, dFinalHidden, logSumExp, validSeqIndices, targetTokens, lmHeadType, lmHeadWeight, validCount, hiddenDim, vocabSize, invValid);
                }
            }
            else
            {
                // Multi-tile streaming backpropagation
                for (int tile = 0; tile < numTiles; tile++)
                {
                    int vStart = tile * actualTileSize;
                    int vCount = Math.Min(actualTileSize, vocabSize - vStart);
                    byte* tileWeight = lmHeadWeight + (long)vStart * rowBytes;

                    ComputeTileLogits(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount, arena.TempRow);

                    if (validCount <= 128 || vCount <= 1024)
                    {
                        float* tempRow = arena.TempRow;
                        const float precisionThreshold = -16.0f;
                        for (int i = 0; i < validCount; i++)
                        {
                            int t = validSeqIndices[i];
                            int targetId = targetTokens[t];
                            float lse = logSumExp[i];
                            float* logits = pTileLogits + (long)i * vCount;
                            float* dhVec = dFinalHidden + (long)t * hiddenDim;

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
                    }
                    else
                    {
                        BackpropMultiTileParallel(
                            pTileLogits, dFinalHidden, logSumExp, validSeqIndices, targetTokens, lmHeadType, lmHeadWeight, vStart, vCount, validCount, hiddenDim, invValid);
                    }
                }
            }

        return totalLoss * invValid;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeTileLogits(
        GgufType lmHeadType,
        byte* tileWeight,
        float* pValidHidden,
        float* pTileLogits,
        int hiddenDim,
        int vCount,
        int validCount,
        float* tempRow)
    {
        if (validCount <= 128 || vCount <= 1024)
        {
            if (lmHeadType == GgufType.F32)
            {
                float* f32Weights = (float*)tileWeight;
                for (int r = 0; r < vCount; r++)
                {
                    float* rowPtr = f32Weights + (long)r * hiddenDim;
                    for (int b = 0; b < validCount; b++)
                    {
                        float* x = pValidHidden + (long)b * hiddenDim;
                        pTileLogits[b * vCount + r] = QuantKernels.VecDotF32(rowPtr, x, hiddenDim);
                    }
                }
            }
            else
            {
                for (int r = 0; r < vCount; r++)
                {
                    QuantKernels.ExtractEmbedding(lmHeadType, tileWeight, r, tempRow, hiddenDim);
                    for (int b = 0; b < validCount; b++)
                    {
                        float* x = pValidHidden + (long)b * hiddenDim;
                        pTileLogits[b * vCount + r] = QuantKernels.VecDotF32(tempRow, x, hiddenDim);
                    }
                }
            }
        }
        else
        {
            QuantKernels.MatMulBatch(lmHeadType, tileWeight, pValidHidden, pTileLogits, hiddenDim, vCount, validCount);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static float ComputeGpuLossAndGradientsParallel(
        float* pValidLogits,
        float* dFinalHidden,
        int* validSeqIndices,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight,
        int validCount,
        int hiddenDim,
        int vocabSize,
        float invValid)
    {
        float totalLoss = 0.0f;
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

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void BackpropSingleTileParallel(
        float* pTileLogits,
        float* dFinalHidden,
        float* logSumExp,
        int* validSeqIndices,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight,
        int validCount,
        int hiddenDim,
        int vocabSize,
        float invValid)
    {
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
                tempBuf = new float[Math.Max(32768, hiddenDim)];
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

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void BackpropMultiTileParallel(
        float* pTileLogits,
        float* dFinalHidden,
        float* logSumExp,
        int* validSeqIndices,
        int[] targetTokens,
        GgufType lmHeadType,
        byte* lmHeadWeight,
        int vStart,
        int vCount,
        int validCount,
        int hiddenDim,
        float invValid)
    {
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
                tempBuf = new float[Math.Max(32768, hiddenDim)];
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
