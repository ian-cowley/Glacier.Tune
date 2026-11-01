using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

namespace Glacier.Tune.Kernels;

public static unsafe partial class LoraKernels
{
    private const int BaseTileRows = 64;
    private const int BaseColBlock = 256;

    /// <summary>
    /// Accumulates frozen base-weight input gradients: dXInput[seqLen, inFeatures] += dY[seqLen, outFeatures] * W_base[outFeatures, inFeatures].
    /// Streams quantized rows once per tile into 64-byte aligned cache memory and executes lock-free SIMD FMA across sequence and feature tiles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateBaseBackward(
        GgufType baseType,
        byte* baseWeightData,
        float* pDY,
        float* pDXInput,
        int seqLen,
        int inFeatures,
        int outFeatures)
    {
        if (baseWeightData == null || pDY == null || pDXInput == null || seqLen <= 0 || inFeatures <= 0 || outFeatures <= 0)
        {
            return;
        }

        int colBlocks = (inFeatures + BaseColBlock - 1) / BaseColBlock;
        int totalSlices = seqLen * colBlocks;

        // Direct zero-dequantization FP32 fast path
        if (baseType == GgufType.F32)
        {
            float* pW = (float*)baseWeightData;
            if (totalSlices >= 2 && (long)seqLen * inFeatures * outFeatures >= 16384)
            {
                Parallel.For(0, totalSlices, workIdx =>
                {
                    int s = workIdx / colBlocks;
                    int colStart = (workIdx % colBlocks) * BaseColBlock;
                    int colLen = Math.Min(BaseColBlock, inFeatures - colStart);
                    float* pDXSlice = pDXInput + (long)s * inFeatures + colStart;
                    float* pDYRow = pDY + (long)s * outFeatures;

                    for (int o = 0; o < outFeatures; o++)
                    {
                        float dy = pDYRow[o];
                        if (dy == 0f) continue;
                        AccumulateScaledSimd(pDXSlice, pW + (long)o * inFeatures + colStart, dy, colLen);
                    }
                });
            }
            else
            {
                for (int s = 0; s < seqLen; s++)
                {
                    float* pDXRow = pDXInput + (long)s * inFeatures;
                    float* pDYRow = pDY + (long)s * outFeatures;
                    for (int o = 0; o < outFeatures; o++)
                    {
                        float dy = pDYRow[o];
                        if (dy == 0f) continue;
                        AccumulateScaledSimd(pDXRow, pW + (long)o * inFeatures, dy, inFeatures);
                    }
                }
            }
            return;
        }

        // Tiled quantized streaming path (supports Q4_K, Q6_K, Q8_0, Q4_0, F16, Q5_K, IQ4, etc.)
        int maxTile = Math.Min(BaseTileRows, outFeatures);
        float* pTileW = (float*)NativeMemory.AlignedAlloc((nuint)((long)maxTile * inFeatures * sizeof(float)), 64);
        int* activeRows = stackalloc int[maxTile];

        try
        {
            for (int oStart = 0; oStart < outFeatures; oStart += BaseTileRows)
            {
                int tileCount = Math.Min(BaseTileRows, outFeatures - oStart);
                int activeCount = 0;

                // Identify rows in this tile that have at least one non-zero upstream gradient across seqLen
                for (int r = 0; r < tileCount; r++)
                {
                    int o = oStart + r;
                    bool nonZero = false;
                    for (int s = 0; s < seqLen; s++)
                    {
                        if (pDY[(long)s * outFeatures + o] != 0f)
                        {
                            nonZero = true;
                            break;
                        }
                    }
                    if (nonZero)
                    {
                        activeRows[activeCount++] = r;
                    }
                }

                if (activeCount == 0) continue;

                // Phase 1: Dequantize each active base weight row ONCE into pTileW
                if (activeCount >= 4 && inFeatures >= 128)
                {
                    int* pActive = activeRows;
                    Parallel.For(0, activeCount, idx =>
                    {
                        int r = pActive[idx];
                        QuantKernels.ExtractEmbedding(baseType, baseWeightData, oStart + r, pTileW + (long)r * inFeatures, inFeatures);
                    });
                }
                else
                {
                    for (int idx = 0; idx < activeCount; idx++)
                    {
                        int r = activeRows[idx];
                        QuantKernels.ExtractEmbedding(baseType, baseWeightData, oStart + r, pTileW + (long)r * inFeatures, inFeatures);
                    }
                }

                // Phase 2: Lock-free parallel SIMD accumulation across (seqLen, colBlock)
                if (totalSlices >= 2 && (long)seqLen * inFeatures * activeCount >= 8192)
                {
                    int* pActive = activeRows;
                    int countSnap = activeCount;
                    int oStartSnap = oStart;
                    Parallel.For(0, totalSlices, workIdx =>
                    {
                        int s = workIdx / colBlocks;
                        int colStart = (workIdx % colBlocks) * BaseColBlock;
                        int colLen = Math.Min(BaseColBlock, inFeatures - colStart);
                        float* pDXSlice = pDXInput + (long)s * inFeatures + colStart;
                        float* pDYRow = pDY + (long)s * outFeatures;

                        for (int idx = 0; idx < countSnap; idx++)
                        {
                            int r = pActive[idx];
                            float dy = pDYRow[oStartSnap + r];
                            if (dy == 0f) continue;
                            AccumulateScaledSimd(pDXSlice, pTileW + (long)r * inFeatures + colStart, dy, colLen);
                        }
                    });
                }
                else
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        float* pDXRow = pDXInput + (long)s * inFeatures;
                        float* pDYRow = pDY + (long)s * outFeatures;
                        for (int idx = 0; idx < activeCount; idx++)
                        {
                            int r = activeRows[idx];
                            float dy = pDYRow[oStart + r];
                            if (dy == 0f) continue;
                            AccumulateScaledSimd(pDXRow, pTileW + (long)r * inFeatures, dy, inFeatures);
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pTileW);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void AccumulateScaledSimd(float* dst, float* src, float scale, int count)
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
                Vector512.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
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
                Vector256.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
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
                Vector128.FusedMultiplyAdd(vSrc, vScale, vDst).Store(dst + i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += scale * src[i];
        }
    }
}
