using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
        float* pValidLogits = (float*)NativeMemory.Alloc((nuint)((long)validCount * vocabSize), sizeof(float));

        try
        {
            for (int i = 0; i < validCount; i++)
            {
                int t = validSeqIndices[i];
                float* src = finalHidden + (long)t * hiddenDim;
                float* dst = pValidHidden + (long)i * hiddenDim;
                Buffer.MemoryCopy(src, dst, (ulong)(hiddenDim * sizeof(float)), (ulong)(hiddenDim * sizeof(float)));
            }

            // 3. Batched GEMM: streams lmHeadWeight once from memory across all valid tokens in parallel
            QuantKernels.MatMulBatch(lmHeadType, lmHeadWeight, pValidHidden, pValidLogits, hiddenDim, vocabSize, validCount);

            // Row buffer for extracting embeddings during gradient backprop
            float* tempRow = (float*)NativeMemory.Alloc((nuint)hiddenDim, sizeof(float));
            try
            {
                for (int i = 0; i < validCount; i++)
                {
                    int t = validSeqIndices[i];
                    int targetId = targetTokens[t];
                    float* logits = pValidLogits + (long)i * vocabSize;
                    float* dhVec = dFinalHidden + (long)t * hiddenDim;

                    // A. Log-Sum-Exp
                    float maxLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        if (logits[v] > maxLogit) maxLogit = logits[v];
                    }

                    float sumExp = 0.0f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        sumExp += MathF.Exp(logits[v] - maxLogit);
                    }

                    float logSumExp = maxLogit + MathF.Log(sumExp);
                    float targetLogit = logits[targetId];
                    float tokenLoss = logSumExp - targetLogit;
                    totalLoss += tokenLoss;

                    // B. Softmax probability for target
                    float pTarget = MathF.Exp(targetLogit - logSumExp);

                    // C. Analytical Gradient Backpropagation:
                    // dL / dh = invValid * [ sum_v (p_v * W_v) - W_target ]
                    // = invValid * [ (p_target - 1) * W_target + sum_{v != target} (p_v * W_v) ]

                    // 1) Target token contribution: (pTarget - 1.0) * invValid * W_target
                    QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, targetId, tempRow, hiddenDim);
                    float targetGradScale = (pTarget - 1.0f) * invValid;
                    for (int d = 0; d < hiddenDim; d++)
                    {
                        dhVec[d] += targetGradScale * tempRow[d];
                    }

                    // 2) Top predicted tokens contribution (where p_v >= 0.005f)
                    float threshold = 0.005f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        if (v == targetId) continue;
                        float diff = logits[v] - logSumExp;
                        if (diff < -5.3f) continue; // exp(-5.3) < 0.005

                        float p = MathF.Exp(diff);
                        if (p >= threshold)
                        {
                            QuantKernels.ExtractEmbedding(lmHeadType, lmHeadWeight, v, tempRow, hiddenDim);
                            float pScale = p * invValid;
                            for (int d = 0; d < hiddenDim; d++)
                            {
                                dhVec[d] += pScale * tempRow[d];
                            }
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.Free(tempRow);
            }
        }
        finally
        {
            NativeMemory.Free(pValidHidden);
            NativeMemory.Free(pValidLogits);
        }

        return totalLoss * invValid;
    }
}
