using System;
using System.Runtime.CompilerServices;

namespace Glacier.Tune.Kernels;

/// <summary>
/// High-performance Rotary Position Embedding (RoPE) forward and backward kernels.
/// Applies orthogonal 2D Givens rotations with exact transpose inversion on the reverse pass.
/// </summary>
public static unsafe class RoPEKernel
{
    public static void ForwardSequence(
        float* q, float* k,
        int seqLen,
        int nHeadsQ, int nHeadsKv, int headDim,
        float freqBase, float freqScale = 1.0f)
    {
        for (int pos = 0; pos < seqLen; pos++)
        {
            float* posQ = q + (long)pos * nHeadsQ * headDim;
            float* posK = k + (long)pos * nHeadsKv * headDim;
            ApplyRoPE(posQ, posK, nHeadsQ, nHeadsKv, headDim, pos, freqBase, freqScale, inverse: false);
        }
    }

    public static void BackwardSequence(
        float* dq, float* dk,
        int seqLen,
        int nHeadsQ, int nHeadsKv, int headDim,
        float freqBase, float freqScale = 1.0f)
    {
        for (int pos = 0; pos < seqLen; pos++)
        {
            float* posDq = dq + (long)pos * nHeadsQ * headDim;
            float* posDk = dk + (long)pos * nHeadsKv * headDim;
            // Reverse rotation has negated sine (-sin(theta))
            ApplyRoPE(posDq, posDk, nHeadsQ, nHeadsKv, headDim, pos, freqBase, freqScale, inverse: true);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRoPE(
        float* q, float* k,
        int nHeadsQ, int nHeadsKv,
        int headDim, int pos,
        float freqBase, float freqScale,
        bool inverse)
    {
        int halfDim = headDim / 2;

        Span<float> cosTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];

        float sinSign = inverse ? -1.0f : 1.0f;

        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(freqBase, (float)(2 * i) / headDim);
            float theta = pos * freq * freqScale;
            cosTable[i] = MathF.Cos(theta);
            sinTable[i] = MathF.Sin(theta) * sinSign;
        }

        fixed (float* pCos = cosTable, pSin = sinTable)
        {
            // Rotate Q heads
            for (int h = 0; h < nHeadsQ; h++)
            {
                float* head = q + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }

            // Rotate K heads
            for (int h = 0; h < nHeadsKv; h++)
            {
                float* head = k + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }
        }
    }
}
