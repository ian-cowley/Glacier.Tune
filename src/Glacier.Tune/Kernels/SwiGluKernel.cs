using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Forward and backward kernels for SwiGLU: Hidden = SiLU(Gate) * Up.
/// </summary>
public static unsafe class SwiGluKernel
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Forward(float* gate, float* up, float* hidden, int length)
    {
        int chunkSize = 32768;
        int numChunks = (length + chunkSize - 1) / chunkSize;
        Parallel.For(0, numChunks, c =>
        {
            int start = c * chunkSize;
            int end = Math.Min(length, start + chunkSize);
            for (int i = start; i < end; i++)
            {
                float g = gate[i];
                float u = up[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                float siluG = g * sig;
                hidden[i] = siluG * u;
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Backward(
        float* dHidden, float* gate, float* up,
        float* dGate, float* dUp, int length)
    {
        int chunkSize = 32768;
        int numChunks = (length + chunkSize - 1) / chunkSize;
        Parallel.For(0, numChunks, c =>
        {
            int start = c * chunkSize;
            int end = Math.Min(length, start + chunkSize);
            for (int i = start; i < end; i++)
            {
                float dh = dHidden[i];
                float g = gate[i];
                float u = up[i];

                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                float siluG = g * sig;

                // d(Hidden) / d(Up) = SiLU(Gate)
                dUp[i] = dh * siluG;

                // d(SiLU(g)) / dg = sig * (1 + g * (1 - sig))
                float dSilu = sig * (1.0f + g * (1.0f - sig));
                dGate[i] = dh * u * dSilu;
            }
        });
    }
}
