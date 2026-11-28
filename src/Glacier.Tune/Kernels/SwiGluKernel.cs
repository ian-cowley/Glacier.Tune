using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Glacier.Tune.Kernels;

/// <summary>
/// 3-tier SIMD (AVX-512 / AVX2+FMA / Vector128) forward and backward kernels for SwiGLU: Hidden = SiLU(Gate) * Up.
/// Uses overflow-safe Cody-Waite exp(-|g|) evaluation via <see cref="TuneSimdMath"/>.
/// </summary>
public static unsafe class SwiGluKernel
{
    private const int ChunkSize = 32768;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Forward(float* gate, float* up, float* hidden, int length)
    {
        if (length <= 0) return;

        if (length < ChunkSize)
        {
            ForwardRange(gate, up, hidden, 0, length);
            return;
        }

        int numChunks = (length + ChunkSize - 1) / ChunkSize;
        Parallel.For(0, numChunks, c =>
        {
            int start = c * ChunkSize;
            int end = Math.Min(length, start + ChunkSize);
            ForwardRange(gate, up, hidden, start, end);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Backward(
        float* dHidden, float* gate, float* up,
        float* dGate, float* dUp, int length)
    {
        if (length <= 0) return;

        if (length < ChunkSize)
        {
            BackwardRange(dHidden, gate, up, dGate, dUp, 0, length);
            return;
        }

        int numChunks = (length + ChunkSize - 1) / ChunkSize;
        Parallel.For(0, numChunks, c =>
        {
            int start = c * ChunkSize;
            int end = Math.Min(length, start + ChunkSize);
            BackwardRange(dHidden, gate, up, dGate, dUp, start, end);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ForwardRange(float* gate, float* up, float* hidden, int start, int end)
    {
        int i = start;

        if (Vector512.IsHardwareAccelerated)
        {
            var zero = Vector512<float>.Zero;
            var one = Vector512<float>.One;
            for (; i <= end - 16; i += 16)
            {
                var g = Vector512.Load(gate + i);
                var u = Vector512.Load(up + i);
                var e = TuneSimdMath.FastExpNonPositiveAvx512(-Vector512.Abs(g));
                var num = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                Vector512.Store(g * sig * u, hidden + i);
            }
        }
        if (Vector256.IsHardwareAccelerated)
        {
            var zero = Vector256<float>.Zero;
            var one = Vector256<float>.One;
            for (; i <= end - 8; i += 8)
            {
                var g = Vector256.Load(gate + i);
                var u = Vector256.Load(up + i);
                var e = TuneSimdMath.FastExpNonPositiveAvx2(-Vector256.Abs(g));
                var num = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                Vector256.Store(g * sig * u, hidden + i);
            }
        }
        if (Vector128.IsHardwareAccelerated)
        {
            var zero = Vector128<float>.Zero;
            var one = Vector128<float>.One;
            for (; i <= end - 4; i += 4)
            {
                var g = Vector128.Load(gate + i);
                var u = Vector128.Load(up + i);
                var e = TuneSimdMath.FastExpNonPositiveAdvSimd(-Vector128.Abs(g));
                var num = Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                Vector128.Store(g * sig * u, hidden + i);
            }
        }

        for (; i < end; i++)
        {
            float g = gate[i];
            float u = up[i];
            float e = TuneSimdMath.FastExpNonPositiveScalar(-MathF.Abs(g));
            float sig = (g >= 0f ? 1.0f : e) / (1.0f + e);
            hidden[i] = g * sig * u;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BackwardRange(
        float* dHidden, float* gate, float* up,
        float* dGate, float* dUp, int start, int end)
    {
        int i = start;

        if (Vector512.IsHardwareAccelerated)
        {
            var zero = Vector512<float>.Zero;
            var one = Vector512<float>.One;
            for (; i <= end - 16; i += 16)
            {
                var dh = Vector512.Load(dHidden + i);
                var g = Vector512.Load(gate + i);
                var u = Vector512.Load(up + i);

                var e = TuneSimdMath.FastExpNonPositiveAvx512(-Vector512.Abs(g));
                var num = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                var siluG = g * sig;
                Vector512.Store(dh * siluG, dUp + i);

                var dSilu = Avx512F.IsSupported
                    ? sig * Avx512F.FusedMultiplyAdd(g, one - sig, one)
                    : sig * (one + g * (one - sig));
                Vector512.Store(dh * u * dSilu, dGate + i);
            }
        }
        if (Vector256.IsHardwareAccelerated)
        {
            var zero = Vector256<float>.Zero;
            var one = Vector256<float>.One;
            for (; i <= end - 8; i += 8)
            {
                var dh = Vector256.Load(dHidden + i);
                var g = Vector256.Load(gate + i);
                var u = Vector256.Load(up + i);

                var e = TuneSimdMath.FastExpNonPositiveAvx2(-Vector256.Abs(g));
                var num = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                var siluG = g * sig;
                Vector256.Store(dh * siluG, dUp + i);

                var dSilu = Fma.IsSupported
                    ? sig * Fma.MultiplyAdd(g, one - sig, one)
                    : sig * (one + g * (one - sig));
                Vector256.Store(dh * u * dSilu, dGate + i);
            }
        }
        if (Vector128.IsHardwareAccelerated)
        {
            var zero = Vector128<float>.Zero;
            var one = Vector128<float>.One;
            for (; i <= end - 4; i += 4)
            {
                var dh = Vector128.Load(dHidden + i);
                var g = Vector128.Load(gate + i);
                var u = Vector128.Load(up + i);

                var e = TuneSimdMath.FastExpNonPositiveAdvSimd(-Vector128.Abs(g));
                var num = Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(g, zero), one, e);
                var sig = num / (one + e);
                var siluG = g * sig;
                Vector128.Store(dh * siluG, dUp + i);

                var dSilu = Fma.IsSupported
                    ? sig * Fma.MultiplyAdd(g, one - sig, one)
                    : sig * (one + g * (one - sig));
                Vector128.Store(dh * u * dSilu, dGate + i);
            }
        }

        for (; i < end; i++)
        {
            float dh = dHidden[i];
            float g = gate[i];
            float u = up[i];
            float e = TuneSimdMath.FastExpNonPositiveScalar(-MathF.Abs(g));
            float sig = (g >= 0f ? 1.0f : e) / (1.0f + e);
            float siluG = g * sig;

            dUp[i] = dh * siluG;
            float dSilu = sig * (1.0f + g * (1.0f - sig));
            dGate[i] = dh * u * dSilu;
        }
    }
}
