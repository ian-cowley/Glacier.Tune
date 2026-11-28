using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Glacier.Tune.Kernels;

/// <summary>
/// Vectorized Cody-Waite range-reduced transcendental math and widening reductions.
/// Delegates directly to <c>Glacier.Tensor.Compute.SimdMath</c> when built against local
/// M3.1 <c>Glacier.Tensor</c>, and provides a standalone-CI-compatible SIMD implementation otherwise.
/// </summary>
internal static unsafe class TuneSimdMath
{
#if GLACIER_TENSOR_LOCAL
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExpNonPositiveScalar(float x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpNonPositiveScalar(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> FastExpNonPositiveAvx512(Vector512<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpNonPositiveAvx512(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastExpNonPositiveAvx2(Vector256<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpNonPositiveAvx2(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> FastExpNonPositiveAdvSimd(Vector128<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpNonPositiveAdvSimd(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExpScalar(float x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpScalar(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> FastExpAvx512(Vector512<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpAvx512(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastExpAvx2(Vector256<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpAvx2(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> FastExpAdvSimd(Vector128<float> x) =>
        Glacier.Tensor.Compute.SimdMath.FastExpAdvSimd(x);
#else
    private const float C1 = 0.693359375f;
    private const float C2 = -2.1219444005469058e-4f;
    private const float Log2e = 1.4426950408889634f;
    private const float UnderflowThreshold = -87.33f;
    private const float OverflowThreshold = 88.72f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExpNonPositiveScalar(float x)
    {
        if (x < UnderflowThreshold) return 0f;
        if (x > 0f) x = 0f;

        float z = x * Log2e;
        int k = (int)MathF.Round(z);
        float r = (x - k * C1) - k * C2;

        float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333333f + r * 0.0013888889f)))));
        float scale = BitConverter.Int32BitsToSingle((k + 127) << 23);
        return p * scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> FastExpNonPositiveAvx512(Vector512<float> x)
    {
        var vUnderflow = Vector512.Create(UnderflowThreshold);
        var vZero = Vector512<float>.Zero;
        var vClamped = Vector512.Min(Vector512.Max(x, vUnderflow), vZero);

        var z = Vector512.Multiply(vClamped, Vector512.Create(Log2e));
        var k = Vector512.ConvertToInt32(Vector512.Round(z));
        var kSingle = Vector512.ConvertToSingle(k);

        var r = Avx512F.IsSupported
            ? Avx512F.FusedMultiplyAdd(kSingle, Vector512.Create(-C2), Avx512F.FusedMultiplyAdd(kSingle, Vector512.Create(-C1), vClamped))
            : Vector512.Subtract(Vector512.Subtract(vClamped, Vector512.Multiply(kSingle, Vector512.Create(C1))), Vector512.Multiply(kSingle, Vector512.Create(C2)));

        var p = EvaluateExpPolyAvx512(r);
        var expBits = Vector512.ShiftLeft(Vector512.Add(k, Vector512.Create(127)), 23);
        var result = Vector512.Multiply(p, Vector512.AsSingle(expBits));
        return Vector512.ConditionalSelect(Vector512.LessThan(x, vUnderflow), vZero, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastExpNonPositiveAvx2(Vector256<float> x)
    {
        var vUnderflow = Vector256.Create(UnderflowThreshold);
        var vZero = Vector256<float>.Zero;
        var vClamped = Vector256.Min(Vector256.Max(x, vUnderflow), vZero);

        var z = Vector256.Multiply(vClamped, Vector256.Create(Log2e));
        var k = Vector256.ConvertToInt32(Vector256.Round(z));
        var kSingle = Vector256.ConvertToSingle(k);

        var r = Fma.IsSupported
            ? Fma.MultiplyAdd(kSingle, Vector256.Create(-C2), Fma.MultiplyAdd(kSingle, Vector256.Create(-C1), vClamped))
            : Vector256.Subtract(Vector256.Subtract(vClamped, Vector256.Multiply(kSingle, Vector256.Create(C1))), Vector256.Multiply(kSingle, Vector256.Create(C2)));

        var p = EvaluateExpPolyAvx2(r);
        var expBits = Vector256.ShiftLeft(Vector256.Add(k, Vector256.Create(127)), 23);
        var result = Vector256.Multiply(p, Vector256.AsSingle(expBits));
        return Vector256.ConditionalSelect(Vector256.LessThan(x, vUnderflow), vZero, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> FastExpNonPositiveAdvSimd(Vector128<float> x)
    {
        var vUnderflow = Vector128.Create(UnderflowThreshold);
        var vZero = Vector128<float>.Zero;
        var vClamped = Vector128.Min(Vector128.Max(x, vUnderflow), vZero);

        var z = Vector128.Multiply(vClamped, Vector128.Create(Log2e));
        var k = Vector128.ConvertToInt32(Vector128.Round(z));
        var kSingle = Vector128.ConvertToSingle(k);

        var r = Vector128.Subtract(
            Vector128.Subtract(vClamped, Vector128.Multiply(kSingle, Vector128.Create(C1))),
            Vector128.Multiply(kSingle, Vector128.Create(C2)));

        var p = EvaluateExpPolyVector128(r);
        var expBits = Vector128.ShiftLeft(Vector128.Add(k, Vector128.Create(127)), 23);
        var result = Vector128.Multiply(p, Vector128.AsSingle(expBits));
        return Vector128.ConditionalSelect(Vector128.LessThan(x, vUnderflow), vZero, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExpScalar(float x)
    {
        if (x < UnderflowThreshold) return 0f;
        if (x > OverflowThreshold) return float.PositiveInfinity;

        float z = x * Log2e;
        int k = (int)MathF.Round(z);
        float r = (x - k * C1) - k * C2;

        float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333333f + r * 0.0013888889f)))));
        int kHalf = k >> 1;
        int kRem = k - kHalf;
        float s1 = BitConverter.Int32BitsToSingle((kHalf + 127) << 23);
        float s2 = BitConverter.Int32BitsToSingle((kRem + 127) << 23);
        return (p * s1) * s2;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> FastExpAvx512(Vector512<float> x)
    {
        var vUnderflow = Vector512.Create(UnderflowThreshold);
        var vOverflow = Vector512.Create(OverflowThreshold);
        var vZero = Vector512<float>.Zero;
        var vClamped = Vector512.Min(Vector512.Max(x, vUnderflow), vOverflow);

        var z = Vector512.Multiply(vClamped, Vector512.Create(Log2e));
        var k = Vector512.ConvertToInt32(Vector512.Round(z));
        var kSingle = Vector512.ConvertToSingle(k);

        var r = Avx512F.IsSupported
            ? Avx512F.FusedMultiplyAdd(kSingle, Vector512.Create(-C2), Avx512F.FusedMultiplyAdd(kSingle, Vector512.Create(-C1), vClamped))
            : Vector512.Subtract(Vector512.Subtract(vClamped, Vector512.Multiply(kSingle, Vector512.Create(C1))), Vector512.Multiply(kSingle, Vector512.Create(C2)));

        var p = EvaluateExpPolyAvx512(r);
        var kHalf = Vector512.ShiftRightArithmetic(k, 1);
        var kRem = Vector512.Subtract(k, kHalf);
        var bias = Vector512.Create(127);
        var s1 = Vector512.AsSingle(Vector512.ShiftLeft(Vector512.Add(kHalf, bias), 23));
        var s2 = Vector512.AsSingle(Vector512.ShiftLeft(Vector512.Add(kRem, bias), 23));
        var result = Vector512.Multiply(Vector512.Multiply(p, s1), s2);
        result = Vector512.ConditionalSelect(Vector512.LessThan(x, vUnderflow), vZero, result);
        return Vector512.ConditionalSelect(Vector512.GreaterThan(x, vOverflow), Vector512.Create(float.PositiveInfinity), result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastExpAvx2(Vector256<float> x)
    {
        var vUnderflow = Vector256.Create(UnderflowThreshold);
        var vOverflow = Vector256.Create(OverflowThreshold);
        var vZero = Vector256<float>.Zero;
        var vClamped = Vector256.Min(Vector256.Max(x, vUnderflow), vOverflow);

        var z = Vector256.Multiply(vClamped, Vector256.Create(Log2e));
        var k = Vector256.ConvertToInt32(Vector256.Round(z));
        var kSingle = Vector256.ConvertToSingle(k);

        var r = Fma.IsSupported
            ? Fma.MultiplyAdd(kSingle, Vector256.Create(-C2), Fma.MultiplyAdd(kSingle, Vector256.Create(-C1), vClamped))
            : Vector256.Subtract(Vector256.Subtract(vClamped, Vector256.Multiply(kSingle, Vector256.Create(C1))), Vector256.Multiply(kSingle, Vector256.Create(C2)));

        var p = EvaluateExpPolyAvx2(r);
        var kHalf = Vector256.ShiftRightArithmetic(k, 1);
        var kRem = Vector256.Subtract(k, kHalf);
        var bias = Vector256.Create(127);
        var s1 = Vector256.AsSingle(Vector256.ShiftLeft(Vector256.Add(kHalf, bias), 23));
        var s2 = Vector256.AsSingle(Vector256.ShiftLeft(Vector256.Add(kRem, bias), 23));
        var result = Vector256.Multiply(Vector256.Multiply(p, s1), s2);
        result = Vector256.ConditionalSelect(Vector256.LessThan(x, vUnderflow), vZero, result);
        return Vector256.ConditionalSelect(Vector256.GreaterThan(x, vOverflow), Vector256.Create(float.PositiveInfinity), result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> FastExpAdvSimd(Vector128<float> x)
    {
        var vUnderflow = Vector128.Create(UnderflowThreshold);
        var vOverflow = Vector128.Create(OverflowThreshold);
        var vZero = Vector128<float>.Zero;
        var vClamped = Vector128.Min(Vector128.Max(x, vUnderflow), vOverflow);

        var z = Vector128.Multiply(vClamped, Vector128.Create(Log2e));
        var k = Vector128.ConvertToInt32(Vector128.Round(z));
        var kSingle = Vector128.ConvertToSingle(k);

        var r = Vector128.Subtract(
            Vector128.Subtract(vClamped, Vector128.Multiply(kSingle, Vector128.Create(C1))),
            Vector128.Multiply(kSingle, Vector128.Create(C2)));

        var p = EvaluateExpPolyVector128(r);
        var kHalf = Vector128.ShiftRightArithmetic(k, 1);
        var kRem = Vector128.Subtract(k, kHalf);
        var bias = Vector128.Create(127);
        var s1 = Vector128.AsSingle(Vector128.ShiftLeft(Vector128.Add(kHalf, bias), 23));
        var s2 = Vector128.AsSingle(Vector128.ShiftLeft(Vector128.Add(kRem, bias), 23));
        var result = Vector128.Multiply(Vector128.Multiply(p, s1), s2);
        result = Vector128.ConditionalSelect(Vector128.LessThan(x, vUnderflow), vZero, result);
        return Vector128.ConditionalSelect(Vector128.GreaterThan(x, vOverflow), Vector128.Create(float.PositiveInfinity), result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> EvaluateExpPolyAvx512(Vector512<float> r)
    {
        var c6 = Vector512.Create(0.0013888889f);
        var c5 = Vector512.Create(0.008333333f);
        var c4 = Vector512.Create(0.041666668f);
        var c3 = Vector512.Create(0.16666667f);
        var c2 = Vector512.Create(0.5f);
        var c1 = Vector512.Create(1.0f);

        if (Avx512F.IsSupported)
        {
            var p = Avx512F.FusedMultiplyAdd(r, c6, c5);
            p = Avx512F.FusedMultiplyAdd(r, p, c4);
            p = Avx512F.FusedMultiplyAdd(r, p, c3);
            p = Avx512F.FusedMultiplyAdd(r, p, c2);
            p = Avx512F.FusedMultiplyAdd(r, p, c1);
            return Avx512F.FusedMultiplyAdd(r, p, c1);
        }

        var res = Vector512.Add(c5, Vector512.Multiply(r, c6));
        res = Vector512.Add(c4, Vector512.Multiply(r, res));
        res = Vector512.Add(c3, Vector512.Multiply(r, res));
        res = Vector512.Add(c2, Vector512.Multiply(r, res));
        res = Vector512.Add(c1, Vector512.Multiply(r, res));
        return Vector512.Add(c1, Vector512.Multiply(r, res));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> EvaluateExpPolyAvx2(Vector256<float> r)
    {
        var c6 = Vector256.Create(0.0013888889f);
        var c5 = Vector256.Create(0.008333333f);
        var c4 = Vector256.Create(0.041666668f);
        var c3 = Vector256.Create(0.16666667f);
        var c2 = Vector256.Create(0.5f);
        var c1 = Vector256.Create(1.0f);

        if (Fma.IsSupported)
        {
            var p = Fma.MultiplyAdd(r, c6, c5);
            p = Fma.MultiplyAdd(r, p, c4);
            p = Fma.MultiplyAdd(r, p, c3);
            p = Fma.MultiplyAdd(r, p, c2);
            p = Fma.MultiplyAdd(r, p, c1);
            return Fma.MultiplyAdd(r, p, c1);
        }

        var res = Vector256.Add(c5, Vector256.Multiply(r, c6));
        res = Vector256.Add(c4, Vector256.Multiply(r, res));
        res = Vector256.Add(c3, Vector256.Multiply(r, res));
        res = Vector256.Add(c2, Vector256.Multiply(r, res));
        res = Vector256.Add(c1, Vector256.Multiply(r, res));
        return Vector256.Add(c1, Vector256.Multiply(r, res));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> EvaluateExpPolyVector128(Vector128<float> r)
    {
        var c6 = Vector128.Create(0.0013888889f);
        var c5 = Vector128.Create(0.008333333f);
        var c4 = Vector128.Create(0.041666668f);
        var c3 = Vector128.Create(0.16666667f);
        var c2 = Vector128.Create(0.5f);
        var c1 = Vector128.Create(1.0f);

        if (AdvSimd.IsSupported)
        {
            var p = AdvSimd.FusedMultiplyAdd(c5, r, c6);
            p = AdvSimd.FusedMultiplyAdd(c4, r, p);
            p = AdvSimd.FusedMultiplyAdd(c3, r, p);
            p = AdvSimd.FusedMultiplyAdd(c2, r, p);
            p = AdvSimd.FusedMultiplyAdd(c1, r, p);
            return AdvSimd.FusedMultiplyAdd(c1, r, p);
        }

        var res = Vector128.Add(c5, Vector128.Multiply(r, c6));
        res = Vector128.Add(c4, Vector128.Multiply(r, res));
        res = Vector128.Add(c3, Vector128.Multiply(r, res));
        res = Vector128.Add(c2, Vector128.Multiply(r, res));
        res = Vector128.Add(c1, Vector128.Multiply(r, res));
        return Vector128.Add(c1, Vector128.Multiply(r, res));
    }
#endif
}
