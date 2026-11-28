using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Glacier.Tensor.Core;
using Glacier.Tensor.Optimizers;
using Glacier.Tune.Model;

namespace Glacier.Tune.Trainer;

/// <summary>
/// Full-featured SIMD-accelerated AdamW optimizer with atomic state persistence (<c>ADMW</c> v1),
/// double-precision gradient norm clipping, and deterministic resume parity.
/// </summary>
public sealed class StatefulAdamW : IOptimizer
{
    public const uint Magic = 0x574D4441u; // 'ADMW'
    public const int FormatVersion = 1;

    private readonly IReadOnlyList<Tensor<float>> _parameters;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private int _stepCount;
    private bool _disposed;

    public float LearningRate { get; set; }
    public float Beta1 { get; }
    public float Beta2 { get; }
    public float Eps { get; }
    public float WeightDecay { get; }
    public int StepCount => _stepCount;

    public StatefulAdamW(
        IReadOnlyList<Tensor<float>> parameters,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.01f)
    {
        _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        LearningRate = lr;
        Beta1 = beta1;
        Beta2 = beta2;
        Eps = eps;
        WeightDecay = weightDecay;

        _m = new float[_parameters.Count][];
        _v = new float[_parameters.Count][];
        for (int i = 0; i < _parameters.Count; i++)
        {
            long count = _parameters[i].ElementCount;
            if (count < 0 || count > Array.MaxLength)
                throw new ArgumentOutOfRangeException(nameof(parameters), $"Parameter[{i}] element count {count} exceeds supported array bounds.");
            _m[i] = new float[(int)count];
            _v[i] = new float[(int)count];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe void Step()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _stepCount++;
        float bc1 = MathF.Max(1e-12f, 1.0f - MathF.Pow(Beta1, _stepCount));
        float bc2 = MathF.Max(1e-12f, 1.0f - MathF.Pow(Beta2, _stepCount));
        float stepSize = LearningRate / bc1;
        float invBc2 = 1.0f / bc2;
        float oneMinusBeta1 = 1.0f - Beta1;
        float oneMinusBeta2 = 1.0f - Beta2;
        float decayFactor = 1.0f - LearningRate * WeightDecay;

        for (int pIdx = 0; pIdx < _parameters.Count; pIdx++)
        {
            var param = _parameters[pIdx];
            if (param.Grad == null) continue;

            int len = (int)param.ElementCount;
            if (len == 0) continue;

            float* pW = param.DataPointer;
            float* pG = param.Grad.DataPointer;
            fixed (float* pM = _m[pIdx], pV = _v[pIdx])
            {
                StepTensorSimd(pW, pG, pM, pV, len, Beta1, oneMinusBeta1, Beta2, oneMinusBeta2, invBc2, stepSize, Eps, decayFactor);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void StepTensorSimd(
        float* w,
        float* g,
        float* m,
        float* v,
        int len,
        float beta1,
        float oneMinusBeta1,
        float beta2,
        float oneMinusBeta2,
        float invBc2,
        float stepSize,
        float eps,
        float decayFactor)
    {
        int i = 0;

        if (Vector512.IsHardwareAccelerated && len >= 16)
        {
            var vB1 = Vector512.Create(beta1);
            var vOmB1 = Vector512.Create(oneMinusBeta1);
            var vB2 = Vector512.Create(beta2);
            var vOmB2 = Vector512.Create(oneMinusBeta2);
            var vInvBc2 = Vector512.Create(invBc2);
            var vStep = Vector512.Create(stepSize);
            var vEps = Vector512.Create(eps);
            var vDecay = Vector512.Create(decayFactor);

            for (; i <= len - 16; i += 16)
            {
                var vg = Vector512.Load(g + i);
                var vm = Vector512.Load(m + i);
                var vv = Vector512.Load(v + i);
                var vw = Vector512.Load(w + i);

                vm = Avx512F.IsSupported
                    ? Avx512F.FusedMultiplyAdd(vOmB1, vg, vm * vB1)
                    : (vm * vB1 + vg * vOmB1);
                vv = Avx512F.IsSupported
                    ? Avx512F.FusedMultiplyAdd(vOmB2, vg * vg, vv * vB2)
                    : (vv * vB2 + vg * vg * vOmB2);

                var denom = Vector512.Sqrt(vv * vInvBc2) + vEps;
                vw = (vw * vDecay) - (vStep * vm / denom);

                Vector512.Store(vm, m + i);
                Vector512.Store(vv, v + i);
                Vector512.Store(vw, w + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= len - 8)
        {
            var vB1 = Vector256.Create(beta1);
            var vOmB1 = Vector256.Create(oneMinusBeta1);
            var vB2 = Vector256.Create(beta2);
            var vOmB2 = Vector256.Create(oneMinusBeta2);
            var vInvBc2 = Vector256.Create(invBc2);
            var vStep = Vector256.Create(stepSize);
            var vEps = Vector256.Create(eps);
            var vDecay = Vector256.Create(decayFactor);

            for (; i <= len - 8; i += 8)
            {
                var vg = Vector256.Load(g + i);
                var vm = Vector256.Load(m + i);
                var vv = Vector256.Load(v + i);
                var vw = Vector256.Load(w + i);

                vm = Fma.IsSupported
                    ? Fma.MultiplyAdd(vOmB1, vg, vm * vB1)
                    : (vm * vB1 + vg * vOmB1);
                vv = Fma.IsSupported
                    ? Fma.MultiplyAdd(vOmB2, vg * vg, vv * vB2)
                    : (vv * vB2 + vg * vg * vOmB2);

                var denom = Vector256.Sqrt(vv * vInvBc2) + vEps;
                vw = (vw * vDecay) - (vStep * vm / denom);

                Vector256.Store(vm, m + i);
                Vector256.Store(vv, v + i);
                Vector256.Store(vw, w + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= len - 4)
        {
            var vB1 = Vector128.Create(beta1);
            var vOmB1 = Vector128.Create(oneMinusBeta1);
            var vB2 = Vector128.Create(beta2);
            var vOmB2 = Vector128.Create(oneMinusBeta2);
            var vInvBc2 = Vector128.Create(invBc2);
            var vStep = Vector128.Create(stepSize);
            var vEps = Vector128.Create(eps);
            var vDecay = Vector128.Create(decayFactor);

            for (; i <= len - 4; i += 4)
            {
                var vg = Vector128.Load(g + i);
                var vm = Vector128.Load(m + i);
                var vv = Vector128.Load(v + i);
                var vw = Vector128.Load(w + i);

                vm = Fma.IsSupported
                    ? Fma.MultiplyAdd(vOmB1, vg, vm * vB1)
                    : (vm * vB1 + vg * vOmB1);
                vv = Fma.IsSupported
                    ? Fma.MultiplyAdd(vOmB2, vg * vg, vv * vB2)
                    : (vv * vB2 + vg * vg * vOmB2);

                var denom = Vector128.Sqrt(vv * vInvBc2) + vEps;
                vw = (vw * vDecay) - (vStep * vm / denom);

                Vector128.Store(vm, m + i);
                Vector128.Store(vv, v + i);
                Vector128.Store(vw, w + i);
            }
        }

        for (; i < len; i++)
        {
            float gi = g[i];
            float mi = beta1 * m[i] + oneMinusBeta1 * gi;
            float vi = beta2 * v[i] + oneMinusBeta2 * gi * gi;
            m[i] = mi;
            v[i] = vi;
            float denom = MathF.Sqrt(vi * invBc2) + eps;
            w[i] = (w[i] * decayFactor) - (stepSize * mi / denom);
        }
    }

    public void ZeroGrad()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int i = 0; i < _parameters.Count; i++)
        {
            _parameters[i].Grad?.AsSpan().Clear();
        }
    }

    public void SaveState(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("Optimizer state path must not be null or whitespace.", nameof(filePath));

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        string tmpPath = LoraBinaryFormat.CreateTempSiblingPath(filePath);
        try
        {
            using (var fs = File.Create(tmpPath))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(Magic);
                bw.Write(FormatVersion);
                bw.Write(_stepCount);
                bw.Write(LearningRate);
                bw.Write(Beta1);
                bw.Write(Beta2);
                bw.Write(Eps);
                bw.Write(WeightDecay);
                bw.Write(_parameters.Count);

                for (int i = 0; i < _parameters.Count; i++)
                {
                    LoraBinaryFormat.ValidateFinite(_m[i], $"Optimizer m[{i}]");
                    LoraBinaryFormat.ValidateFinite(_v[i], $"Optimizer v[{i}]");
                    bw.Write((long)_m[i].Length);
                    bw.Write(MemoryMarshal.AsBytes(_m[i].AsSpan()));
                    bw.Write(MemoryMarshal.AsBytes(_v[i].AsSpan()));
                }

                bw.Flush();
                fs.Flush(true);
            }

            File.Move(tmpPath, filePath, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(tmpPath)) File.Delete(tmpPath); }
            catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"[StatefulAdamW] Failed to clean temp file '{tmpPath}': {cleanupEx.Message}");
            }
            throw;
        }
    }

    public void LoadState(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Optimizer state file not found: {filePath}", filePath);

        using var fs = File.OpenRead(filePath);
        if (fs.Length < 36)
            throw new InvalidDataException($"Truncated optimizer state header in '{filePath}' ({fs.Length} bytes).");

        using var br = new BinaryReader(fs);

        uint magic = br.ReadUInt32();
        if (magic != Magic)
            throw new InvalidDataException($"Invalid optimizer state magic 0x{magic:X8}; expected 0x{Magic:X8}.");

        int version = br.ReadInt32();
        if (version != FormatVersion)
            throw new InvalidDataException($"Unsupported optimizer state version {version}; expected {FormatVersion}.");

        int stepCount = br.ReadInt32();
        if (stepCount < 0)
            throw new InvalidDataException($"Invalid optimizer step count {stepCount}.");

        float lr = br.ReadSingle();
        float b1 = br.ReadSingle();
        float b2 = br.ReadSingle();
        float eps = br.ReadSingle();
        float wd = br.ReadSingle();
        if (!float.IsFinite(lr) || !float.IsFinite(b1) || !float.IsFinite(b2) || !float.IsFinite(eps) || !float.IsFinite(wd))
            throw new InvalidDataException("Optimizer state file contains non-finite hyperparameter values.");

        int paramCount = br.ReadInt32();
        if (paramCount != _parameters.Count)
            throw new InvalidDataException($"Optimizer state parameter count {paramCount} does not match model parameter count {_parameters.Count}.");

        var loadedM = new float[paramCount][];
        var loadedV = new float[paramCount][];

        for (int i = 0; i < paramCount; i++)
        {
            if (fs.Length - fs.Position < sizeof(long))
                throw new InvalidDataException($"Optimizer state truncated before parameter[{i}] size.");

            long elemCount = br.ReadInt64();
            long expectedCount = _parameters[i].ElementCount;
            if (elemCount != expectedCount)
                throw new InvalidDataException($"Optimizer state parameter[{i}] element count {elemCount} does not match model tensor size {expectedCount}.");

            long byteCountLong = checked(elemCount * sizeof(float));
            if (fs.Length - fs.Position < byteCountLong * 2)
                throw new InvalidDataException($"Optimizer state truncated before parameter[{i}] moments.");

            var mBuf = new float[(int)elemCount];
            var vBuf = new float[(int)elemCount];
            fs.ReadExactly(MemoryMarshal.AsBytes(mBuf.AsSpan()));
            fs.ReadExactly(MemoryMarshal.AsBytes(vBuf.AsSpan()));

            LoraBinaryFormat.ValidateFinite(mBuf, $"LoadState m[{i}]");
            LoraBinaryFormat.ValidateFinite(vBuf, $"LoadState v[{i}]");

            loadedM[i] = mBuf;
            loadedV[i] = vBuf;
        }

        if (fs.Position != fs.Length)
            throw new InvalidDataException($"Unexpected trailing bytes ({fs.Length - fs.Position}) in optimizer state file '{filePath}'.");

        for (int i = 0; i < paramCount; i++)
        {
            loadedM[i].CopyTo(_m[i], 0);
            loadedV[i].CopyTo(_v[i], 0);
        }

        _stepCount = stepCount;
        LearningRate = lr;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float ClipGradients(IReadOnlyList<Tensor<float>> parameters, float maxNorm)
    {
        if (maxNorm <= 0f || !float.IsFinite(maxNorm)) return 0f;

        double totalNormSq = 0.0;
        for (int pIdx = 0; pIdx < parameters.Count; pIdx++)
        {
            var grad = parameters[pIdx].Grad;
            if (grad == null) continue;
            totalNormSq += ComputeSumSqDoubleSimd(grad.AsSpan());
        }

        double totalNormD = Math.Sqrt(totalNormSq);
        if (!double.IsFinite(totalNormD))
        {
            for (int pIdx = 0; pIdx < parameters.Count; pIdx++)
            {
                parameters[pIdx].Grad?.AsSpan().Clear();
            }
            return 0f;
        }

        if (totalNormD > maxNorm)
        {
            float scale = (float)(maxNorm / (totalNormD + 1e-6));
            ScaleGradients(parameters, scale);
        }

        return (float)Math.Min(totalNormD, float.MaxValue);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void ScaleGradients(IReadOnlyList<Tensor<float>> parameters, float scale)
    {
        if (scale == 1.0f) return;

        for (int pIdx = 0; pIdx < parameters.Count; pIdx++)
        {
            var grad = parameters[pIdx].Grad;
            if (grad == null) continue;

            var gradSpan = grad.AsSpan();
            int n = gradSpan.Length;
            fixed (float* ptr = gradSpan)
            {
                int i = 0;
                if (Vector512.IsHardwareAccelerated && n >= 16)
                {
                    var vScale = Vector512.Create(scale);
                    for (; i <= n - 16; i += 16)
                    {
                        Vector512.Store(Vector512.Load(ptr + i) * vScale, ptr + i);
                    }
                }
                if (Vector256.IsHardwareAccelerated && i <= n - 8)
                {
                    var vScale = Vector256.Create(scale);
                    for (; i <= n - 8; i += 8)
                    {
                        Vector256.Store(Vector256.Load(ptr + i) * vScale, ptr + i);
                    }
                }
                if (Vector128.IsHardwareAccelerated && i <= n - 4)
                {
                    var vScale = Vector128.Create(scale);
                    for (; i <= n - 4; i += 4)
                    {
                        Vector128.Store(Vector128.Load(ptr + i) * vScale, ptr + i);
                    }
                }
                for (; i < n; i++)
                {
                    ptr[i] *= scale;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe double ComputeSumSqDoubleSimd(ReadOnlySpan<float> span)
    {
        int n = span.Length;
        if (n == 0) return 0.0;

        double total = 0.0;
        fixed (float* ptr = span)
        {
            int i = 0;
            if (Avx.IsSupported && Vector256.IsHardwareAccelerated && n >= 8)
            {
                var vAcc0 = Vector256<double>.Zero;
                var vAcc1 = Vector256<double>.Zero;
                for (; i <= n - 8; i += 8)
                {
                    var vf = Vector256.Load(ptr + i);
                    var d0 = Avx.ConvertToVector256Double(vf.GetLower());
                    var d1 = Avx.ConvertToVector256Double(vf.GetUpper());
                    vAcc0 = Fma.IsSupported ? Fma.MultiplyAdd(d0, d0, vAcc0) : (vAcc0 + d0 * d0);
                    vAcc1 = Fma.IsSupported ? Fma.MultiplyAdd(d1, d1, vAcc1) : (vAcc1 + d1 * d1);
                }
                total += Vector256.Sum(vAcc0) + Vector256.Sum(vAcc1);
            }
            for (; i < n; i++)
            {
                double v = ptr[i];
                total += v * v;
            }
        }
        return total;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            for (int i = 0; i < _m.Length; i++)
            {
                Array.Clear(_m[i]);
                Array.Clear(_v[i]);
            }
        }
    }
}
