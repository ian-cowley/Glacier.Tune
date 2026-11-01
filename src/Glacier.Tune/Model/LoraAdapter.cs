using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Kernels;

namespace Glacier.Tune.Model;

/// <summary>
/// Parameter-efficient LoRA adapter for quantized base model weights.
/// Connects directly to unmanaged GGUF base weights without materializing FP32 tensors,
/// while training low-rank factorized matrices A and B:
/// Y = MatMul(BaseWeight, X) + (alpha / r) * (X * A) * B
/// </summary>
public sealed unsafe class LoraAdapter : IDisposable
{
    private readonly int _inFeatures;
    private readonly int _outFeatures;
    private readonly int _rank;
    private readonly float _scaling;
    private readonly bool _isEnabled;
    private readonly Tensor<float> _adapterA; // [InFeatures, Rank]
    private readonly Tensor<float> _adapterB; // [Rank, OutFeatures]
    private readonly List<Tensor<float>> _trainableParameters;
    private readonly GpuTarget _target;
    private Glacier.Tune.Gpu.GpuLoraEngine? _gpuEngine;
    private IntPtr _dBaseWeight = IntPtr.Zero;
    private IntPtr _dBaseBias = IntPtr.Zero;
    private bool _disposed;

    public int InFeatures => _inFeatures;
    public int OutFeatures => _outFeatures;
    public int Rank => _rank;
    public float Scaling => _scaling;
    public bool IsEnabled => _isEnabled;
    public Tensor<float> AdapterA => _adapterA;
    public Tensor<float> AdapterB => _adapterB;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;
    public GpuTarget Target => _target;

    /// <summary>
    /// Returns true if this adapter is disabled or if AdapterB is currently all zeros (Delta W == 0).
    /// Exits on the first non-zero weight in O(1) once training updates B.
    /// </summary>
    public bool IsZeroDelta
    {
        get
        {
            if (!_isEnabled || _scaling == 0f) return true;
            float* pB = _adapterB.DataPointer;
            int len = (int)_adapterB.ElementCount;
            int i = 0;
            if (Vector256.IsHardwareAccelerated && len >= 8)
            {
                for (; i <= len - 8; i += 8)
                {
                    if (Vector256.Load(pB + i) != Vector256<float>.Zero)
                    {
                        return false;
                    }
                }
            }
            for (; i < len; i++)
            {
                if (pB[i] != 0f) return false;
            }
            return true;
        }
    }

    public void SetGpuBaseWeights(Glacier.Tune.Gpu.GpuLoraEngine engine, IntPtr dBaseWeight, IntPtr dBaseBias = default)
    {
        _gpuEngine = engine;
        _dBaseWeight = dBaseWeight;
        _dBaseBias = dBaseBias;
    }

    public LoraAdapter(
        int inFeatures,
        int outFeatures,
        int rank = 16,
        float alpha = 32f,
        int? seed = null,
        GpuTarget target = GpuTarget.Auto,
        bool isEnabled = true)
    {
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;
        _rank = rank;
        _scaling = rank > 0 ? alpha / rank : 0f;
        _isEnabled = isEnabled && rank > 0;
        _target = GpuTarget.Cpu; // Fast SIMD cache-blocked microkernel for low-rank adapters

        if (_isEnabled)
        {
            // Gaussian init std = 1 / sqrt(inFeatures)
            float stdA = 1.0f / MathF.Sqrt(inFeatures);
            _adapterA = TensorFloatExtensions.RandomNormal([inFeatures, rank], mean: 0f, std: stdA, seed: seed ?? 42);
            _adapterB = Tensor<float>.Zeros(rank, outFeatures);

            _adapterA.RequiresGrad = true;
            _adapterB.RequiresGrad = true;
            _adapterA.Grad = Tensor<float>.Zeros(inFeatures, rank);
            _adapterB.Grad = Tensor<float>.Zeros(rank, outFeatures);

            _trainableParameters = [_adapterA, _adapterB];
        }
        else
        {
            _adapterA = Tensor<float>.Zeros(inFeatures, Math.Max(1, rank));
            _adapterB = Tensor<float>.Zeros(Math.Max(1, rank), outFeatures);
            _trainableParameters = [];
        }
    }

    /// <summary>
    /// Executes forward pass: computes base model quantized projection (streamed from unmanaged GGUF memory or GPU VRAM)
    /// and adds the low-rank LoRA delta (X * A) * B * (alpha / r).
    /// </summary>
    public void Forward(GgufType type, byte* weightData, float* bias, Tensor<float> x, Tensor<float> output, int seqLen)
    {
        // 1. Quantized base projection
        if (_gpuEngine != null &&
            _dBaseWeight != IntPtr.Zero &&
            _gpuEngine.SupportsForwardGemm(type))
        {
            _gpuEngine.ForwardProjection(type, _dBaseWeight, _dBaseBias, x, output, _inFeatures, _outFeatures, seqLen);
        }
        else
        {
            QuantKernels.MatMulBatch(type, weightData, (float*)x.DataPointer, (float*)output.DataPointer, _inFeatures, _outFeatures, seqLen);

            if (bias != null)
            {
                float* pOut = (float*)output.DataPointer;
                for (int t = 0; t < seqLen; t++)
                {
                    float* pRow = pOut + (long)t * _outFeatures;
                    for (int d = 0; d < _outFeatures; d++)
                    {
                        pRow[d] += bias[d];
                    }
                }
            }
        }

        // 2. LoRA adapter: output += (alpha / r) * (X * A) * B
        if (!_isEnabled || IsZeroDelta) return;

        if (x.IsContiguous && output.IsContiguous && _adapterA.IsContiguous && _adapterB.IsContiguous)
        {
            LoraKernels.ApplyLoraDelta(
                x.DataPointer,
                _adapterA.DataPointer,
                _adapterB.DataPointer,
                output.DataPointer,
                seqLen,
                _inFeatures,
                _outFeatures,
                _rank,
                _scaling);
        }
        else
        {
            using var lowRank = TensorOps.MatMul(x, _adapterA, _target);
            using var delta = TensorOps.MatMul(lowRank, _adapterB, _target);
            using var scaledDelta = TensorOps.Scale(delta, _scaling);

            var outSpan = output.AsSpan();
            var deltaSpan = scaledDelta.AsSpan();
            for (int i = 0; i < outSpan.Length; i++)
            {
                outSpan[i] += deltaSpan[i];
            }
        }
    }

    /// <summary>
    /// Executes full reverse-mode backpropagation through both the frozen base projection
    /// (dXInput += dY * W_base) and the trainable LoRA adapter (dA, dB, and dXInput += scaling * dY * B^T * A^T).
    /// </summary>
    public void Backward(GgufType baseType, byte* baseWeightData, Tensor<float> dY, Tensor<float> x, Tensor<float>? dXInput = null)
    {
        if (dXInput != null)
        {
            int seqLen = x.Shape[0];
            if (_gpuEngine != null &&
                _dBaseWeight != IntPtr.Zero &&
                _gpuEngine.SupportsBackwardGemm(baseType))
            {
                _gpuEngine.BackwardProjection(
                    baseType, _dBaseWeight, dY, dXInput, _inFeatures, _outFeatures, seqLen);
            }
            else if (baseWeightData != null)
            {
                LoraKernels.AccumulateBaseBackward(
                    baseType,
                    baseWeightData,
                    dY.DataPointer,
                    dXInput.DataPointer,
                    seqLen,
                    _inFeatures,
                    _outFeatures);
            }
        }

        if (_isEnabled)
        {
            Backward(dY, x, dXInput);
        }
    }

    /// <summary>
    /// Computes adapter gradients dA and dB from upstream gradient dY and input X,
    /// and optionally adds adapter gradient contribution (alpha / r) * (dY * B^T) * A^T to dXInput.
    /// </summary>
    public void Backward(Tensor<float> dY, Tensor<float> x, Tensor<float>? dXInput = null)
    {
        if (!_isEnabled) return;

        if (x.IsContiguous && dY.IsContiguous && _adapterA.IsContiguous && _adapterB.IsContiguous &&
            (dXInput == null || dXInput.IsContiguous))
        {
            int seqLen = x.Shape[0];
            float* pDX = dXInput != null ? dXInput.DataPointer : null;
            LoraKernels.Backward(
                x.DataPointer,
                dY.DataPointer,
                _adapterA.DataPointer,
                _adapterB.DataPointer,
                _adapterA.Grad!.DataPointer,
                _adapterB.Grad!.DataPointer,
                pDX,
                seqLen,
                _inFeatures,
                _outFeatures,
                _rank,
                _scaling);
            return;
        }

        // lowRank = X * A  [seqLen, rank]
        using var lowRank = TensorOps.MatMul(x, _adapterA, _target);

        // dB += scaling * lowRank^T * dY  [rank, outFeatures]
        using var lowRankT = lowRank.Transpose();
        using var deltaB = TensorOps.MatMul(lowRankT, dY, _target);
        using var scaledDeltaB = TensorOps.Scale(deltaB, _scaling);

        var bGradSpan = _adapterB.Grad!.AsSpan();
        var deltaBSpan = scaledDeltaB.AsSpan();
        for (int i = 0; i < bGradSpan.Length; i++)
        {
            bGradSpan[i] += deltaBSpan[i];
        }

        // dLowRank = dY * B^T  [seqLen, rank]
        using var bT = _adapterB.Transpose();
        using var dLowRank = TensorOps.MatMul(dY, bT, _target);

        // dA += scaling * X^T * dLowRank  [inFeatures, rank]
        using var inT = x.Transpose();
        using var deltaA = TensorOps.MatMul(inT, dLowRank, _target);
        using var scaledDeltaA = TensorOps.Scale(deltaA, _scaling);

        var aGradSpan = _adapterA.Grad!.AsSpan();
        var deltaASpan = scaledDeltaA.AsSpan();
        for (int i = 0; i < aGradSpan.Length; i++)
        {
            aGradSpan[i] += deltaASpan[i];
        }

        // dX_lora = scaling * dLowRank * A^T  [seqLen, inFeatures]
        if (dXInput != null)
        {
            using var aT = _adapterA.Transpose();
            using var dX_lora = TensorOps.MatMul(dLowRank, aT, _target);
            using var dX_scaled = TensorOps.Scale(dX_lora, _scaling);

            var dXSpan = dXInput.AsSpan();
            var loraSpan = dX_scaled.AsSpan();
            for (int i = 0; i < dXSpan.Length; i++)
            {
                dXSpan[i] += loraSpan[i];
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _gpuEngine = null;
            _adapterA.Dispose();
            _adapterB.Dispose();
        }
    }
}
