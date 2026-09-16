using System;
using System.Collections.Generic;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

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
    private readonly Tensor<float> _adapterA; // [InFeatures, Rank]
    private readonly Tensor<float> _adapterB; // [Rank, OutFeatures]
    private readonly List<Tensor<float>> _trainableParameters;
    private bool _disposed;

    public int InFeatures => _inFeatures;
    public int OutFeatures => _outFeatures;
    public int Rank => _rank;
    public float Scaling => _scaling;
    public Tensor<float> AdapterA => _adapterA;
    public Tensor<float> AdapterB => _adapterB;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;

    public LoraAdapter(int inFeatures, int outFeatures, int rank = 16, float alpha = 32f, int? seed = null)
    {
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;
        _rank = rank;
        _scaling = alpha / rank;

        // Gaussian init std = 1 / sqrt(inFeatures)
        float stdA = 1.0f / MathF.Sqrt(inFeatures);
        _adapterA = TensorFloatExtensions.RandomNormal([inFeatures, rank], mean: 0f, std: stdA, seed: seed ?? 42);

        // Zero init for B so Delta W = 0 at start
        _adapterB = Tensor<float>.Zeros(rank, outFeatures);

        _adapterA.RequiresGrad = true;
        _adapterB.RequiresGrad = true;
        _adapterA.Grad = Tensor<float>.Zeros(inFeatures, rank);
        _adapterB.Grad = Tensor<float>.Zeros(rank, outFeatures);

        _trainableParameters = [_adapterA, _adapterB];
    }

    /// <summary>
    /// Executes forward pass: computes base model quantized projection (streamed from unmanaged GGUF memory)
    /// and adds the low-rank LoRA delta (X * A) * B * (alpha / r).
    /// </summary>
    public void Forward(GgufType type, byte* weightData, float* bias, Tensor<float> x, Tensor<float> output, int seqLen)
    {
        // 1. Quantized base projection: output = BaseWeight * X
        QuantKernels.MatMulBatch(type, weightData, (float*)x.DataPointer, (float*)output.DataPointer, _inFeatures, _outFeatures, seqLen);

        // 2. Add bias if present
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

        // 3. LoRA adapter: output += (alpha / r) * (X * A) * B
        using var lowRank = TensorOps.MatMul(x, _adapterA);
        using var delta = TensorOps.MatMul(lowRank, _adapterB);
        using var scaledDelta = TensorOps.Scale(delta, _scaling);

        var outSpan = output.AsSpan();
        var deltaSpan = scaledDelta.AsSpan();
        for (int i = 0; i < outSpan.Length; i++)
        {
            outSpan[i] += deltaSpan[i];
        }
    }

    /// <summary>
    /// Computes adapter gradients dA and dB from upstream gradient dY and input X,
    /// and optionally adds adapter gradient contribution (alpha / r) * (dY * B^T) * A^T to dXInput.
    /// </summary>
    public void Backward(Tensor<float> dY, Tensor<float> x, Tensor<float>? dXInput = null)
    {
        // lowRank = X * A  [seqLen, rank]
        using var lowRank = TensorOps.MatMul(x, _adapterA);

        // dB += scaling * lowRank^T * dY  [rank, outFeatures]
        using var lowRankT = lowRank.Transpose();
        using var deltaB = TensorOps.MatMul(lowRankT, dY);
        using var scaledDeltaB = TensorOps.Scale(deltaB, _scaling);

        var bGradSpan = _adapterB.Grad!.AsSpan();
        var deltaBSpan = scaledDeltaB.AsSpan();
        for (int i = 0; i < bGradSpan.Length; i++)
        {
            bGradSpan[i] += deltaBSpan[i];
        }

        // dLowRank = dY * B^T  [seqLen, rank]
        using var bT = _adapterB.Transpose();
        using var dLowRank = TensorOps.MatMul(dY, bT);

        // dA += scaling * X^T * dLowRank  [inFeatures, rank]
        using var inT = x.Transpose();
        using var deltaA = TensorOps.MatMul(inT, dLowRank);
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
            using var dX_lora = TensorOps.MatMul(dLowRank, aT);
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
            _adapterA.Dispose();
            _adapterB.Dispose();
        }
    }
}
