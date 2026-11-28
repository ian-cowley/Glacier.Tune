using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Model;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Kernels;

namespace Glacier.Tune.Model;

/// <summary>
/// Intermediate activations saved for a single Gated DeltaNet (GDN) layer during forward pass
/// for exact Backpropagation Through Time (BPTT).
/// </summary>
public sealed class GdnActivationState : IDisposable
{
    public Tensor<float>? Norm1X { get; set; }
    public Tensor<float>? Z { get; set; }
    public Tensor<float>? Alpha { get; set; }
    public Tensor<float>? BetaRaw { get; set; }
    public Tensor<float>? ConvPre { get; set; }
    public Tensor<float>? QNorm { get; set; }
    public Tensor<float>? KNorm { get; set; }
    public Tensor<float>? V { get; set; }
    public Tensor<float>? InvNormQ { get; set; }
    public Tensor<float>? InvNormK { get; set; }
    public Tensor<float>? Decay { get; set; }
    public Tensor<float>? Beta { get; set; }
    public Tensor<float>? YRec { get; set; }
    public Tensor<float>? YNorm { get; set; }
    public Tensor<float>? YGated { get; set; }
    public Tensor<float>? X1 { get; set; }
    public Tensor<float>? Norm2X { get; set; }
    public Tensor<float>? Gate { get; set; }
    public Tensor<float>? Up { get; set; }
    public Tensor<float>? Hidden { get; set; }

    public void Dispose()
    {
        Norm1X?.Dispose();
        Z?.Dispose();
        Alpha?.Dispose();
        BetaRaw?.Dispose();
        ConvPre?.Dispose();
        QNorm?.Dispose();
        KNorm?.Dispose();
        V?.Dispose();
        InvNormQ?.Dispose();
        InvNormK?.Dispose();
        Decay?.Dispose();
        Beta?.Dispose();
        YRec?.Dispose();
        YNorm?.Dispose();
        YGated?.Dispose();
        X1?.Dispose();
        Norm2X?.Dispose();
        Gate?.Dispose();
        Up?.Dispose();
        Hidden?.Dispose();
    }
}

/// <summary>
/// Pure-C# Qwen 3.5 / 3.6 Gated DeltaNet (GDN) hybrid layer block with zero-copy GGUF base weights,
/// trainable LoRA adapters on GDN and SwiGLU FFN projections, and exact parallel per-head BPTT.
/// </summary>
public sealed unsafe class GdnBlock : ITuneBlock
{
    private readonly int _layerIndex;
    private readonly int _hiddenDim;
    private readonly int _ffnDim;
    private readonly float _rmsNormEps;
    private readonly int _convKernel;
    private readonly int _stateDim;
    private readonly int _groupCount;
    private readonly int _ssmHeads;
    private readonly int _ssmInnerSize;
    private readonly int _gdnConvChannels;
    private readonly bool _useSigmoidGate;
    private readonly LayerWeights _weights;

    private readonly LoraAdapter _qkvProj;
    private readonly LoraAdapter _attnGateProj;
    private readonly LoraAdapter _alphaProj;
    private readonly LoraAdapter _betaProj;
    private readonly LoraAdapter _ssmOutProj;
    private readonly LoraAdapter _gateProj;
    private readonly LoraAdapter _upProj;
    private readonly LoraAdapter _downProj;

    private readonly List<Tensor<float>> _trainableParameters = [];
    private bool _disposed;

    public int LayerIndex => _layerIndex;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;

    public LoraAdapter QkvProj => _qkvProj;
    public LoraAdapter AttnGateProj => _attnGateProj;
    public LoraAdapter AlphaProj => _alphaProj;
    public LoraAdapter BetaProj => _betaProj;
    public LoraAdapter SsmOutProj => _ssmOutProj;
    public LoraAdapter GateProj => _gateProj;
    public LoraAdapter UpProj => _upProj;
    public LoraAdapter DownProj => _downProj;

    public GdnBlock(
        int layerIndex,
        LayerWeights weights,
        ModelWeights modelWeights,
        LoraConfig config,
        GpuTarget target = GpuTarget.Auto)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(modelWeights);
        ArgumentNullException.ThrowIfNull(config);

        _layerIndex = layerIndex;
        _weights = weights;
        _hiddenDim = modelWeights.EmbeddingLength;
        _ffnDim = modelWeights.FeedForwardLength;
        _rmsNormEps = modelWeights.RmsNormEps;
        _convKernel = modelWeights.SsmConvKernel;
        _stateDim = modelWeights.SsmStateSize;
        _groupCount = modelWeights.SsmGroupCount;
        _ssmHeads = modelWeights.SsmTimeStepRank;
        _ssmInnerSize = modelWeights.SsmInnerSize > 0 ? modelWeights.SsmInnerSize : _ssmHeads * _stateDim;
        _gdnConvChannels = (_groupCount * 2 + _ssmHeads) * _stateDim;

        if (_convKernel <= 0 || _stateDim <= 0 || _groupCount <= 0 || _ssmHeads <= 0 || _ssmInnerSize <= 0)
        {
            throw new InvalidDataException(
                $"Invalid GDN dimensions on layer {layerIndex}: convKernel={_convKernel}, stateDim={_stateDim}, groupCount={_groupCount}, ssmHeads={_ssmHeads}, innerSize={_ssmInnerSize}.");
        }

        if (_weights.QkvWeight == null || _weights.AttnGateWeight == null ||
            _weights.SsmAlphaWeight == null || _weights.SsmBetaWeight == null ||
            _weights.SsmConv1dWeight == null || _weights.SsmOutWeight == null ||
            _weights.AttnNormWeight == null || _weights.FfnNormWeight == null ||
            _weights.FfnGateWeight == null || _weights.FfnUpWeight == null || _weights.FfnDownWeight == null)
        {
            throw new InvalidDataException($"Missing required GDN or FFN base weight tensor(s) on layer {layerIndex}.");
        }

        long expectedConvElements = (long)_convKernel * _gdnConvChannels;
        if (_weights.SsmConv1dDimensions is { Length: > 0 } convDims)
        {
            bool validConvShape =
                (convDims.Length == 2 && convDims[0] == (ulong)_convKernel && convDims[1] == (ulong)_gdnConvChannels) ||
                (convDims.Length == 1 && convDims[0] == (ulong)expectedConvElements);
            if (!validConvShape)
            {
                throw new InvalidDataException(
                    $"Invalid 'ssm_conv1d.weight' shape [{string.Join(", ", convDims)}] on layer {layerIndex}; " +
                    $"expected [{_convKernel}, {_gdnConvChannels}] (or 1D [{expectedConvElements}]).");
            }
        }

        string arch = modelWeights.Gguf.Architecture;
        _useSigmoidGate = string.Equals(arch, "qwen4exp", StringComparison.OrdinalIgnoreCase) ||
                          string.Equals(arch, "qwen38", StringComparison.OrdinalIgnoreCase) ||
                          modelWeights.HyperConnectionCount > 0;

        int baseSeed = config.Seed ?? 42;

        _qkvProj = new LoraAdapter(_hiddenDim, _gdnConvChannels, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 0), target, HasTarget(config, "in_proj_qkv", "attn_qkv", "qkv_proj"));
        _attnGateProj = new LoraAdapter(_hiddenDim, _ssmInnerSize, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 1), target, HasTarget(config, "in_proj_z", "attn_gate"));
        _alphaProj = new LoraAdapter(_hiddenDim, _ssmHeads, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 2), target, HasTarget(config, "in_proj_a", "ssm_alpha", "alpha_proj"));
        _betaProj = new LoraAdapter(_hiddenDim, _ssmHeads, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 3), target, HasTarget(config, "in_proj_b", "ssm_beta", "beta_proj"));
        _ssmOutProj = new LoraAdapter(_ssmInnerSize, _hiddenDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 4), target, HasTarget(config, "out_proj", "ssm_out"));

        _gateProj = new LoraAdapter(_hiddenDim, _ffnDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 5), target, HasTarget(config, "gate_proj", "ffn_gate"));
        _upProj = new LoraAdapter(_hiddenDim, _ffnDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 6), target, HasTarget(config, "up_proj", "ffn_up"));
        _downProj = new LoraAdapter(_ffnDim, _hiddenDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 7), target, HasTarget(config, "down_proj", "ffn_down"));

        RegisterAdapter(_qkvProj);
        RegisterAdapter(_attnGateProj);
        RegisterAdapter(_alphaProj);
        RegisterAdapter(_betaProj);
        RegisterAdapter(_ssmOutProj);
        RegisterAdapter(_gateProj);
        RegisterAdapter(_upProj);
        RegisterAdapter(_downProj);
    }

    private static int DeriveSeed(int baseSeed, int layerIndex, int projOrdinal) =>
        unchecked(baseSeed + layerIndex * 1009 + projOrdinal * 9176);

    private static bool HasTarget(LoraConfig config, string primaryName, string ggufAlias, string? altAlias = null)
    {
        if (config.TargetModules == null || config.TargetModules.Length == 0)
        {
            return true;
        }

        foreach (string module in config.TargetModules)
        {
            if (string.Equals(module, primaryName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(module, ggufAlias, StringComparison.OrdinalIgnoreCase) ||
                (altAlias != null && string.Equals(module, altAlias, StringComparison.OrdinalIgnoreCase)) ||
                string.Equals(module, "all", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(module, "all-linear", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private void RegisterAdapter(LoraAdapter lora)
    {
        _trainableParameters.AddRange(lora.TrainableParameters);
    }

    public void ConnectGpuWeights(Glacier.Tune.Gpu.GpuLoraEngine engine)
    {
        // GDN layers execute in pure-C# CPU mode; no GPU base weight pointers are bound.
    }

    Tensor<float> ITuneBlock.Forward(Tensor<float> x, int seqLen, bool saveActivations, out IDisposable? state)
    {
        var output = Forward(x, seqLen, saveActivations, out GdnActivationState? gdnState);
        state = gdnState;
        return output;
    }

    Tensor<float> ITuneBlock.Backward(Tensor<float> dOutput, IDisposable state, Tensor<float> xInput, int seqLen, AutogradScratchWorkspace? workspace)
    {
        if (state is not GdnActivationState gdnState)
        {
            throw new ArgumentException($"Expected {nameof(GdnActivationState)} for layer {_layerIndex}.", nameof(state));
        }
        return Backward(dOutput, gdnState, xInput, seqLen, workspace);
    }

    internal void UpdateAlpha(float alpha)
    {
        _qkvProj.UpdateAlpha(alpha);
        _attnGateProj.UpdateAlpha(alpha);
        _alphaProj.UpdateAlpha(alpha);
        _betaProj.UpdateAlpha(alpha);
        _ssmOutProj.UpdateAlpha(alpha);
        _gateProj.UpdateAlpha(alpha);
        _upProj.UpdateAlpha(alpha);
        _downProj.UpdateAlpha(alpha);
    }

    public LoraAdapter? FindAdapter(string projectionName)
    {
        return projectionName switch
        {
            "attn_qkv" or "in_proj_qkv" or "qkv_proj" => _qkvProj,
            "attn_gate" or "in_proj_z" => _attnGateProj,
            "ssm_alpha" or "in_proj_a" or "alpha_proj" => _alphaProj,
            "ssm_beta" or "in_proj_b" or "beta_proj" => _betaProj,
            "ssm_out" or "out_proj" or "attn_output" => _ssmOutProj,
            "ffn_gate" or "gate_proj" => _gateProj,
            "ffn_up" or "up_proj" => _upProj,
            "ffn_down" or "down_proj" => _downProj,
            _ => null
        };
    }

    public int CountEnabledTensors()
    {
        int count = 0;
        if (_qkvProj.IsEnabled) count += 2;
        if (_attnGateProj.IsEnabled) count += 2;
        if (_alphaProj.IsEnabled) count += 2;
        if (_betaProj.IsEnabled) count += 2;
        if (_ssmOutProj.IsEnabled) count += 2;
        if (_gateProj.IsEnabled) count += 2;
        if (_upProj.IsEnabled) count += 2;
        if (_downProj.IsEnabled) count += 2;
        return count;
    }

    public void WriteAdapterTensors(BinaryWriter bw)
    {
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "attn_qkv", _qkvProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "attn_gate", _attnGateProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ssm_alpha", _alphaProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ssm_beta", _betaProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ssm_out", _ssmOutProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ffn_gate", _gateProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ffn_up", _upProj);
        TuneBlockSerialization.WriteAdapterPair(bw, _layerIndex, "ffn_down", _downProj);
    }

    public void WriteManifestEntries(StreamWriter writer)
    {
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "attn_qkv", _qkvProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "attn_gate", _attnGateProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "ssm_alpha", _alphaProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "ssm_beta", _betaProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "ssm_out", _ssmOutProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "gate_proj", _gateProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "up_proj", _upProj);
        TuneBlockSerialization.WriteManifestEntry(writer, _layerIndex, "down_proj", _downProj);
    }

    public Tensor<float> Forward(Tensor<float> x, int seqLen, bool saveActivations, out GdnActivationState? state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        state = saveActivations ? new GdnActivationState() : null;

        // 1. Pre-RMSNorm: norm1X = RMSNorm(x, attn_norm)
        var norm1X = new Tensor<float>(seqLen, _hiddenDim);
        using (var wNorm1 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.AttnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNorm(x, wNorm1, norm1X, _rmsNormEps);
        }
        if (state != null) state.Norm1X = norm1X;

        // 2. Linear projections (+ LoRA): QKV, Z, Alpha, Beta
        using var qkv = new Tensor<float>(seqLen, _gdnConvChannels);
        var z = new Tensor<float>(seqLen, _ssmInnerSize);
        var alpha = new Tensor<float>(seqLen, _ssmHeads);
        var betaRaw = new Tensor<float>(seqLen, _ssmHeads);

        Parallel.Invoke(
            () => _qkvProj.Forward(_weights.QkvType, _weights.QkvWeight, null, norm1X, qkv, seqLen),
            () => _attnGateProj.Forward(_weights.AttnGateType, _weights.AttnGateWeight, null, norm1X, z, seqLen),
            () => _alphaProj.Forward(_weights.SsmAlphaType, _weights.SsmAlphaWeight, null, norm1X, alpha, seqLen),
            () => _betaProj.Forward(_weights.SsmBetaType, _weights.SsmBetaWeight, null, norm1X, betaRaw, seqLen)
        );

        if (state != null)
        {
            state.Z = z;
            state.Alpha = alpha;
            state.BetaRaw = betaRaw;
        }
        else
        {
            norm1X.Dispose();
        }

        // 3. Causal depthwise 1D convolution + SiLU
        var convPre = state != null ? new Tensor<float>(seqLen, _gdnConvChannels) : null;
        using var convOut = new Tensor<float>(seqLen, _gdnConvChannels);
        float* pQkv = (float*)qkv.DataPointer;
        float* pConvPre = convPre != null ? (float*)convPre.DataPointer : null;
        float* pConvOut = (float*)convOut.DataPointer;
        float* pConvW = _weights.SsmConv1dWeight;
        float* pConvB = _weights.SsmConv1dBias;

        Parallel.For(0, _gdnConvChannels, c =>
        {
            float* w = pConvW + (long)c * _convKernel;
            float b = pConvB != null ? pConvB[c] : 0f;
            for (int t = 0; t < seqLen; t++)
            {
                float acc = b;
                for (int k = 0; k < _convKernel; k++)
                {
                    int srcT = t - (_convKernel - 1 - k);
                    if (srcT >= 0)
                    {
                        acc += pQkv[(long)srcT * _gdnConvChannels + c] * w[k];
                    }
                }
                if (pConvPre != null)
                {
                    pConvPre[(long)t * _gdnConvChannels + c] = acc;
                }
                float sig = 1.0f / (1.0f + MathF.Exp(-acc));
                pConvOut[(long)t * _gdnConvChannels + c] = acc * sig;
            }
        });
        if (state != null) state.ConvPre = convPre;

        // 4. Split Q, K, V and apply per-group L2 normalization
        int qkGroupFloats = _groupCount * _stateDim;
        var qNorm = new Tensor<float>(seqLen, qkGroupFloats);
        var kNorm = new Tensor<float>(seqLen, qkGroupFloats);
        var v = new Tensor<float>(seqLen, _ssmInnerSize);
        var invNormQ = state != null ? new Tensor<float>(seqLen, _groupCount) : null;
        var invNormK = state != null ? new Tensor<float>(seqLen, _groupCount) : null;

        float* pQNorm = (float*)qNorm.DataPointer;
        float* pKNorm = (float*)kNorm.DataPointer;
        float* pV = (float*)v.DataPointer;
        float* pInvQ = invNormQ != null ? (float*)invNormQ.DataPointer : null;
        float* pInvK = invNormK != null ? (float*)invNormK.DataPointer : null;
        float qkScale = 1.0f / MathF.Sqrt(_stateDim);
        ulong vRowBytes = (ulong)(_ssmInnerSize * sizeof(float));

        for (int t = 0; t < seqLen; t++)
        {
            float* rowOut = pConvOut + (long)t * _gdnConvChannels;
            float* qSrc = rowOut;
            float* kSrc = rowOut + qkGroupFloats;
            float* vSrc = rowOut + qkGroupFloats * 2;
            Buffer.MemoryCopy(vSrc, pV + (long)t * _ssmInnerSize, vRowBytes, vRowBytes);

            for (int g = 0; g < _groupCount; g++)
            {
                float* qgSrc = qSrc + g * _stateDim;
                float* kgSrc = kSrc + g * _stateDim;
                float* qgDst = pQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* kgDst = pKNorm + ((long)t * _groupCount + g) * _stateDim;

                float sumSqQ = 0f;
                float sumSqK = 0f;
                for (int d = 0; d < _stateDim; d++)
                {
                    float qVal = qgSrc[d];
                    float kVal = kgSrc[d];
                    sumSqQ += qVal * qVal;
                    sumSqK += kVal * kVal;
                }

                float invQ = sumSqQ > 1e-12f ? (1.0f / MathF.Sqrt(sumSqQ + 1e-6f)) * qkScale : 0f;
                float invK = sumSqK > 1e-12f ? 1.0f / MathF.Sqrt(sumSqK + 1e-6f) : 0f;
                if (pInvQ != null) pInvQ[(long)t * _groupCount + g] = invQ;
                if (pInvK != null) pInvK[(long)t * _groupCount + g] = invK;

                for (int d = 0; d < _stateDim; d++)
                {
                    qgDst[d] = qgSrc[d] * invQ;
                    kgDst[d] = kgSrc[d] * invK;
                }
            }
        }

        if (state != null)
        {
            state.QNorm = qNorm;
            state.KNorm = kNorm;
            state.V = v;
            state.InvNormQ = invNormQ;
            state.InvNormK = invNormK;
        }

        // 5. Parallel per-head DeltaNet recurrence
        var yRec = new Tensor<float>(seqLen, _ssmInnerSize);
        var decayTensor = state != null ? new Tensor<float>(seqLen, _ssmHeads) : null;
        var betaTensor = state != null ? new Tensor<float>(seqLen, _ssmHeads) : null;

        float* pAlpha = (float*)alpha.DataPointer;
        float* pBetaRaw = (float*)betaRaw.DataPointer;
        float* pYRec = (float*)yRec.DataPointer;
        float* pDecay = decayTensor != null ? (float*)decayTensor.DataPointer : null;
        float* pBeta = betaTensor != null ? (float*)betaTensor.DataPointer : null;
        float* pSsmA = _weights.SsmAWeight;
        float* pDtBias = _weights.SsmDtBias;

        var gdnWs = GdnScratchWorkspace.GetOrCreateForForward(_ssmHeads, _stateDim);

        Parallel.For(0, _ssmHeads, h =>
        {
            int g = h % _groupCount;
            float aLog = pSsmA != null ? pSsmA[h] : 0f;
            float dtBias = pDtBias != null ? pDtBias[h] : 0f;
            gdnWs.GetForwardHeadScratch(h, _stateDim, out float* sMat, out float* delta);

            for (int t = 0; t < seqLen; t++)
            {
                float aVal = pAlpha[(long)t * _ssmHeads + h] + dtBias;
                float dt = aVal > 20.0f ? aVal : MathF.Log(1.0f + MathF.Exp(aVal));
                float decay = MathF.Exp(aLog * dt);
                float beta = 1.0f / (1.0f + MathF.Exp(-pBetaRaw[(long)t * _ssmHeads + h]));

                if (pDecay != null) pDecay[(long)t * _ssmHeads + h] = decay;
                if (pBeta != null) pBeta[(long)t * _ssmHeads + h] = beta;

                float* qHead = pQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* kHead = pKNorm + ((long)t * _groupCount + g) * _stateDim;
                float* vHead = pV + ((long)t * _ssmHeads + h) * _stateDim;
                float* yHead = pYRec + ((long)t * _ssmHeads + h) * _stateDim;

                for (int i = 0; i < _stateDim; i++)
                {
                    float* sRow = sMat + (long)i * _stateDim;
                    float kvMem = 0f;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        float decayed = sRow[j] * decay;
                        sRow[j] = decayed;
                        kvMem += decayed * kHead[j];
                    }
                    delta[i] = (vHead[i] - kvMem) * beta;
                }

                for (int i = 0; i < _stateDim; i++)
                {
                    float* sRow = sMat + (long)i * _stateDim;
                    float dVal = delta[i];
                    float dot = 0f;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        float updated = sRow[j] + dVal * kHead[j];
                        sRow[j] = updated;
                        dot += updated * qHead[j];
                    }
                    yHead[i] = dot;
                }
            }
        });

        if (state != null)
        {
            state.Decay = decayTensor;
            state.Beta = betaTensor;
            state.YRec = yRec;
        }
        else
        {
            alpha.Dispose();
            betaRaw.Dispose();
            qNorm.Dispose();
            kNorm.Dispose();
            v.Dispose();
        }

        // 6. Per-head RMSNorm (ssm_norm)
        var yNorm = new Tensor<float>(seqLen, _ssmInnerSize);
        if (_weights.SsmNormWeight != null)
        {
            using var yRecView = yRec.Reshape(seqLen * _ssmHeads, _stateDim);
            using var yNormView = yNorm.Reshape(seqLen * _ssmHeads, _stateDim);
            using var wSsmNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.SsmNormWeight, _stateDim), [_stateDim]);
            ElementwiseKernels.RMSNorm(yRecView, wSsmNorm, yNormView, _rmsNormEps);
        }
        else
        {
            Buffer.MemoryCopy((float*)yRec.DataPointer, (float*)yNorm.DataPointer,
                (ulong)((long)seqLen * _ssmInnerSize * sizeof(float)),
                (ulong)((long)seqLen * _ssmInnerSize * sizeof(float)));
        }

        if (state != null)
        {
            state.YNorm = yNorm;
        }
        else
        {
            yRec.Dispose();
        }

        // 7. Output gating with z
        var yGated = new Tensor<float>(seqLen, _ssmInnerSize);
        float* pYNorm = (float*)yNorm.DataPointer;
        float* pZ = (float*)z.DataPointer;
        float* pYGated = (float*)yGated.DataPointer;
        int totalInner = seqLen * _ssmInnerSize;

        if (_useSigmoidGate)
        {
            for (int i = 0; i < totalInner; i++)
            {
                float sig = 1.0f / (1.0f + MathF.Exp(-pZ[i]));
                pYGated[i] = pYNorm[i] * sig;
            }
        }
        else
        {
            for (int i = 0; i < totalInner; i++)
            {
                float zv = pZ[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-zv));
                pYGated[i] = pYNorm[i] * (zv * sig);
            }
        }

        if (state != null)
        {
            state.YGated = yGated;
        }
        else
        {
            yNorm.Dispose();
            z.Dispose();
        }

        // 8. Output projection + Residual
        var ssmProj = new Tensor<float>(seqLen, _hiddenDim);
        _ssmOutProj.Forward(_weights.SsmOutType, _weights.SsmOutWeight, null, yGated, ssmProj, seqLen);
        var x1 = TensorOps.Add(x, ssmProj);
        ssmProj.Dispose();

        if (state != null)
        {
            state.X1 = x1;
        }
        else
        {
            yGated.Dispose();
        }

        // 9. Dense SwiGLU FFN
        var norm2X = new Tensor<float>(seqLen, _hiddenDim);
        using (var wNorm2 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.FfnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNorm(x1, wNorm2, norm2X, _rmsNormEps);
        }

        var gate = new Tensor<float>(seqLen, _ffnDim);
        var up = new Tensor<float>(seqLen, _ffnDim);
        var hidden = new Tensor<float>(seqLen, _ffnDim);
        var downOut = new Tensor<float>(seqLen, _hiddenDim);

        Parallel.Invoke(
            () => _gateProj.Forward(_weights.FfnGateType, _weights.FfnGateWeight, null, norm2X, gate, seqLen),
            () => _upProj.Forward(_weights.FfnUpType, _weights.FfnUpWeight, null, norm2X, up, seqLen)
        );

        SwiGluKernel.Forward(
            (float*)gate.DataPointer, (float*)up.DataPointer,
            (float*)hidden.DataPointer, seqLen * _ffnDim);

        _downProj.Forward(_weights.FfnDownType, _weights.FfnDownWeight, null, hidden, downOut, seqLen);

        var output = TensorOps.Add(x1, downOut);
        downOut.Dispose();

        if (state != null)
        {
            state.Norm2X = norm2X;
            state.Gate = gate;
            state.Up = up;
            state.Hidden = hidden;
        }
        else
        {
            x1.Dispose();
            norm2X.Dispose();
            gate.Dispose();
            up.Dispose();
            hidden.Dispose();
        }

        return output;
    }

    public Tensor<float> Backward(
        Tensor<float> dOutput,
        GdnActivationState state,
        Tensor<float> xInput,
        int seqLen,
        AutogradScratchWorkspace? workspace = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ws = workspace ?? AutogradScratchWorkspace.GetOrCreate(seqLen, _hiddenDim, _ffnDim, 1, 1, _stateDim);

        // -------------------------------------------------------------
        // A. FFN Backward Pass
        // -------------------------------------------------------------
        using var dHidden = AutogradScratchWorkspace.Wrap(ws.FfnScratchA, seqLen, _ffnDim);
        _downProj.Backward(_weights.FfnDownType, _weights.FfnDownWeight, dOutput, state.Hidden!, dHidden);

        using var dGate = AutogradScratchWorkspace.Wrap(ws.FfnScratchB, seqLen, _ffnDim);
        using var dUp = AutogradScratchWorkspace.Wrap(ws.FfnScratchC, seqLen, _ffnDim);
        SwiGluKernel.Backward(
            (float*)dHidden.DataPointer,
            (float*)state.Gate!.DataPointer,
            (float*)state.Up!.DataPointer,
            (float*)dGate.DataPointer,
            (float*)dUp.DataPointer,
            seqLen * _ffnDim);

        using var dNorm2X_Gate = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, seqLen, _hiddenDim);
        using var dNorm2X_Up = AutogradScratchWorkspace.Wrap(ws.HiddenScratchB, seqLen, _hiddenDim);
        Parallel.Invoke(
            () => _gateProj.Backward(_weights.FfnGateType, _weights.FfnGateWeight, dGate, state.Norm2X!, dNorm2X_Gate),
            () => _upProj.Backward(_weights.FfnUpType, _weights.FfnUpWeight, dUp, state.Norm2X!, dNorm2X_Up)
        );
        using var dNorm2X = AutogradScratchWorkspace.Wrap(ws.HiddenScratchC, seqLen, _hiddenDim);
        ElementwiseKernels.Add(dNorm2X_Gate, dNorm2X_Up, dNorm2X);

        using var dX1_norm = AutogradScratchWorkspace.Wrap(ws.HiddenScratchD, seqLen, _hiddenDim);
        using (var wNorm2 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.FfnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dNorm2X, state.X1!, wNorm2, dX1_norm, null, _rmsNormEps);
        }

        using var dX1 = TensorOps.Add(dOutput, dX1_norm);

        // -------------------------------------------------------------
        // B. GDN Backward Pass (BPTT via pre-allocated GdnScratchWorkspace)
        // -------------------------------------------------------------
        int qkGroupFloats = _groupCount * _stateDim;
        var gdnWs = GdnScratchWorkspace.GetOrCreateForBackward(
            _ssmHeads, _stateDim, seqLen, _ssmInnerSize, qkGroupFloats, _gdnConvChannels);
        gdnWs.GetBackwardActivationBuffers(
            out float* pDYGatedBuf,
            out float* pDYNormBuf,
            out float* pDZBuf,
            out float* pDYRecBuf,
            out float* pDVBuf,
            out float* pDQPerHeadBuf,
            out float* pDKPerHeadBuf,
            out float* pDQNormBuf,
            out float* pDKNormBuf,
            out float* pDAlphaBuf,
            out float* pDBetaRawBuf,
            out float* pDConvOutBuf,
            out float* pDConvPreBuf,
            out float* pDQkvBuf);

        // 1. ssm_out backward
        using var dYGated = AutogradScratchWorkspace.Wrap(pDYGatedBuf, seqLen, _ssmInnerSize);
        _ssmOutProj.Backward(_weights.SsmOutType, _weights.SsmOutWeight, dX1, state.YGated!, dYGated);

        // 2. Output gate backward -> dYNorm, dZ
        using var dYNorm = AutogradScratchWorkspace.Wrap(pDYNormBuf, seqLen, _ssmInnerSize);
        using var dZ = AutogradScratchWorkspace.Wrap(pDZBuf, seqLen, _ssmInnerSize);
        float* pDYGated = (float*)dYGated.DataPointer;
        float* pYNorm = (float*)state.YNorm!.DataPointer;
        float* pZ = (float*)state.Z!.DataPointer;
        float* pDYNorm = (float*)dYNorm.DataPointer;
        float* pDZ = (float*)dZ.DataPointer;
        int totalInner = seqLen * _ssmInnerSize;

        if (_useSigmoidGate)
        {
            for (int i = 0; i < totalInner; i++)
            {
                float sig = 1.0f / (1.0f + MathF.Exp(-pZ[i]));
                float dy = pDYGated[i];
                pDYNorm[i] = dy * sig;
                pDZ[i] = dy * pYNorm[i] * sig * (1.0f - sig);
            }
        }
        else
        {
            for (int i = 0; i < totalInner; i++)
            {
                float zv = pZ[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-zv));
                float dy = pDYGated[i];
                pDYNorm[i] = dy * (zv * sig);
                pDZ[i] = dy * pYNorm[i] * sig * (1.0f + zv * (1.0f - sig));
            }
        }

        // 3. Per-head ssm_norm RMSNorm backward -> dYRec
        using var dYRec = AutogradScratchWorkspace.Wrap(pDYRecBuf, seqLen, _ssmInnerSize);
        if (_weights.SsmNormWeight != null)
        {
            using var dYNormView = dYNorm.Reshape(seqLen * _ssmHeads, _stateDim);
            using var yRecView = state.YRec!.Reshape(seqLen * _ssmHeads, _stateDim);
            using var dYRecView = dYRec.Reshape(seqLen * _ssmHeads, _stateDim);
            using var wSsmNorm = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.SsmNormWeight, _stateDim), [_stateDim]);
            ElementwiseKernels.RMSNormBackward(dYNormView, yRecView, wSsmNorm, dYRecView, null, _rmsNormEps);
        }
        else
        {
            Buffer.MemoryCopy(pDYNorm, (float*)dYRec.DataPointer,
                (ulong)((long)totalInner * sizeof(float)),
                (ulong)((long)totalInner * sizeof(float)));
        }

        // 4. Parallel per-head BPTT across DeltaNet recurrence
        using var dQPerHead = AutogradScratchWorkspace.Wrap(pDQPerHeadBuf, _ssmHeads, seqLen, _stateDim);
        using var dKPerHead = AutogradScratchWorkspace.Wrap(pDKPerHeadBuf, _ssmHeads, seqLen, _stateDim);
        using var dV = AutogradScratchWorkspace.Wrap(pDVBuf, seqLen, _ssmInnerSize);
        using var dAlpha = AutogradScratchWorkspace.Wrap(pDAlphaBuf, seqLen, _ssmHeads);
        using var dBetaRaw = AutogradScratchWorkspace.Wrap(pDBetaRawBuf, seqLen, _ssmHeads);

        float* pQNorm = (float*)state.QNorm!.DataPointer;
        float* pKNorm = (float*)state.KNorm!.DataPointer;
        float* pV = (float*)state.V!.DataPointer;
        float* pAlpha = (float*)state.Alpha!.DataPointer;
        float* pDecay = (float*)state.Decay!.DataPointer;
        float* pBeta = (float*)state.Beta!.DataPointer;
        float* pDYRec = (float*)dYRec.DataPointer;
        float* pDQPerHead = (float*)dQPerHead.DataPointer;
        float* pDKPerHead = (float*)dKPerHead.DataPointer;
        float* pDV = (float*)dV.DataPointer;
        float* pDAlpha = (float*)dAlpha.DataPointer;
        float* pDBetaRaw = (float*)dBetaRaw.DataPointer;
        float* pSsmA = _weights.SsmAWeight;
        float* pDtBias = _weights.SsmDtBias;

        Parallel.For(0, _ssmHeads, h =>
        {
            int g = h % _groupCount;
            int matElems = _stateDim * _stateDim;
            float aLog = pSsmA != null ? pSsmA[h] : 0f;
            float dtBias = pDtBias != null ? pDtBias[h] : 0f;

            gdnWs.GetBackwardHeadScratch(
                h,
                _stateDim,
                out float* sPrevHist,
                out float* mHist,
                out float* deltaHist,
                out float* sCur,
                out float* sPrime,
                out float* sMat,
                out float* dS,
                out float* dSTot,
                out float* dDelta);

            // Replay forward state trajectory S_{t-1}, m_t, delta_t for head h
            for (int t = 0; t < seqLen; t++)
            {
                Buffer.MemoryCopy(sCur, sPrevHist + (long)t * matElems, (ulong)(matElems * sizeof(float)), (ulong)(matElems * sizeof(float)));

                float decay = pDecay[(long)t * _ssmHeads + h];
                float beta = pBeta[(long)t * _ssmHeads + h];
                float* kHead = pKNorm + ((long)t * _groupCount + g) * _stateDim;
                float* vHead = pV + ((long)t * _ssmHeads + h) * _stateDim;
                float* mT = mHist + (long)t * _stateDim;
                float* deltaT = deltaHist + (long)t * _stateDim;

                for (int i = 0; i < _stateDim; i++)
                {
                    float* sRow = sCur + (long)i * _stateDim;
                    float kvMem = 0f;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        float decayed = sRow[j] * decay;
                        sRow[j] = decayed;
                        kvMem += decayed * kHead[j];
                    }
                    mT[i] = kvMem;
                    float dVal = (vHead[i] - kvMem) * beta;
                    deltaT[i] = dVal;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        sRow[j] += dVal * kHead[j];
                    }
                }
            }

            // Reverse-time BPTT
            for (int t = seqLen - 1; t >= 0; t--)
            {
                float decay = pDecay[(long)t * _ssmHeads + h];
                float beta = pBeta[(long)t * _ssmHeads + h];
                float* sPrevT = sPrevHist + (long)t * matElems;
                float* mT = mHist + (long)t * _stateDim;
                float* deltaT = deltaHist + (long)t * _stateDim;

                float* qHead = pQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* kHead = pKNorm + ((long)t * _groupCount + g) * _stateDim;
                float* vHead = pV + ((long)t * _ssmHeads + h) * _stateDim;
                float* dy = pDYRec + ((long)t * _ssmHeads + h) * _stateDim;

                float* dqOut = pDQPerHead + ((long)h * seqLen + t) * _stateDim;
                float* dkOut = pDKPerHead + ((long)h * seqLen + t) * _stateDim;
                float* dvOut = pDV + ((long)t * _ssmHeads + h) * _stateDim;

                // Reconstruct S' = decay * S_{t-1} and S_t = S' + delta * k^T
                for (int i = 0; i < _stateDim; i++)
                {
                    float dVal = deltaT[i];
                    int rowOff = i * _stateDim;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        float sp = decay * sPrevT[rowOff + j];
                        sPrime[rowOff + j] = sp;
                        sMat[rowOff + j] = sp + dVal * kHead[j];
                    }
                }

                // Readout y = S_t * q -> dQ and dSTot = dS + dy * q^T
                for (int j = 0; j < _stateDim; j++) dqOut[j] = 0f;
                for (int i = 0; i < _stateDim; i++)
                {
                    float dyi = dy[i];
                    int rowOff = i * _stateDim;
                    for (int j = 0; j < _stateDim; j++)
                    {
                        dqOut[j] += sMat[rowOff + j] * dyi;
                        dSTot[rowOff + j] = dS[rowOff + j] + dyi * qHead[j];
                    }
                }

                // State update S_t = S' + delta * k^T -> dDelta, dK part 1
                for (int j = 0; j < _stateDim; j++) dkOut[j] = 0f;
                for (int i = 0; i < _stateDim; i++)
                {
                    int rowOff = i * _stateDim;
                    float dd = 0f;
                    float dVal = deltaT[i];
                    for (int j = 0; j < _stateDim; j++)
                    {
                        float dst = dSTot[rowOff + j];
                        dd += dst * kHead[j];
                        dkOut[j] += dst * dVal;
                    }
                    dDelta[i] = dd;
                }

                // Delta = beta * (v - m) -> dV, dM = -dV, dBetaRaw
                float dBeta = 0f;
                for (int i = 0; i < _stateDim; i++)
                {
                    float dd = dDelta[i];
                    dvOut[i] = beta * dd;
                    dBeta += dd * (vHead[i] - mT[i]);
                }
                pDBetaRaw[(long)t * _ssmHeads + h] = dBeta * beta * (1.0f - beta);

                // Retrieval m = S' * k, dSPrime = dSTot - dV * k^T, dK += S'^T * (-dV)
                float dDecay = 0f;
                for (int i = 0; i < _stateDim; i++)
                {
                    int rowOff = i * _stateDim;
                    float dm = -dvOut[i];
                    for (int j = 0; j < _stateDim; j++)
                    {
                        dkOut[j] += sPrime[rowOff + j] * dm;
                        float dsp = dSTot[rowOff + j] + dm * kHead[j];
                        dDecay += dsp * sPrevT[rowOff + j];
                        dS[rowOff + j] = decay * dsp;
                    }
                }

                float aRaw = pAlpha[(long)t * _ssmHeads + h] + dtBias;
                float dtGrad = dDecay * decay * aLog;
                float sigA = aRaw > 20.0f ? 1.0f : 1.0f / (1.0f + MathF.Exp(-aRaw));
                pDAlpha[(long)t * _ssmHeads + h] = dtGrad * sigA;
            }
        });

        // Reduce per-head dQ, dK into [seqLen, G * d]
        using var dQNorm = AutogradScratchWorkspace.Wrap(pDQNormBuf, seqLen, qkGroupFloats);
        using var dKNorm = AutogradScratchWorkspace.Wrap(pDKNormBuf, seqLen, qkGroupFloats);
        float* pDQNorm = (float*)dQNorm.DataPointer;
        float* pDKNorm = (float*)dKNorm.DataPointer;

        for (int h = 0; h < _ssmHeads; h++)
        {
            int g = h % _groupCount;
            for (int t = 0; t < seqLen; t++)
            {
                float* srcQ = pDQPerHead + ((long)h * seqLen + t) * _stateDim;
                float* srcK = pDKPerHead + ((long)h * seqLen + t) * _stateDim;
                float* dstQ = pDQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* dstK = pDKNorm + ((long)t * _groupCount + g) * _stateDim;
                for (int d = 0; d < _stateDim; d++)
                {
                    dstQ[d] += srcQ[d];
                    dstK[d] += srcK[d];
                }
            }
        }

        // 5. Group L2 norm backward -> dConvOut [seqLen, C]
        using var dConvOut = AutogradScratchWorkspace.Wrap(pDConvOutBuf, seqLen, _gdnConvChannels);
        float* pDConvOut = (float*)dConvOut.DataPointer;
        float* pInvQ = (float*)state.InvNormQ!.DataPointer;
        float* pInvK = (float*)state.InvNormK!.DataPointer;
        ulong vRowBytes = (ulong)(_ssmInnerSize * sizeof(float));

        for (int t = 0; t < seqLen; t++)
        {
            float* rowDst = pDConvOut + (long)t * _gdnConvChannels;
            float* qHatDst = rowDst;
            float* kHatDst = rowDst + qkGroupFloats;
            float* vDst = rowDst + qkGroupFloats * 2;
            Buffer.MemoryCopy(pDV + (long)t * _ssmInnerSize, vDst, vRowBytes, vRowBytes);

            for (int g = 0; g < _groupCount; g++)
            {
                float gammaQ = pInvQ[(long)t * _groupCount + g];
                float* q = pQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* dq = pDQNorm + ((long)t * _groupCount + g) * _stateDim;
                float* dqHat = qHatDst + g * _stateDim;
                if (gammaQ > 0f)
                {
                    float dotQ = 0f;
                    for (int d = 0; d < _stateDim; d++) dotQ += dq[d] * q[d];
                    float scaleDot = _stateDim * dotQ;
                    for (int d = 0; d < _stateDim; d++) dqHat[d] = gammaQ * (dq[d] - scaleDot * q[d]);
                }
                else
                {
                    new Span<float>(dqHat, _stateDim).Clear();
                }

                float gammaK = pInvK[(long)t * _groupCount + g];
                float* k = pKNorm + ((long)t * _groupCount + g) * _stateDim;
                float* dk = pDKNorm + ((long)t * _groupCount + g) * _stateDim;
                float* dkHat = kHatDst + g * _stateDim;
                if (gammaK > 0f)
                {
                    float dotK = 0f;
                    for (int d = 0; d < _stateDim; d++) dotK += dk[d] * k[d];
                    for (int d = 0; d < _stateDim; d++) dkHat[d] = gammaK * (dk[d] - dotK * k[d]);
                }
                else
                {
                    new Span<float>(dkHat, _stateDim).Clear();
                }
            }
        }

        // 6. SiLU + Causal Conv1d backward -> dQkv [seqLen, C]
        using var dConvPre = AutogradScratchWorkspace.Wrap(pDConvPreBuf, seqLen, _gdnConvChannels);
        using var dQkv = AutogradScratchWorkspace.Wrap(pDQkvBuf, seqLen, _gdnConvChannels);
        float* pConvPre = (float*)state.ConvPre!.DataPointer;
        float* pDConvPre = (float*)dConvPre.DataPointer;
        float* pDQkv = (float*)dQkv.DataPointer;
        float* pConvW = _weights.SsmConv1dWeight;

        Parallel.For(0, _gdnConvChannels, c =>
        {
            for (int t = 0; t < seqLen; t++)
            {
                float a = pConvPre[(long)t * _gdnConvChannels + c];
                float sig = 1.0f / (1.0f + MathF.Exp(-a));
                float dSilu = sig * (1.0f + a * (1.0f - sig));
                pDConvPre[(long)t * _gdnConvChannels + c] = pDConvOut[(long)t * _gdnConvChannels + c] * dSilu;
            }

            float* w = pConvW + (long)c * _convKernel;
            for (int t = 0; t < seqLen; t++)
            {
                int maxM = Math.Min(_convKernel - 1, seqLen - 1 - t);
                float sum = 0f;
                for (int m = 0; m <= maxM; m++)
                {
                    sum += pDConvPre[(long)(t + m) * _gdnConvChannels + c] * w[_convKernel - 1 - m];
                }
                pDQkv[(long)t * _gdnConvChannels + c] = sum;
            }
        });

        // 7. Input projections backward -> dNorm1X
        using var dNorm1X_Qkv = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, seqLen, _hiddenDim);
        using var dNorm1X_Z = AutogradScratchWorkspace.Wrap(ws.HiddenScratchB, seqLen, _hiddenDim);
        using var dNorm1X_A = AutogradScratchWorkspace.Wrap(ws.HiddenScratchC, seqLen, _hiddenDim);
        using var dNorm1X_B = AutogradScratchWorkspace.Wrap(ws.HiddenScratchD, seqLen, _hiddenDim);

        Parallel.Invoke(
            () => _qkvProj.Backward(_weights.QkvType, _weights.QkvWeight, dQkv, state.Norm1X!, dNorm1X_Qkv),
            () => _attnGateProj.Backward(_weights.AttnGateType, _weights.AttnGateWeight, dZ, state.Norm1X!, dNorm1X_Z),
            () => _alphaProj.Backward(_weights.SsmAlphaType, _weights.SsmAlphaWeight, dAlpha, state.Norm1X!, dNorm1X_A),
            () => _betaProj.Backward(_weights.SsmBetaType, _weights.SsmBetaWeight, dBetaRaw, state.Norm1X!, dNorm1X_B)
        );

        using var dNorm1X = AutogradScratchWorkspace.Wrap(ws.HiddenScratchE, seqLen, _hiddenDim);
        ElementwiseKernels.Add(dNorm1X_Qkv, dNorm1X_Z, dNorm1X);
        ElementwiseKernels.Add(dNorm1X, dNorm1X_A, dNorm1X);
        ElementwiseKernels.Add(dNorm1X, dNorm1X_B, dNorm1X);

        // 8. Pre-RMSNorm1 backward + residual
        using var dX_norm = AutogradScratchWorkspace.Wrap(ws.HiddenScratchF, seqLen, _hiddenDim);
        using (var wNorm1 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.AttnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dNorm1X, xInput, wNorm1, dX_norm, null, _rmsNormEps);
        }

        return TensorOps.Add(dX1, dX_norm);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _qkvProj.Dispose();
            _attnGateProj.Dispose();
            _alphaProj.Dispose();
            _betaProj.Dispose();
            _ssmOutProj.Dispose();
            _gateProj.Dispose();
            _upProj.Dispose();
            _downProj.Dispose();
        }
    }
}
