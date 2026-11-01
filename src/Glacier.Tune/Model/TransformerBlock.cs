using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Glacier.Inference.Model;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Kernels;

namespace Glacier.Tune.Model;

/// <summary>
/// Intermediate activations saved for a single layer during forward pass for backward chain-rule propagation.
/// </summary>
public sealed class BlockActivationState : IDisposable
{
    public Tensor<float>? Norm1X { get; set; }
    public Tensor<float>? Q { get; set; }
    public Tensor<float>? K { get; set; }
    public Tensor<float>? V { get; set; }
    public Tensor<float>? AttnProbs { get; set; }
    public Tensor<float>? AttnOut { get; set; }
    public Tensor<float>? X1 { get; set; }
    public Tensor<float>? Norm2X { get; set; }
    public Tensor<float>? Gate { get; set; }
    public Tensor<float>? Up { get; set; }
    public Tensor<float>? Hidden { get; set; }

    public void Dispose()
    {
        Norm1X?.Dispose();
        Q?.Dispose();
        K?.Dispose();
        V?.Dispose();
        AttnProbs?.Dispose();
        AttnOut?.Dispose();
        X1?.Dispose();
        Norm2X?.Dispose();
        Gate?.Dispose();
        Up?.Dispose();
        Hidden?.Dispose();
    }
}

/// <summary>
/// Encapsulates a full causal transformer block with zero-copy GGUF base weights
/// and trainable LoRA adapters on attention and FFN projections.
/// Supports exact forward pass and reverse-mode backpropagation.
/// </summary>
public sealed unsafe class TransformerBlock : IDisposable
{
    private readonly int _layerIndex;
    private readonly int _hiddenDim;
    private readonly int _ffnDim;
    private readonly int _nHeadsQ;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly float _ropeFreqBase;
    private readonly float _rmsNormEps;
    private readonly LayerWeights _weights;
    private readonly LoraConfig _config;

    private readonly LoraAdapter _qProj;
    private readonly LoraAdapter _kProj;
    private readonly LoraAdapter _vProj;
    private readonly LoraAdapter _oProj;
    private readonly LoraAdapter _gateProj;
    private readonly LoraAdapter _upProj;
    private readonly LoraAdapter _downProj;

    private readonly List<Tensor<float>> _trainableParameters = [];
    private Glacier.Tune.Gpu.GpuLoraEngine? _gpuEngine;
    private bool _disposed;

    public int LayerIndex => _layerIndex;
    public IReadOnlyList<Tensor<float>> TrainableParameters => _trainableParameters;

    public LoraAdapter QProj => _qProj;
    public LoraAdapter KProj => _kProj;
    public LoraAdapter VProj => _vProj;
    public LoraAdapter OProj => _oProj;
    public LoraAdapter GateProj => _gateProj;
    public LoraAdapter UpProj => _upProj;
    public LoraAdapter DownProj => _downProj;

    public TransformerBlock(
        int layerIndex,
        LayerWeights weights,
        int hiddenDim,
        int ffnDim,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        float ropeFreqBase,
        float rmsNormEps,
        LoraConfig config,
        GpuTarget target = GpuTarget.Auto)
    {
        _layerIndex = layerIndex;
        _weights = weights;
        _hiddenDim = hiddenDim;
        _ffnDim = ffnDim;
        _nHeadsQ = nHeadsQ;
        _nHeadsKv = nHeadsKv;
        _headDim = headDim;
        _ropeFreqBase = ropeFreqBase;
        _rmsNormEps = rmsNormEps;
        _config = config;

        int qDim = _nHeadsQ * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int baseSeed = config.Seed ?? 42;

        // Initialize LoRA adapters wrapping base weights with deterministic per-(layer, projection) seeds
        _qProj = new LoraAdapter(_hiddenDim, qDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 0), target, HasTarget(config, "q_proj", "attn_q"));
        _kProj = new LoraAdapter(_hiddenDim, kvDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 1), target, HasTarget(config, "k_proj", "attn_k"));
        _vProj = new LoraAdapter(_hiddenDim, kvDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 2), target, HasTarget(config, "v_proj", "attn_v"));
        _oProj = new LoraAdapter(qDim, _hiddenDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 3), target, HasTarget(config, "o_proj", "attn_output"));

        _gateProj = new LoraAdapter(_hiddenDim, _ffnDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 4), target, HasTarget(config, "gate_proj", "ffn_gate"));
        _upProj = new LoraAdapter(_hiddenDim, _ffnDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 5), target, HasTarget(config, "up_proj", "ffn_up"));
        _downProj = new LoraAdapter(_ffnDim, _hiddenDim, config.Rank, config.Alpha, DeriveSeed(baseSeed, layerIndex, 6), target, HasTarget(config, "down_proj", "ffn_down"));

        RegisterAdapter(_qProj);
        RegisterAdapter(_kProj);
        RegisterAdapter(_vProj);
        RegisterAdapter(_oProj);
        RegisterAdapter(_gateProj);
        RegisterAdapter(_upProj);
        RegisterAdapter(_downProj);
    }

    private static int DeriveSeed(int baseSeed, int layerIndex, int projOrdinal) =>
        unchecked(baseSeed + layerIndex * 1009 + projOrdinal * 9176);

    private static bool HasTarget(LoraConfig config, string canonicalName, string ggufAlias)
    {
        if (config.TargetModules == null || config.TargetModules.Length == 0)
        {
            return true;
        }

        foreach (string module in config.TargetModules)
        {
            if (string.Equals(module, canonicalName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(module, ggufAlias, StringComparison.OrdinalIgnoreCase) ||
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

    /// <summary>
    /// Binds VRAM pointers for frozen base weights to all LoRA adapters in this block.
    /// </summary>
    public void ConnectGpuWeights(Glacier.Tune.Gpu.GpuLoraEngine engine)
    {
        _gpuEngine = engine;
        _qProj.SetGpuBaseWeights(engine, engine.GetQ(_layerIndex), engine.GetQBias(_layerIndex));
        _kProj.SetGpuBaseWeights(engine, engine.GetK(_layerIndex), engine.GetKBias(_layerIndex));
        _vProj.SetGpuBaseWeights(engine, engine.GetV(_layerIndex), engine.GetVBias(_layerIndex));
        _oProj.SetGpuBaseWeights(engine, engine.GetAttnOut(_layerIndex), engine.GetAttnOutBias(_layerIndex));
        _gateProj.SetGpuBaseWeights(engine, engine.GetGate(_layerIndex));
        _upProj.SetGpuBaseWeights(engine, engine.GetUp(_layerIndex));
        _downProj.SetGpuBaseWeights(engine, engine.GetDown(_layerIndex));
    }

    /// <summary>
    /// Forward pass through the transformer block.
    /// If saveActivations is true, captures intermediate state required for backpropagation.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> x, int seqLen, bool saveActivations, out BlockActivationState? state)
    {
        state = saveActivations ? new BlockActivationState() : null;

        Stopwatch? swDbg = _layerIndex == 0 && saveActivations ? Stopwatch.StartNew() : null;
        long tNorm1 = 0, tQkv = 0, tRope = 0, tAttn = 0, tO = 0, tNorm2 = 0, tDown = 0;

        // 1. Attention Pre-RMSNorm: norm1X = RMSNorm(x, attn_norm)
        var norm1X = new Tensor<float>(seqLen, _hiddenDim);
        using (var wNorm1 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.AttnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNorm(x, wNorm1, norm1X, _rmsNormEps);
        }
        if (state != null) state.Norm1X = norm1X;
        if (swDbg is not null) { tNorm1 = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // 2. Linear Projections Q, K, V with base weights + LoRA
        int qDim = _nHeadsQ * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        var q = new Tensor<float>(seqLen, qDim);
        var k = new Tensor<float>(seqLen, kvDim);
        var v = new Tensor<float>(seqLen, kvDim);

        var gpu = _gpuEngine;
        if (gpu != null &&
            gpu.SupportsForwardGemm(_weights.QType) &&
            gpu.SupportsForwardGemm(_weights.KType) &&
            gpu.SupportsForwardGemm(_weights.VType))
        {
            gpu.ForwardQKV(
                _layerIndex, norm1X, q, k, v, _qProj, _kProj, _vProj, seqLen);
        }
        else
        {
            Parallel.Invoke(
                () => _qProj.Forward(_weights.QType, _weights.QWeight, _weights.QBias, norm1X, q, seqLen),
                () => _kProj.Forward(_weights.KType, _weights.KWeight, _weights.KBias, norm1X, k, seqLen),
                () => _vProj.Forward(_weights.VType, _weights.VWeight, _weights.VBias, norm1X, v, seqLen)
            );
        }
        if (swDbg is not null) { tQkv = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // 3 & 4. RoPE + Causal Multi-Head GQA Attention
        var attnOut = new Tensor<float>(seqLen, qDim);
        var attnProbs = state != null ? new Tensor<float>(_nHeadsQ, seqLen, seqLen) : null;

        if (gpu != null && gpu.SupportsAttention(seqLen, _headDim))
        {
            gpu.ForwardAttention(
                _layerIndex, q, k, v, attnOut, attnProbs,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim, _ropeFreqBase);
        }
        else
        {
            RoPEKernel.ForwardSequence(
                (float*)q.DataPointer, (float*)k.DataPointer,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim, _ropeFreqBase);

            attnProbs ??= new Tensor<float>(_nHeadsQ, seqLen, seqLen);
            CausalAttentionKernel.Forward(
                (float*)q.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer,
                (float*)attnOut.DataPointer, (float*)attnProbs.DataPointer,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim);
        }
        if (swDbg is not null) { tAttn = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        if (state != null)
        {
            state.Q = q;
            state.K = k;
            state.V = v;
            state.AttnProbs = attnProbs!;
            state.AttnOut = attnOut;
        }
        else
        {
            attnProbs?.Dispose();
        }

        // 5. Output projection + Residual: x1 = x + OProj(attnOut)
        var projOut = new Tensor<float>(seqLen, _hiddenDim);
        _oProj.Forward(_weights.AttnOutType, _weights.AttnOutWeight, _weights.AttnOutBias, attnOut, projOut, seqLen);
        var x1 = TensorOps.Add(x, projOut);
        projOut.Dispose();
        if (state != null) state.X1 = x1;
        if (swDbg is not null) { tO = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        if (!saveActivations)
        {
            norm1X.Dispose();
            q.Dispose();
            k.Dispose();
            v.Dispose();
            attnOut.Dispose();
        }

        // 6. FFN Pre-RMSNorm: norm2X = RMSNorm(x1, ffn_norm)
        var norm2X = new Tensor<float>(seqLen, _hiddenDim);
        using (var wNorm2 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.FfnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNorm(x1, wNorm2, norm2X, _rmsNormEps);
        }
        if (state != null) state.Norm2X = norm2X;
        if (swDbg is not null) { tNorm2 = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // 7 & 8. Fused SwiGLU FFN & Down Projection in VRAM / CPU
        var downOut = new Tensor<float>(seqLen, _hiddenDim);
        Tensor<float>? gate = null;
        Tensor<float>? up = null;
        Tensor<float>? hidden = null;

        if (state != null)
        {
            gate = new Tensor<float>(seqLen, _ffnDim);
            up = new Tensor<float>(seqLen, _ffnDim);
            hidden = new Tensor<float>(seqLen, _ffnDim);
            state.Gate = gate;
            state.Up = up;
            state.Hidden = hidden;
        }

        if (gpu != null &&
            gpu.SupportsForwardGemm(_weights.FfnGateType) &&
            gpu.SupportsForwardGemm(_weights.FfnUpType) &&
            gpu.SupportsForwardGemm(_weights.FfnDownType))
        {
            gpu.ForwardFfn(
                _layerIndex, norm2X, gate, up, hidden, downOut,
                _gateProj, _upProj, _downProj, seqLen);
        }
        else
        {
            gate ??= new Tensor<float>(seqLen, _ffnDim);
            up ??= new Tensor<float>(seqLen, _ffnDim);
            hidden ??= new Tensor<float>(seqLen, _ffnDim);

            Parallel.Invoke(
                () => _gateProj.Forward(_weights.FfnGateType, _weights.FfnGateWeight, null, norm2X, gate, seqLen),
                () => _upProj.Forward(_weights.FfnUpType, _weights.FfnUpWeight, null, norm2X, up, seqLen)
            );

            SwiGluKernel.Forward(
                (float*)gate.DataPointer, (float*)up.DataPointer,
                (float*)hidden.DataPointer, seqLen * _ffnDim);

            _downProj.Forward(_weights.FfnDownType, _weights.FfnDownWeight, null, hidden, downOut, seqLen);

            if (state == null)
            {
                gate.Dispose();
                up.Dispose();
                hidden.Dispose();
            }
        }

        var output = TensorOps.Add(x1, downOut);
        downOut.Dispose();

        if (swDbg is not null)
        {
            tDown = swDbg.ElapsedMilliseconds;
            GlacierDiagnostics.LogDebug($"[LAYER 0 FWD BREAKDOWN] Norm1: {tNorm1}ms | QKV: {tQkv}ms | RoPE: {tRope}ms | Attn: {tAttn}ms | O: {tO}ms | Norm2: {tNorm2}ms | FFN(Gate/Up/SwiGLU/Down): {tDown}ms");
        }

        if (!saveActivations)
        {
            x1.Dispose();
            norm2X.Dispose();
        }

        return output;
    }

    /// <summary>
    /// Executes full reverse-mode backpropagation through the transformer block.
    /// Propagates gradients through frozen base projections and LoRA adapters across
    /// DownProj, SwiGLU, Gate/Up, Norm2, OProj, CausalAttention, RoPE, Q/K/V, and Norm1.
    /// Returns dXInput for the preceding layer.
    /// </summary>
    public Tensor<float> Backward(
        Tensor<float> dOutput,
        BlockActivationState state,
        Tensor<float> xInput,
        int seqLen,
        AutogradScratchWorkspace? workspace = null)
    {
        var ws = workspace ?? AutogradScratchWorkspace.GetOrCreate(seqLen, _hiddenDim, _ffnDim, _nHeadsQ, _nHeadsKv, _headDim);

        Stopwatch? swDbg = _layerIndex == 0 ? Stopwatch.StartNew() : null;
        long tDownBwd = 0, tSwigluBwd = 0, tGateUpBwd = 0, tNorm2Bwd = 0, tOBwd = 0, tAttnBwd = 0, tRopeBwd = 0, tQkvBwd = 0, tNorm1Bwd = 0;

        // -------------------------------------------------------------
        // A. FFN Backward Pass
        // -------------------------------------------------------------
        // Down projection backward: dHidden = dOutput * (W_down + DeltaW_down)
        using var dHidden = AutogradScratchWorkspace.Wrap(ws.FfnScratchA, seqLen, _ffnDim);
        _downProj.Backward(_weights.FfnDownType, _weights.FfnDownWeight, dOutput, state.Hidden!, dHidden);
        if (swDbg is not null) { tDownBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // SwiGLU backward: dGate, dUp
        using var dGate = AutogradScratchWorkspace.Wrap(ws.FfnScratchB, seqLen, _ffnDim);
        using var dUp = AutogradScratchWorkspace.Wrap(ws.FfnScratchC, seqLen, _ffnDim);
        SwiGluKernel.Backward(
            (float*)dHidden.DataPointer,
            (float*)state.Gate!.DataPointer,
            (float*)state.Up!.DataPointer,
            (float*)dGate.DataPointer,
            (float*)dUp.DataPointer,
            seqLen * _ffnDim);
        if (swDbg is not null) { tSwigluBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // Gate & Up projections backward -> dNorm2X
        using var dNorm2X_Gate = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, seqLen, _hiddenDim);
        using var dNorm2X_Up = AutogradScratchWorkspace.Wrap(ws.HiddenScratchB, seqLen, _hiddenDim);
        Parallel.Invoke(
            () => _gateProj.Backward(_weights.FfnGateType, _weights.FfnGateWeight, dGate, state.Norm2X!, dNorm2X_Gate),
            () => _upProj.Backward(_weights.FfnUpType, _weights.FfnUpWeight, dUp, state.Norm2X!, dNorm2X_Up)
        );
        using var dNorm2X = AutogradScratchWorkspace.Wrap(ws.HiddenScratchC, seqLen, _hiddenDim);
        ElementwiseKernels.Add(dNorm2X_Gate, dNorm2X_Up, dNorm2X);
        if (swDbg is not null) { tGateUpBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // RMSNorm2 backward -> dX1_norm
        using var dX1_norm = AutogradScratchWorkspace.Wrap(ws.HiddenScratchD, seqLen, _hiddenDim);
        using (var wNorm2 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.FfnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dNorm2X, state.X1!, wNorm2, dX1_norm, null, _rmsNormEps);
        }
        if (swDbg is not null) { tNorm2Bwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // Residual connection: dX1 = dOutput + dX1_norm
        var dX1 = TensorOps.Add(dOutput, dX1_norm);

        // -------------------------------------------------------------
        // B. Attention Backward Pass
        // -------------------------------------------------------------
        int qDim = _nHeadsQ * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // OProj backward: dAttnOut = dX1 * (W_o + DeltaW_o)
        using var dAttnOut = AutogradScratchWorkspace.Wrap(ws.QDimScratchA, seqLen, qDim);
        _oProj.Backward(_weights.AttnOutType, _weights.AttnOutWeight, dX1, state.AttnOut!, dAttnOut);
        if (swDbg is not null) { tOBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // Causal Attention & RoPE backward -> dQ, dK, dV
        using var dQ = AutogradScratchWorkspace.Wrap(ws.QDimScratchB, seqLen, qDim);
        using var dK = AutogradScratchWorkspace.Wrap(ws.KvDimScratchA, seqLen, kvDim);
        using var dV = AutogradScratchWorkspace.Wrap(ws.KvDimScratchB, seqLen, kvDim);

        if (_gpuEngine != null && _gpuEngine.SupportsAttention(seqLen, _headDim))
        {
            _gpuEngine.BackwardAttention(
                _layerIndex, dAttnOut,
                state.Q!, state.K!, state.V!, state.AttnProbs!,
                dQ, dK, dV,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim, _ropeFreqBase);
        }
        else
        {
            CausalAttentionKernel.Backward(
                (float*)dAttnOut.DataPointer,
                (float*)state.Q!.DataPointer, (float*)state.K!.DataPointer, (float*)state.V!.DataPointer,
                (float*)state.AttnProbs!.DataPointer,
                (float*)dQ.DataPointer, (float*)dK.DataPointer, (float*)dV.DataPointer,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim);

            RoPEKernel.BackwardSequence(
                (float*)dQ.DataPointer, (float*)dK.DataPointer,
                seqLen, _nHeadsQ, _nHeadsKv, _headDim, _ropeFreqBase);
        }
        if (swDbg is not null) { tAttnBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // Q, K, V projections backward -> dNorm1X
        using var dNorm1X_Q = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, seqLen, _hiddenDim);
        using var dNorm1X_K = AutogradScratchWorkspace.Wrap(ws.HiddenScratchB, seqLen, _hiddenDim);
        using var dNorm1X_V = AutogradScratchWorkspace.Wrap(ws.HiddenScratchC, seqLen, _hiddenDim);
        Parallel.Invoke(
            () => _qProj.Backward(_weights.QType, _weights.QWeight, dQ, state.Norm1X!, dNorm1X_Q),
            () => _kProj.Backward(_weights.KType, _weights.KWeight, dK, state.Norm1X!, dNorm1X_K),
            () => _vProj.Backward(_weights.VType, _weights.VWeight, dV, state.Norm1X!, dNorm1X_V)
        );
        using var dNorm1X = AutogradScratchWorkspace.Wrap(ws.HiddenScratchD, seqLen, _hiddenDim);
        ElementwiseKernels.Add(dNorm1X_Q, dNorm1X_K, dNorm1X);
        ElementwiseKernels.Add(dNorm1X, dNorm1X_V, dNorm1X);
        if (swDbg is not null) { tQkvBwd = swDbg.ElapsedMilliseconds; swDbg.Restart(); }

        // RMSNorm1 backward -> dX_norm
        using var dX_norm = AutogradScratchWorkspace.Wrap(ws.HiddenScratchE, seqLen, _hiddenDim);
        using (var wNorm1 = Tensor<float>.FromSpan(new ReadOnlySpan<float>(_weights.AttnNormWeight, _hiddenDim), [_hiddenDim]))
        {
            ElementwiseKernels.RMSNormBackward(dNorm1X, xInput, wNorm1, dX_norm, null, _rmsNormEps);
        }
        if (swDbg is not null)
        {
            tNorm1Bwd = swDbg.ElapsedMilliseconds;
            GlacierDiagnostics.LogDebug($"[LAYER 0 BWD BREAKDOWN] Down: {tDownBwd}ms | SwiGLU: {tSwigluBwd}ms | GateUp: {tGateUpBwd}ms | Norm2: {tNorm2Bwd}ms | O: {tOBwd}ms | Attn: {tAttnBwd}ms | RoPE: {tRopeBwd}ms | QKV: {tQkvBwd}ms | Norm1: {tNorm1Bwd}ms");
        }

        // Residual connection: dXInput = dX1 + dX_norm
        var dXInput = TensorOps.Add(dX1, dX_norm);
        dX1.Dispose();

        return dXInput;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _qProj.Dispose();
            _kProj.Dispose();
            _vProj.Dispose();
            _oProj.Dispose();
            _gateProj.Dispose();
            _upProj.Dispose();
            _downProj.Dispose();
        }
    }
}
