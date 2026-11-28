# Glacier.Tune Architecture

## 1. Zero-Copy GGUF Loading & Frozen Base Parameters

`GgufLoraModel.Load` memory-maps the base `.gguf` file via `Glacier.Inference.Gguf.GgufFile` without decompressing frozen weights in host RAM during training. Each layer's low-rank factor matrices $A \in \mathbb{R}^{d_{\text{in}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{out}}}$ are allocated as trainable unmanaged `Tensor<float>` pairs (`B = 0`, deterministic per-layer/per-projection seeded Kaiming $A$) scaling updates by $\alpha / r$.

When targeting CUDA (`GpuTarget.Nvidia`), `GpuLoraEngine` (`GpuLoraEngine.cs`, `.Forward.cs`, `.Backward.cs`, `.Memory.cs`) uploads quantized base weights once into device VRAM and executes forward/backward projections without per-step host-to-device weight streaming.

---

## 2. Reverse-Mode Transformer & Hybrid GDN Backpropagation

```text
GGUF MMap -> TransformerBlock / GdnBlock Forward -> FusedCrossEntropyLoss
                 ^                                       |
                 | (X_l checkpoints + scratch reuse)     v
          dX += dY * W_base + LoRA Backward <----- Grad Hidden State
```

- **Frozen Base-Weight Transpose-GEMM (`LoraKernels.AccumulateBaseBackward` & `tune_bwd.cubin`):** Propagates $dX \mathrel{+}= dY \cdot W_{\text{base}}$ directly from quantized base tensors (`Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, `F32`) through all 7 projections (`Q`, `K`, `V`, `O`, `Gate`, `Up`, `Down`) from step 0 ($B = 0$) onward.
- **Causal GQA Attention (`CausalAttentionKernel` & `tune_attn_causal_gqa_*`):** Computes warp-aligned forward and backward attention passes with optional 2D online tiling and activation checkpointing (`TrainingArguments.CheckpointActivations`) that retains only layer inputs $X_l$ and recomputes intermediates during the backward sweep.
- **Analytical SwiGLU & Orthogonal RoPE Inversion:** `SwiGluKernel` and `RoPEKernel.BackwardSequence` ($\sin(-\theta) = -\sin\theta$) compute exact layer derivatives over reusable `AutogradScratchWorkspace` and `GdnScratchWorkspace` unmanaged slabs.
- **Tiled Fused Cross-Entropy (`FusedCrossEntropyLoss`):** Computes exact 64-bit online Log-Sum-Exp normalization across vocabulary tiles without materializing full `[seqLen, vocabSize]` logit matrices.

---

## 3. Alignment, Consolidation & Adapter Export

- **DPO (`DpoTrainer`):** Computes log-probabilities under both policy and reference weights (optionally cached in `'DPRF'` binary files) and applies `DpoLoss.Compute` with optional SFT anchor regularization.
- **Sleep Consolidation (`PreallocatedMoeExpansionBank`):** Reserves 64-byte-aligned unmanaged expert slots (`MoeExpertSlotDescriptor`) up front so `SleepConsolidationTrainer` can unlock experts and distill filtered dream rollouts via `DistillationLoss` (`ComputeLogitKlAndCe`, `ComputeHiddenMseAndCosine`) and prune volatile buffers cleanly.
- **Adapter Format & Merging:** Checkpoints write `LORA` (`0x41524F4C`) binary files alongside `ADMW` (`0x574D4441`) AdamW state. `GgufMerger` dequantizes adapted rows, adds $(\alpha / r) \cdot A \cdot B$, re-quantizes to the requested `TargetQuantization`, and copies unadapted tensors directly.
