# Changelog

## v1.7.0

- **Direct Preference Optimization (`DpoTrainer` / `DpoLoss` / `DpoDataset`):** Added DPO and conservative cDPO preference training in `Glacier.Tune.Trainer` with JSONL loading, `'DPRF'` reference log-probability caching, `AdamW` gradient clipping, and Native AOT `DpoJsonContext`.

## v1.1.0

- **Sleep Consolidation & MoE Parameter Expansion:** Added `PreallocatedMoeExpansionBank`, SIMD `DreamQualityFilter` (AVX2/AVX-512 cosine similarity + Levenshtein), `DistillationLoss`, and `SleepConsolidationTrainer` for offline Generalized Knowledge Distillation and volatile buffer pruning.
- **Ecosystem Dependencies:** Updated `Glacier.Inference` and `Glacier.Tensor` references.

## v1.0.7

- **Frozen Base-Weight Backpropagation (`tune_bwd.cubin` & `LoraKernels.AccumulateBaseBackward`):** Added transposed quantized and FP32 backward GEMM kernels (`Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, `F32`) propagating input gradients across all 7 projections per block.
- **Arbitrary-`head_dim` CUDA Causal GQA Attention:** Added warp-aligned attention training forward and backward kernels supporting head dimensions `32`–`1024`.
- **Tiled Online Fused Cross-Entropy:** Upgraded `FusedCrossEntropyLoss` to 64-bit online Log-Sum-Exp accumulation over bounded tiles without logit truncation bias.
- **Prompt Header Masking, Gradient Accumulation & `TargetModules`:** Masked `<|im_start|>assistant\n` role headers in `ChatMlDataset`, added microbatch gradient averaging and `SaveSteps` checkpoints in `LoraTrainer`, and added deterministic per-projection seeds.
- **Native AOT & GGUF Merger Hardening:** Enabled `<IsAotCompatible>true</IsAotCompatible>`, added `TargetQuantization.F32`, `GgufLoraModel.ForwardLogits`, and overwrite guards in `GgufMerger`.

## v1.0.6

- **Pluggable Diagnostics (`GlacierDiagnostics`):** Replaced direct console output with `IGlacierLogger` (`ConsoleGlacierLogger`, `DelegateGlacierLogger`, `NullGlacierLogger`).
- **`GpuLoraEngine` Split & Benchmark Suite:** Split `GpuLoraEngine` into partial files and added `Glacier.Tune.Benchmarks`.

## v1.0.5

- **`AutogradScratchWorkspace` & 2D Tiled Softmax:** Added pre-allocated scratch buffers and O(1) extra-memory online softmax across forward passes.
