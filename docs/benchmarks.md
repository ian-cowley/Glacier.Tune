# Glacier.Tune Benchmarks

All numbers below come from physical hardware runs using `Glacier.Tune.Benchmarks` or recorded physical GPU training telemetry.

## Test Environment

- **CPU:** AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, Zen 5, AVX-512)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6, `sm_89`)
- **RAM:** 31 GB
- **OS:** Windows 11 (`10.0.26300.0`)
- **Runtime:** `.NET 10.0.15` (`.NET SDK 11.0.100`, Release x64)

## Reproduce Locally

```bash
# Fast console kernel microbenchmarks
dotnet run --project benchmarks/Glacier.Tune.Benchmarks/Glacier.Tune.Benchmarks.csproj -c Release

# BenchmarkDotNet suite (LoraKernels, FusedCrossEntropyLoss, RoPEKernel, SwiGluKernel)
dotnet run --project benchmarks/Glacier.Tune.Benchmarks/Glacier.Tune.Benchmarks.csproj -c Release -- --bdn
```

---

## 1. Executed Kernel Microbenchmarks (Release, `.NET 10.0.15`)

### LoRA Delta & Backward (`LoraKernels`, `InFeatures=3584, OutFeatures=3584`)

| Sequence Length | Rank | Forward (`ApplyLoraDelta`) | Backward (`Backward`) |
| :---: | :---: | ---: | ---: |
| `128` | `8` | `2.210 ms` | `7.178 ms` |
| `128` | `16` | `2.101 ms` | `6.380 ms` |
| `512` | `8` | `4.605 ms` | `8.629 ms` |
| `512` | `16` | `1.434 ms` | `7.957 ms` |

### Vocabulary Cross-Entropy, RoPE, SwiGLU, DPO & Dream Filter

| Kernel | Parameters | Forward / Call | Backward |
| :--- | :--- | ---: | ---: |
| `FusedCrossEntropyLoss` | `T=128, D=3584, V=1024` (`F32`) | **12.068 ms** | *(included)* |
| `FusedCrossEntropyLoss` | `T=128, D=3584, V=4096` (`F32`) | **88.780 ms** | *(included)* |
| `FusedCrossEntropyLoss` | `T=128, D=3584, V=32000` (`F32`) | **705.042 ms** | *(included)* |
| `RoPEKernel` | `T=128, Hq=28, Hkv=4, D=128` | **89.0 us** | **88.8 us** |
| `RoPEKernel` | `T=512, Hq=28, Hkv=4, D=128` | **397.2 us** | **398.1 us** |
| `SwiGluKernel` | `T=128, Dffn=18944` (`2.42M` elem) | **0.416 ms** | **1.009 ms** |
| `SwiGluKernel` | `T=512, Dffn=18944` (`9.70M` elem) | **2.594 ms** | **4.884 ms` |
| `DpoLoss.Compute` | `1,000,000` iterations | **29.65 ns / call** | — |
| `DreamQualityFilter.CosineSimilarity` | `D=1024`, `200,000` iterations | **64.15 ns / call** | — |
| `DreamQualityFilter.ComputeScore` | `D=1024` + 64-char Levenshtein (`20,000` iters) | **0.273 us / call** | — |

---

## 2. Historical Physical 7B LoRA Run (Pre-v1.0.7 Telemetry)

Recorded on physical NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) + AMD Ryzen AI 9 HX 370 using `Glacier.Tune.Demo` with `Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` (`4.36 GB` GGUF, `40,370,176` trainable LoRA parameters at `r=16, alpha=32`, max seq length `512`) over a 10-step run before v1.0.7 added full frozen base-weight transpose-GEMM backpropagation (`dX += dY * W_base`):

| Metric | Measured Value |
| :--- | :--- |
| Cold GGUF memory-map load | `< 350 ms` |
| VRAM base weight upload | `4.42 s` (`4,421 ms` for `4.36 GB`) |
| 10-step run duration | `70.4 s` total (`6,960.9 ms` avg step, `46.0 tok/s`) |
| Step-1 loss (across 8 kernel stages) | `21.8704` |
| Standalone GGUF adapter merge | `6.82 s` |

*Note: Only 10-step hardware telemetry was recorded; full multi-hundred-step convergence epochs were not executed.*
