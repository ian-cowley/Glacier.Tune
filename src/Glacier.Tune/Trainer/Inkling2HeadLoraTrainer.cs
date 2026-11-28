#if !ANDROID
namespace Glacier.Tune.Trainer;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;

/// <summary>
/// Pure-C# .NET 10 post-Epilogue Head-LoRA trainer for Thinking Machines Lab's Inkling-2-Mini.
/// Phase 1 extracts and caches exact post-OutputNorm hidden states via Inkling2D3D12Model (RTX 4060 / 890M).
/// Phase 2 trains Inkling2HeadLoraAdapter (A [Rank, Embd], B [Rank, Vocab]) with exact FinalLogitSoftcapping
/// and Cross-Entropy gradients via AVX-512 parallel AdamW + cosine LR schedule.
/// </summary>
public sealed unsafe class Inkling2HeadLoraTrainer
{
    public const uint CacheMagic = 0x48324B49; // "IK2H"

    public sealed record TrainConfig(
        int Rank = 64,
        float Alpha = 32.0f,
        bool UseSiLU = true,
        int Epochs = 2,
        int BatchTokens = 32,
        float LearningRate = 1.5e-4f,
        float WeightDecay = 0.01f,
        float MaxGradNorm = 1.0f,
        float WarmupFraction = 0.1f,
        float KlBeta = 0.5f,
        int Seed = 42);

    public sealed class CachedHiddenDataset
    {
        public int EmbdDim { get; }
        public int Count => Targets.Length;
        public Half[] HiddenFp16 { get; } // [Count * EmbdDim]
        public int[] Targets { get; }

        public CachedHiddenDataset(int embdDim, Half[] hiddenFp16, int[] targets)
        {
            EmbdDim = embdDim;
            HiddenFp16 = hiddenFp16;
            Targets = targets;
        }

        public void CopyHiddenTo(int index, Span<float> dst)
        {
            int off = index * EmbdDim;
            for (int d = 0; d < EmbdDim; d++)
                dst[d] = (float)HiddenFp16[off + d];
        }

        public void Save(string path)
        {
            string tmp = LoraBinaryFormat.CreateTempSiblingPath(path);
            try
            {
                using (var fs = File.Create(tmp))
                using (var bw = new BinaryWriter(fs))
                {
                    bw.Write(CacheMagic);
                    bw.Write(EmbdDim);
                    bw.Write(Targets.Length);
                    bw.Flush();
                    fs.Write(MemoryMarshal.AsBytes(Targets.AsSpan()));
                    fs.Write(MemoryMarshal.AsBytes(HiddenFp16.AsSpan()));
                    fs.Flush(true);
                }
                File.Move(tmp, path, overwrite: true);
            }
            catch
            {
                try { if (File.Exists(tmp)) File.Delete(tmp); }
                catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
                {
                    System.Diagnostics.Trace.TraceWarning($"[Inkling2HeadLoraTrainer] Failed to clean temp file '{tmp}': {cleanupEx.Message}");
                }
                throw;
            }
        }

        public static CachedHiddenDataset Load(string path)
        {
            using var fs = File.OpenRead(path);
            if (fs.Length < 12)
                throw new InvalidDataException($"Truncated hidden cache header in '{path}' ({fs.Length} bytes).");

            using var br = new BinaryReader(fs);
            uint magic = br.ReadUInt32();
            if (magic != CacheMagic)
                throw new InvalidDataException($"Invalid hidden cache magic: 0x{magic:X8}");

            int embdDim = br.ReadInt32();
            int count = br.ReadInt32();
            if (embdDim <= 0 || embdDim > 65536)
                throw new InvalidDataException($"Invalid hidden cache embedding dimension: {embdDim}.");
            if (count < 0 || count > int.MaxValue / sizeof(int))
                throw new InvalidDataException($"Invalid hidden cache count: {count}.");

            long totalHiddenElems = (long)count * embdDim;
            if (totalHiddenElems > int.MaxValue / sizeof(ushort))
                throw new InvalidDataException($"Hidden cache element count {totalHiddenElems} exceeds maximum buffer size.");

            long expectedPayloadBytes = checked((long)count * sizeof(int) + totalHiddenElems * sizeof(ushort));
            long remaining = fs.Length - fs.Position;
            if (expectedPayloadBytes != remaining)
                throw new InvalidDataException($"Hidden cache payload size mismatch in '{path}': expected {expectedPayloadBytes} bytes, found {remaining}.");

            var targets = new int[count];
            var hidden = new Half[checked((int)totalHiddenElems)];
            fs.ReadExactly(MemoryMarshal.AsBytes(targets.AsSpan()));
            fs.ReadExactly(MemoryMarshal.AsBytes(hidden.AsSpan()));

            for (int i = 0; i < hidden.Length; i++)
            {
                if (!Half.IsFinite(hidden[i]))
                {
                    throw new InvalidDataException($"Non-finite FP16 hidden value ({hidden[i]}) in cache '{path}' at index {i}.");
                }
            }

            return new CachedHiddenDataset(embdDim, hidden, targets);
        }
    }

    /// <summary>
    /// Runs Inkling2D3D12Model forward over every example in <paramref name="dataset"/> and caches
    /// (post-OutputNorm hidden vector, next supervised target token) pairs in FP16.
    /// </summary>
    public static CachedHiddenDataset ExtractOrLoadHiddenStates(
        Inkling2D3D12Model model,
        Inkling2Dataset dataset,
        string cachePath,
        string splitLabel = "train")
    {
        if (File.Exists(cachePath))
        {
            var cached = CachedHiddenDataset.Load(cachePath);
            if (cached.EmbdDim != model.EmbeddingDim)
            {
                throw new InvalidDataException(
                    $"Cached hidden embedding dimension ({cached.EmbdDim}) does not match model embedding dimension ({model.EmbeddingDim}).");
            }
            for (int i = 0; i < cached.Targets.Length; i++)
            {
                if ((uint)cached.Targets[i] >= (uint)model.VocabSize)
                {
                    throw new InvalidDataException(
                        $"Cached target token {cached.Targets[i]} at index {i} is outside model vocabulary bounds [0, {model.VocabSize}).");
                }
            }
            GlacierDiagnostics.LogInformation($"  [Cache Hit:{splitLabel}] Loaded {cached.Count:N0} supervised tokens ({cached.EmbdDim}d FP16) from {cachePath}");
            return cached;
        }

        int embd = model.EmbeddingDim;
        var targetsList = new List<int>(dataset.Count * 128);
        var hiddenChunks = new List<Half[]>(dataset.Count);
        int totalInputTokens = 0;
        var sw = Stopwatch.StartNew();

        // Ensure adapter is detached while extracting frozen base hidden states
        var prevLora = model.HeadLora;
        model.HeadLora = null;
        try
        {
            float[] seqHidden = Array.Empty<float>();
            for (int exIdx = 0; exIdx < dataset.Count; exIdx++)
            {
                var ex = dataset.Examples[exIdx];
                int n = ex.InputIds.Length;
                if (n < 2) continue;

                if (seqHidden.Length < n * embd)
                    seqHidden = new float[n * embd];

                model.ResetState();
                model.ExtractHiddenStatesBatch(ex.InputIds.AsSpan(0, n - 1), 0, seqHidden.AsSpan(0, (n - 1) * embd));
                totalInputTokens += n - 1;

                // Collect supervised positions where pre-shifted labels[t] != IgnoreIndex
                int supInEx = 0;
                for (int t = 0; t < n - 1; t++)
                    if (ex.Labels[t] != Inkling2Dataset.IgnoreIndex) supInEx++;
                if (supInEx == 0) continue;

                var chunk = new Half[supInEx * embd];
                int dstRow = 0;
                for (int t = 0; t < n - 1; t++)
                {
                    int target = ex.Labels[t];
                    if (target == Inkling2Dataset.IgnoreIndex) continue;
                    if ((uint)target >= (uint)model.VocabSize)
                        throw new InvalidDataException($"Target token {target} at example {exIdx} pos {t} is outside vocabulary [0, {model.VocabSize}).");
                    targetsList.Add(target);
                    int srcOff = t * embd;
                    int dstOff = dstRow * embd;
                    for (int d = 0; d < embd; d++)
                        chunk[dstOff + d] = (Half)seqHidden[srcOff + d];
                    dstRow++;
                }
                hiddenChunks.Add(chunk);

                if ((exIdx + 1) % 10 == 0 || exIdx == dataset.Count - 1)
                {
                    double tps = totalInputTokens / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                    GlacierDiagnostics.LogInformation($"  [Extract:{splitLabel}] {exIdx + 1}/{dataset.Count} examples | {targetsList.Count:N0} supervised tokens | {tps:F1} tok/s | {sw.Elapsed.TotalSeconds:F1}s");
                }
            }
        }
        finally
        {
            model.HeadLora = prevLora;
        }

        int totalSup = targetsList.Count;
        var allHidden = new Half[checked(totalSup * embd)];
        int copyOff = 0;
        foreach (var ch in hiddenChunks)
        {
            Array.Copy(ch, 0, allHidden, copyOff, ch.Length);
            copyOff += ch.Length;
        }

        var result = new CachedHiddenDataset(embd, allHidden, targetsList.ToArray());
        result.Save(cachePath);
        GlacierDiagnostics.LogInformation($"  [Cache Saved:{splitLabel}] {result.Count:N0} supervised tokens -> {cachePath}");
        return result;
    }

    /// <summary>
    /// Computes exact forward loss (CE + optional klBeta * KL(p_base || p_lora)) and accumulates
    /// exact analytical gradients into gradA and gradB for one token.
    /// Shared by training and finite-difference unit verification.
    /// </summary>
    public static float ComputeSingleTokenLossAndGrad(
        Inkling2HeadLoraAdapter adapter,
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> baseRawLogits,
        int targetToken,
        float softcap,
        Span<float> workLogits,
        Span<float> workDzRaw,
        float[] gradA,
        float[] gradB,
        bool accumulateGrad = true,
        float klBeta = 0.0f,
        Span<float> workBaseProbs = default)
    {
        int embd = adapter.EmbdDim;
        int vocab = adapter.VocabSize;
        int rank = adapter.Rank;
        float scale = adapter.Scaling;
        bool useSiLU = adapter.UseSiLU;
        bool useKl = klBeta > 0.0f;

        if ((uint)targetToken >= (uint)vocab)
            throw new ArgumentOutOfRangeException(nameof(targetToken), $"targetToken {targetToken} is outside vocabulary [0, {vocab}).");

        // If KL regularization is enabled, compute frozen base distribution p_base = softmax(softcap(baseRawLogits))
        int argMaxBase = 0;
        double baseSumOther = 0.0;
        double invSumBase = 1.0;
        double baseLogZOffset = 0.0;
        if (useKl)
        {
            if (workBaseProbs.Length < vocab)
                throw new ArgumentException($"workBaseProbs length {workBaseProbs.Length} < {vocab}.", nameof(workBaseProbs));

            float maxBase = float.NegativeInfinity;
            float invCapB = softcap > 0f ? 1.0f / softcap : 0f;
            for (int v = 0; v < vocab; v++)
            {
                float zb = baseRawLogits[v];
                if (softcap > 0f) zb = softcap * MathF.Tanh(zb * invCapB);
                workBaseProbs[v] = zb;
                if (v == 0 || zb > maxBase)
                {
                    maxBase = zb;
                    argMaxBase = v;
                }
            }
            for (int v = 0; v < vocab; v++)
            {
                if (v != argMaxBase)
                    baseSumOther += Math.Exp((double)workBaseProbs[v] - maxBase);
            }
            baseLogZOffset = FusedCrossEntropyLoss.Log1p(baseSumOther);
            invSumBase = 1.0 / (1.0 + baseSumOther);
            for (int v = 0; v < vocab; v++)
            {
                workBaseProbs[v] = (v == argMaxBase)
                    ? (float)invSumBase
                    : (float)(Math.Exp((double)workBaseProbs[v] - maxBase) * invSumBase);
            }
        }

        float[] uArr = ArrayPool<float>.Shared.Rent(rank);
        float[] aArr = ArrayPool<float>.Shared.Rent(rank);
        float[] hArr = ArrayPool<float>.Shared.Rent(embd);
        try
        {
            Span<float> u = uArr.AsSpan(0, rank);
            Span<float> a = aArr.AsSpan(0, rank);
            hidden.Slice(0, embd).CopyTo(hArr.AsSpan(0, embd));

            baseRawLogits.Slice(0, vocab).CopyTo(workLogits.Slice(0, vocab));

            fixed (float* pH = hArr)
            fixed (float* pA = adapter.A)
            fixed (float* pB = adapter.B)
            fixed (float* pZ = workLogits)
            {
                for (int r = 0; r < rank; r++)
                {
                    float ur = Inkling2HeadLoraAdapter.DotSimd(pA + r * embd, pH, embd);
                    u[r] = ur;
                    if (useSiLU)
                    {
                        float sig = 1.0f / (1.0f + MathF.Exp(-ur));
                        a[r] = ur * sig;
                    }
                    else
                    {
                        a[r] = ur;
                    }
                }

                for (int r = 0; r < rank; r++)
                {
                    float w = a[r] * scale;
                    if (w != 0f)
                        Inkling2HeadLoraAdapter.AxpySimd(pZ, pB + r * vocab, w, vocab);
                }
            }

            // Apply FinalLogitSoftcapping and compute Barron-stabilized split-sum softmax + NLL (+ KL) loss
            float maxLogit = float.NegativeInfinity;
            int argMax = 0;
            float invCap = softcap > 0f ? 1.0f / softcap : 0f;
            for (int v = 0; v < vocab; v++)
            {
                float zRaw = workLogits[v];
                float z = zRaw;
                if (softcap > 0f)
                {
                    float t = MathF.Tanh(zRaw * invCap);
                    z = softcap * t;
                    workDzRaw[v] = 1.0f - t * t; // store softcap derivative temporarily
                }
                else
                {
                    workDzRaw[v] = 1.0f;
                }
                workLogits[v] = z;
                if (v == 0 || z > maxLogit)
                {
                    maxLogit = z;
                    argMax = v;
                }
            }

            double sumOther = 0.0;
            for (int v = 0; v < vocab; v++)
            {
                if (v != argMax)
                    sumOther += Math.Exp((double)workLogits[v] - maxLogit);
            }

            double logZOffset = FusedCrossEntropyLoss.Log1p(sumOther);
            double invSum = 1.0 / (1.0 + sumOther);
            double nll = (targetToken == argMax)
                ? logZOffset
                : (logZOffset + ((double)maxLogit - (double)workLogits[targetToken]));
            double totalLoss = nll;

            if (useKl)
            {
                double klSum = 0.0;
                for (int v = 0; v < vocab; v++)
                {
                    double pb = (v == argMaxBase) ? invSumBase : workBaseProbs[v];
                    if (pb > 0.0)
                    {
                        double logBase = (v == argMaxBase) ? -baseLogZOffset : Math.Log(pb);
                        double logP = (v == argMax)
                            ? -logZOffset
                            : (((double)workLogits[v] - maxLogit) - logZOffset);
                        klSum += pb * (logBase - logP);
                    }
                }
                totalLoss += klBeta * Math.Max(0.0, klSum);
            }

            float loss = (float)totalLoss;
            if (!accumulateGrad)
                return loss;

            // dz_raw = [(prob - one_hot(target)) + klBeta * (prob - p_base)] * dSoftcap
            for (int v = 0; v < vocab; v++)
            {
                double prob = (v == argMax)
                    ? invSum
                    : Math.Exp((double)workLogits[v] - maxLogit) * invSum;

                double ceGrad = (v == targetToken && v == argMax)
                    ? (-sumOther * invSum)
                    : (prob - (v == targetToken ? 1.0 : 0.0));

                double dz = ceGrad;
                if (useKl)
                {
                    double klGrad = (v == argMax && v == argMaxBase)
                        ? ((baseSumOther - sumOther) * invSum * invSumBase)
                        : (prob - workBaseProbs[v]);
                    dz += klBeta * klGrad;
                }

                workDzRaw[v] = (float)(dz * workDzRaw[v]);
            }

            // Parallelize across rank rows for fast AVX-512 gradB += (scale * a[r]) * dz_raw and da[r] = scale * Dot(B[r], dz_raw)
            fixed (float* pDzRaw = workDzRaw)
            fixed (float* pB = adapter.B)
            fixed (float* pGradB = gradB)
            fixed (float* pGradA = gradA)
            fixed (float* pU = uArr)
            fixed (float* pAct = aArr)
            fixed (float* pH = hArr)
            {
                nint dzPtr = (nint)pDzRaw;
                nint bPtr = (nint)pB;
                nint gbPtr = (nint)pGradB;
                nint gaPtr = (nint)pGradA;
                nint uPtr = (nint)pU;
                nint aPtr = (nint)pAct;
                nint hPtr = (nint)pH;

                Parallel.For(0, rank, r =>
                {
                    float* dz = (float*)dzPtr;
                    float* bRow = (float*)bPtr + r * vocab;
                    float* gbRow = (float*)gbPtr + r * vocab;
                    float* gaRow = (float*)gaPtr + r * embd;
                    float* hVec = (float*)hPtr;

                    // gradB[r, :] += (scale * a[r]) * dz_raw
                    float wB = scale * ((float*)aPtr)[r];
                    if (wB != 0f)
                        Inkling2HeadLoraAdapter.AxpySimd(gbRow, dz, wB, vocab);

                    // da[r] = scale * dot(B[r, :], dz_raw)
                    float da = scale * Inkling2HeadLoraAdapter.DotSimd(bRow, dz, vocab);
                    float du = da;
                    if (useSiLU)
                    {
                        float ur = ((float*)uPtr)[r];
                        float sig = 1.0f / (1.0f + MathF.Exp(-ur));
                        du = da * (sig * (1.0f + ur * (1.0f - sig)));
                    }

                    // gradA[r, :] += du * hidden
                    if (du != 0f)
                        Inkling2HeadLoraAdapter.AxpySimd(gaRow, hVec, du, embd);
                });
            }

            return loss;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(uArr);
            ArrayPool<float>.Shared.Return(aArr);
            ArrayPool<float>.Shared.Return(hArr);
        }
    }

    /// <summary>
    /// Trains an Inkling2HeadLoraAdapter on cached hidden states using RTX 4060 / 890M GPU lm_head base logits
    /// and AVX-512 parallel AdamW with cosine decay, saving the best validation checkpoint to <paramref name="outputPath"/>.
    /// </summary>
    public static Inkling2HeadLoraAdapter Train(
        Inkling2D3D12Model model,
        CachedHiddenDataset trainData,
        CachedHiddenDataset? valData,
        string outputPath,
        TrainConfig? config = null)
    {
        config ??= new TrainConfig();
        int embd = model.EmbeddingDim;
        int vocab = model.VocabSize;
        float softcap = model.FinalLogitSoftcapping;

        var adapter = new Inkling2HeadLoraAdapter(embd, vocab, config.Rank, config.Alpha, config.UseSiLU, config.Seed);
        float[] gradA = new float[adapter.A.Length];
        float[] gradB = new float[adapter.B.Length];
        float[] mA = new float[adapter.A.Length];
        float[] vA = new float[adapter.A.Length];
        float[] mB = new float[adapter.B.Length];
        float[] vB = new float[adapter.B.Length];

        float[] hBuf = new float[embd];
        float[] baseLogits = new float[vocab];
        float[] workLogits = new float[vocab];
        float[] workDzRaw = new float[vocab];
        float[] workBaseProbs = config.KlBeta > 0f ? new float[vocab] : Array.Empty<float>();

        long totalParams = adapter.A.Length + adapter.B.Length;
        GlacierDiagnostics.LogInformation($"================================================================================");
        GlacierDiagnostics.LogInformation($"  GLACIER.TUNE PURE-C# INKLING-2 HEAD-LoRA TRAINER");
        GlacierDiagnostics.LogInformation($"  Rank={config.Rank}, Alpha={config.Alpha}, SiLU={config.UseSiLU}, KlBeta={config.KlBeta} | Trainable={totalParams:N0} FP32 params");
        GlacierDiagnostics.LogInformation($"  Train Tokens={trainData.Count:N0} | Val Tokens={valData?.Count ?? 0:N0} | Epochs={config.Epochs} | Batch={config.BatchTokens}");
        GlacierDiagnostics.LogInformation($"================================================================================");

        float bestValLoss = float.PositiveInfinity;
        if (valData != null && valData.Count > 0)
        {
            float baseValLoss = EvaluateLoss(model, adapter, valData, hBuf, baseLogits, workLogits, workDzRaw, softcap);
            bestValLoss = baseValLoss;
            GlacierDiagnostics.LogInformation($"  [Epoch 0 / Base] val_loss={baseValLoss:F4} (ppl={MathF.Exp(Math.Min(20f, baseValLoss)):F2})");
        }

        int stepsPerEpoch = (trainData.Count + config.BatchTokens - 1) / config.BatchTokens;
        int totalSteps = Math.Max(1, stepsPerEpoch * config.Epochs);
        int warmupSteps = Math.Max(1, (int)(totalSteps * config.WarmupFraction));
        int globalStep = 0;

        int[] order = new int[trainData.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        var rng = new Random(config.Seed);
        var swTotal = Stopwatch.StartNew();

        for (int epoch = 1; epoch <= config.Epochs; epoch++)
        {
            // Fisher-Yates shuffle each epoch
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            double epochLossSum = 0.0;
            int epochTokens = 0;
            var swEpoch = Stopwatch.StartNew();

            for (int start = 0; start < order.Length; start += config.BatchTokens)
            {
                int count = Math.Min(config.BatchTokens, order.Length - start);
                Array.Clear(gradA);
                Array.Clear(gradB);

                double batchLoss = 0.0;
                for (int bi = 0; bi < count; bi++)
                {
                    int idx = order[start + bi];
                    trainData.CopyHiddenTo(idx, hBuf);
                    model.ComputeLmHeadRawLogits(hBuf, baseLogits);
                    float loss = ComputeSingleTokenLossAndGrad(
                        adapter, hBuf, baseLogits, trainData.Targets[idx], softcap,
                        workLogits, workDzRaw, gradA, gradB, accumulateGrad: true,
                        klBeta: config.KlBeta, workBaseProbs: workBaseProbs);
                    batchLoss += loss;
                }

                globalStep++;
                float invBatch = 1.0f / count;
                float lr = ComputeCosineLr(config.LearningRate, globalStep, warmupSteps, totalSteps);
                AdamWStep(adapter.A, gradA, mA, vA, adapter.B, gradB, mB, vB, invBatch, lr, config.WeightDecay, config.MaxGradNorm, globalStep);

                epochLossSum += batchLoss;
                epochTokens += count;

                if (globalStep % 25 == 0 || start + count >= order.Length)
                {
                    double tokPerSec = epochTokens / Math.Max(0.001, swEpoch.Elapsed.TotalSeconds);
                    GlacierDiagnostics.LogInformation($"  [Epoch {epoch}/{config.Epochs} Step {globalStep,4}/{totalSteps}] batch_loss={batchLoss / count:F4} | lr={lr:E2} | {tokPerSec:F1} tok/s");
                }
            }

            double avgTrainLoss = epochLossSum / Math.Max(1, epochTokens);
            if (valData != null && valData.Count > 0)
            {
                float valLoss = EvaluateLoss(model, adapter, valData, hBuf, baseLogits, workLogits, workDzRaw, softcap);
                bool improved = valLoss < bestValLoss;
                if (improved)
                {
                    bestValLoss = valLoss;
                    adapter.Save(outputPath);
                }
                GlacierDiagnostics.LogInformation($"  >>> [Epoch {epoch} Done in {swEpoch.Elapsed.TotalSeconds:F1}s] train_loss={avgTrainLoss:F4} | val_loss={valLoss:F4} (ppl={MathF.Exp(Math.Min(20f, valLoss)):F2}) {(improved ? "*SAVED BEST*" : "")}");
            }
            else
            {
                adapter.Save(outputPath);
                GlacierDiagnostics.LogInformation($"  >>> [Epoch {epoch} Done in {swEpoch.Elapsed.TotalSeconds:F1}s] train_loss={avgTrainLoss:F4} *SAVED*");
            }
        }

        GlacierDiagnostics.LogInformation($"  [Training Complete] Total elapsed: {swTotal.Elapsed.TotalSeconds:F1}s | Best adapter: {outputPath}");
        return File.Exists(outputPath) ? Inkling2HeadLoraAdapter.Load(outputPath) : adapter;
    }

    private static float EvaluateLoss(
        Inkling2D3D12Model model,
        Inkling2HeadLoraAdapter adapter,
        CachedHiddenDataset valData,
        float[] hBuf,
        float[] baseLogits,
        float[] workLogits,
        float[] workDzRaw,
        float softcap)
    {
        double lossSum = 0.0;
        for (int i = 0; i < valData.Count; i++)
        {
            valData.CopyHiddenTo(i, hBuf);
            model.ComputeLmHeadRawLogits(hBuf, baseLogits);
            lossSum += ComputeSingleTokenLossAndGrad(
                adapter, hBuf, baseLogits, valData.Targets[i], softcap,
                workLogits, workDzRaw, Array.Empty<float>(), Array.Empty<float>(), accumulateGrad: false);
        }
        return (float)(lossSum / Math.Max(1, valData.Count));
    }

    private static float ComputeCosineLr(float baseLr, int step, int warmupSteps, int totalSteps)
    {
        if (step <= warmupSteps)
            return baseLr * ((float)step / Math.Max(1, warmupSteps));
        float progress = (float)(step - warmupSteps) / Math.Max(1, totalSteps - warmupSteps);
        progress = Math.Clamp(progress, 0f, 1f);
        return baseLr * (0.05f + 0.95f * 0.5f * (1.0f + MathF.Cos(MathF.PI * progress)));
    }

    private static void AdamWStep(
        float[] wA, float[] gA, float[] mA, float[] vA,
        float[] wB, float[] gB, float[] mB, float[] vB,
        float invBatch, float lr, float weightDecay, float maxGradNorm, int step)
    {
        double normSq = 0.0;
        for (int i = 0; i < gA.Length; i++)
        {
            float g = gA[i] * invBatch;
            gA[i] = g;
            normSq += (double)g * g;
        }
        for (int i = 0; i < gB.Length; i++)
        {
            float g = gB[i] * invBatch;
            gB[i] = g;
            normSq += (double)g * g;
        }

        float clipScale = 1.0f;
        float gradNorm = (float)Math.Sqrt(normSq);
        if (maxGradNorm > 0f && gradNorm > maxGradNorm)
            clipScale = maxGradNorm / gradNorm;

        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc2 = 1.0f - MathF.Pow(beta2, step);
        float wdFactor = 1.0f - lr * weightDecay;

        UpdateTensorAdamW(wA, gA, mA, vA, clipScale, lr, wdFactor, beta1, beta2, eps, bc1, bc2);
        UpdateTensorAdamW(wB, gB, mB, vB, clipScale, lr, wdFactor, beta1, beta2, eps, bc1, bc2);
    }

    private static void UpdateTensorAdamW(
        float[] w, float[] g, float[] m, float[] v,
        float clipScale, float lr, float wdFactor,
        float beta1, float beta2, float eps, float bc1, float bc2)
    {
        int n = w.Length;
        int chunks = Math.Max(1, Environment.ProcessorCount);
        int chunkSize = (n + chunks - 1) / chunks;
        Parallel.For(0, chunks, c =>
        {
            int start = c * chunkSize;
            int end = Math.Min(n, start + chunkSize);
            for (int i = start; i < end; i++)
            {
                float gi = g[i] * clipScale;
                float mi = beta1 * m[i] + (1.0f - beta1) * gi;
                float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
                m[i] = mi;
                v[i] = vi;
                float mHat = mi / bc1;
                float vHat = vi / bc2;
                w[i] = w[i] * wdFactor - lr * mHat / (MathF.Sqrt(vHat) + eps);
            }
        });
    }
}
#endif
