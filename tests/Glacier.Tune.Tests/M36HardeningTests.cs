using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Tokenizer;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Expansion;
using Glacier.Tune.Export;
using Glacier.Tune.Gpu;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Glacier.Tune.Sleep;
using Glacier.Tune.Trainer;
using Xunit;

namespace Glacier.Tune.Tests;

public sealed unsafe class M36HardeningTests
{
    private sealed class ListLogger(List<string> messages) : IGlacierLogger
    {
        public bool IsEnabled(LogLevel level) => true;
        public void Log(LogLevel level, string message, Exception? exception = null) => messages.Add(message);
    }

    private static string CreateTempDir(string prefix)
    {
        string dir = Path.Combine(Path.GetTempPath(), prefix + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    // =========================================================================
    // Feature #1 — Backend Honesty & CPU Override Removal
    // =========================================================================

    [Fact]
    public void LoraAdapter_PreservesExplicitTargets_AndRejectsUnsupportedGpuTargets()
    {
        using var cpu = new LoraAdapter(32, 16, rank: 4, alpha: 8f, target: GpuTarget.Cpu);
        Assert.Equal(GpuTarget.Cpu, cpu.Target);

        using var nvidia = new LoraAdapter(32, 16, rank: 4, alpha: 8f, target: GpuTarget.Nvidia);
        Assert.Equal(GpuTarget.Nvidia, nvidia.Target);

        using var tensorCore = new LoraAdapter(32, 16, rank: 4, alpha: 8f, target: GpuTarget.NvidiaTensorCore);
        Assert.Equal(GpuTarget.NvidiaTensorCore, tensorCore.Target);

        Assert.Throws<NotSupportedException>(() => new LoraAdapter(32, 16, target: GpuTarget.Direct3D12));
        Assert.Throws<NotSupportedException>(() => new LoraAdapter(32, 16, target: GpuTarget.Vulkan));
        Assert.Throws<NotSupportedException>(() => new LoraAdapter(32, 16, target: GpuTarget.Amd));
        Assert.Throws<NotSupportedException>(() => new LoraAdapter(32, 16, target: GpuTarget.DualGpu));
    }

    [Fact]
    public void GgufLoraModel_ResolveTarget_ParsesAllBackendsAndRejectsUnknown()
    {
        Assert.Equal(GpuTarget.Auto, GgufLoraModel.ResolveTarget(null));
        Assert.Equal(GpuTarget.Auto, GgufLoraModel.ResolveTarget(""));
        Assert.Equal(GpuTarget.Auto, GgufLoraModel.ResolveTarget("auto"));
        Assert.Equal(GpuTarget.Cpu, GgufLoraModel.ResolveTarget("cpu"));
        Assert.Equal(GpuTarget.Nvidia, GgufLoraModel.ResolveTarget("cuda"));
        Assert.Equal(GpuTarget.Nvidia, GgufLoraModel.ResolveTarget("gpu"));
        Assert.Equal(GpuTarget.Nvidia, GgufLoraModel.ResolveTarget("nvidia"));
        Assert.Equal(GpuTarget.NvidiaTensorCore, GgufLoraModel.ResolveTarget("tensorcore"));
        Assert.Equal(GpuTarget.Direct3D12, GgufLoraModel.ResolveTarget("d3d12"));
        Assert.Equal(GpuTarget.Vulkan, GgufLoraModel.ResolveTarget("vulkan"));
        Assert.Equal(GpuTarget.Amd, GgufLoraModel.ResolveTarget("rocm"));
        Assert.Equal(GpuTarget.DualGpu, GgufLoraModel.ResolveTarget("dualgpu"));

        Assert.Throws<ArgumentException>(() => GgufLoraModel.ResolveTarget("tpu_magic"));
    }

    [Fact]
    public void GgufLoraModel_Load_ExplicitUnsupportedGpuOrHybridArch_ThrowsOrFallsBackHonestly()
    {
        string tempDir = CreateTempDir("m36_f1_");
        try
        {
            string ggufPath = Path.Combine(tempDir, "qwen35_hybrid.gguf");
            SyntheticGgufBuilder.WriteQwen35HybridGguf(ggufPath, new SyntheticGgufBuilder.Qwen35Spec
            {
                BlockCount = 2,
                HiddenDim = 16,
                FfnDim = 24,
                HeadCount = 2,
                HeadCountKv = 1,
                HeadDim = 8,
                RopeDim = 4,
                FullAttentionInterval = 2,
                SsmConvKernel = 4,
                SsmStateSize = 4,
                SsmGroupCount = 2,
                SsmTimeStepRank = 4,
                GdnOnEvenLayers = true,
                UseQGate = true,
                UseQkNorm = true,
                Seed = 2026
            });

            Assert.Throws<NotSupportedException>(() => GgufLoraModel.Load(ggufPath, device: "vulkan"));

            var ex = Assert.Throws<NotSupportedException>(() => GgufLoraModel.Load(ggufPath, device: "nvidia"));
            Assert.Contains("GDN", ex.Message, StringComparison.OrdinalIgnoreCase);

            using var autoModel = GgufLoraModel.Load(ggufPath, new LoraConfig { Rank = 4, Alpha = 8f }, device: "auto");
            Assert.False(autoModel.IsGpuAccelerated);
            Assert.Equal(GpuTarget.Cpu, autoModel.Target);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void LoraTrainer_ReportsAccurateExecutionBackendLog()
    {
        var messages = new List<string>();
        GlacierDiagnostics.Logger = new ListLogger(messages);
        try
        {
            using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 32, ffnDim: 64, vocabSize: 32);
            using var model = fixture.CreateLoraModel(device: "cpu");
            using var trainer = new LoraTrainer(model, new TrainingArguments { Epochs = 1, OutputDir = Path.Combine(fixture.DirectoryPath, "out"), LoggingSteps = 1 });

            var ds = new ChatMlDataset();
            var listField = typeof(ChatMlDataset).GetField("_examples", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            ((List<TrainingExample>)listField.GetValue(ds)!).Add(new TrainingExample([1, 2, 3, 4], [-100, 2, 3, 4]));

            trainer.Train(ds);
            Assert.Contains(messages, m => m.Contains("CPU SIMD Accelerated", StringComparison.Ordinal));
            Assert.DoesNotContain(messages, m => m.Contains("GPU VRAM Accelerated", StringComparison.Ordinal));
        }
        finally
        {
            GlacierDiagnostics.Reset();
        }
    }

    // =========================================================================
    // Feature #2 — AutogradScratchWorkspace Hardening & Opt #14 SIMD Parity
    // =========================================================================

    [Fact]
    public void AutogradScratchWorkspace_ValidatesArgs_WrapsBounds_AndDisposesCleanly()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.GetOrCreate(0, 16, 32, 2, 1, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.GetOrCreate(4, -1, 32, 2, 1, 8));

        var ws = AutogradScratchWorkspace.GetOrCreate(8, 32, 64, 4, 2, 8);
        using (var t = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, 8, 32))
        {
            Assert.Equal(8, t.Shape[0]);
            Assert.Equal(32, t.Shape[1]);
        }

        Assert.Throws<ArgumentNullException>(() => AutogradScratchWorkspace.Wrap(null!, 4, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, 0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, 100_000, 100_000));

        AutogradScratchWorkspace.DisposeAllThreadLocal();
        var ws2 = AutogradScratchWorkspace.GetOrCreate(4, 16, 32, 2, 1, 8);
        Assert.True(ws2.HiddenScratchA != null);
        AutogradScratchWorkspace.DisposeAllThreadLocal();
    }

    [Fact]
    public void AutogradScratchWorkspace_OomDuringEnsureCapacity_DoesNotPoisonThreadLocalOrInstance()
    {
        AutogradScratchWorkspace.DisposeAllThreadLocal();
        try
        {
            var initial = AutogradScratchWorkspace.GetOrCreate(4, 16, 32, 2, 1, 8);
            Assert.True(initial.HiddenScratchA != null);

            Assert.ThrowsAny<Exception>(() => AutogradScratchWorkspace.GetOrCreate(int.MaxValue, 1024, 1024, 8, 2, 64));

            var recovered = AutogradScratchWorkspace.GetOrCreate(8, 16, 32, 2, 1, 8);
            Assert.True(recovered.FfnScratchA != null);
            Assert.True(recovered.FfnScratchB != null);
            Assert.True(recovered.FfnScratchC != null);
            Assert.True(recovered.HiddenScratchA != null);
            Assert.True(recovered.HiddenScratchB != null);
            Assert.True(recovered.HiddenScratchC != null);
            Assert.True(recovered.HiddenScratchD != null);
            Assert.True(recovered.HiddenScratchE != null);
            Assert.True(recovered.HiddenScratchF != null);
            Assert.True(recovered.QDimScratchA != null);
            Assert.True(recovered.QDimScratchB != null);
            Assert.True(recovered.KvDimScratchA != null);
            Assert.True(recovered.KvDimScratchB != null);

            using (var wrapped = AutogradScratchWorkspace.Wrap(recovered.HiddenScratchA, 8, 16))
            {
                wrapped.AsSpan()[^1] = 3.5f;
                Assert.Equal(3.5f, recovered.HiddenScratchA[8 * 16 - 1]);
            }

            // Invalid argument failure must not poison an existing valid thread-local workspace
            Assert.Throws<ArgumentOutOfRangeException>(() => AutogradScratchWorkspace.GetOrCreate(-1, 16, 32, 2, 1, 8));
            Assert.Same(recovered, AutogradScratchWorkspace.GetOrCreate(8, 16, 32, 2, 1, 8));

            // Direct instance EnsureCapacity must also recover cleanly after OOM
            using var direct = new AutogradScratchWorkspace(4, 16, 32, 2, 1, 8);
            Assert.ThrowsAny<Exception>(() => direct.EnsureCapacity(int.MaxValue, 1024, 1024, 8, 2, 64));
            direct.EnsureCapacity(8, 16, 32, 2, 1, 8);
            Assert.True(direct.HiddenScratchA != null);
            using (var wrappedDirect = AutogradScratchWorkspace.Wrap(direct.FfnScratchA, 8, 32))
            {
                wrappedDirect.AsSpan()[^1] = 7.25f;
                Assert.Equal(7.25f, direct.FfnScratchA[8 * 32 - 1]);
            }
        }
        finally
        {
            AutogradScratchWorkspace.DisposeAllThreadLocal();
        }
    }

    // =========================================================================
    // Feature #3 — GdnScratchWorkspace Alignment & Zero-Leak Finite-Diff Parity
    // =========================================================================

    [Fact]
    public void GdnScratchWorkspace_AlignedPointers_AndReusableAcrossCalls()
    {
        var wsFwd = GdnScratchWorkspace.GetOrCreateForForward(ssmHeads: 4, stateDim: 8);
        wsFwd.GetForwardHeadScratch(0, 8, out float* s0, out float* d0);
        wsFwd.GetForwardHeadScratch(1, 8, out float* s1, out _);
        Assert.Equal(0UL, (ulong)s0 % 64UL);
        Assert.Equal(0UL, (ulong)s1 % 64UL);
        Assert.Equal(0UL, (ulong)d0 % 64UL);

        var wsBwd = GdnScratchWorkspace.GetOrCreateForBackward(
            ssmHeads: 4, stateDim: 8, seqLen: 8, ssmInnerSize: 32, qkGroupFloats: 16, gdnConvChannels: 64);
        wsBwd.GetBackwardHeadScratch(0, 8, out float* sPrev, out _, out _, out float* sCur, out _, out _, out float* dS, out _, out _);
        Assert.Equal(0UL, (ulong)sPrev % 64UL);
        Assert.Equal(0UL, (ulong)sCur % 64UL);
        Assert.Equal(0UL, (ulong)dS % 64UL);
        wsBwd.GetBackwardActivationBuffers(out float* dYGated, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out float* dQkv);
        Assert.True(dYGated != null && dQkv != null);
        GdnScratchWorkspace.DisposeAllThreadLocal();
    }

    // =========================================================================
    // Feature #4 — SleepConsolidationTrainer KL + DreamEmbedding & DreamQualityFilter
    // =========================================================================

    [Fact]
    public void DreamQualityFilter_EmptyStringsWithSameVector_PreservesCosineContribution_AndMyersMatchesDp()
    {
        float[] v1 = new float[23];
        float[] v2 = new float[23];
        for (int i = 0; i < 23; i++)
        {
            v1[i] = MathF.Sin(i + 1);
            v2[i] = MathF.Sin(i + 1);
        }

        float cos = DreamQualityFilter.CosineSimilarity(v1, v2);
        Assert.Equal(1.0f, cos, precision: 5);

        float emptyScore = DreamQualityFilter.ComputeScore("", "", v1, v2, alpha: 0.65f);
        Assert.Equal(1.0f, emptyScore, precision: 4);

        string[] samplesA = ["", "kitten", "abcdef0123456789", new string('a', 64), new string('x', 80) + "middle" + new string('y', 40)];
        string[] samplesB = ["", "sitting", "abczef0123456780", new string('b', 64), new string('x', 80) + "muddle" + new string('y', 40)];
        for (int i = 0; i < samplesA.Length; i++)
        {
            int fast = DreamQualityFilter.FastLevenshtein(samplesA[i], samplesB[i]);
            int refDist = NaiveLevenshtein(samplesA[i], samplesB[i]);
            Assert.Equal(refDist, fast);
        }
    }

    [Fact]
    public async Task SleepConsolidationTrainer_UsesDreamEmbeddingAndRealKlGradientAsynchronously()
    {
        const int dim = 16;
        using var bank = new PreallocatedMoeExpansionBank(dim, rank: 4, maxExperts: 2);
        float[] centroid = new float[dim];
        centroid[0] = 1.0f;
        int slot = bank.UnlockNextExpertSlot(centroid);
        bank.ActivateConsolidatedSlot(slot);

        float[] qVec = new float[dim];
        float[] dreamVec = new float[dim];
        float[] teacherOut = new float[dim];
        qVec[0] = 0.8f;
        dreamVec[1] = 1.0f;
        teacherOut[2] = 0.5f;

        // Verify backward compatibility with 7-arg positional DreamCandidate constructor
        var legacyCandidate = new DreamCandidate("prompt", "dream", qVec, teacherOut, 0.9f, 0.9f, 0.9f);
        Assert.Null(legacyCandidate.DreamEmbedding);

        var dreamCandidate = legacyCandidate with { DreamEmbedding = dreamVec };

        using var mseOnly = new SleepConsolidationTrainer(bank, klBeta: 0f, klWeight: 0f);
        var resNoKl = await mseOnly.ConsolidateKnowledgeAsync(slot, [dreamCandidate], distillationSteps: 3, learningRate: 0.05f, qualityThreshold: 0.1f);
        Assert.True(resNoKl.InitialGkdLoss > 0f && float.IsFinite(resNoKl.FinalGkdLoss));
        Assert.Equal(1, resNoKl.TotalDreamsAccepted);

        using var bankKl = new PreallocatedMoeExpansionBank(dim, rank: 4, maxExperts: 2);
        int slotKl = bankKl.UnlockNextExpertSlot(centroid);
        bankKl.ActivateConsolidatedSlot(slotKl);

        using var withKl = new SleepConsolidationTrainer(bankKl, klBeta: 0.5f, klWeight: 0.4f, klTemperature: 1.0f);
        var resWithKl = await withKl.ConsolidateKnowledgeAsync(slotKl, [dreamCandidate], distillationSteps: 3, learningRate: 0.05f, qualityThreshold: 0.1f);
        Assert.True(resWithKl.InitialGkdLoss > resNoKl.InitialGkdLoss, $"Expected KL penalty to raise initial loss ({resWithKl.InitialGkdLoss} vs {resNoKl.InitialGkdLoss}).");

        // Verify DreamEmbedding drives updates on MatrixB (B_e * dreamVec) -> non-zero column 1 in B_e
        float* bPtr = bank.GetMatrixBPtr(slot);
        float col1Norm = 0f;
        for (int r = 0; r < 4; r++) col1Norm += MathF.Abs(bPtr[r * dim + 1]);
        Assert.True(col1Norm > 1e-6f, "Expected DreamEmbedding dimension 1 to drive non-zero gradient into MatrixB column 1.");
    }

    // =========================================================================
    // Feature #5 — Inkling2Dataset Label Alignment, Binary Format Guards & CachedHiddenDataset
    // =========================================================================

    [Fact]
    public void Inkling2Dataset_ShiftsLabelsLikeChatMlDataset_AndSkipsMalformedJsonGracefully()
    {
        string tempDir = CreateTempDir("m36_inkling2_");
        try
        {
            string tokPath = Path.Combine(tempDir, "tok.gguf");
            string dataPath = Path.Combine(tempDir, "data.jsonl");
            BuildMinimalTokenizerGguf(tokPath);

            using var gguf = new GgufFile(tokPath);
            var tokenizer = new BpeTokenizer(gguf);

            File.WriteAllText(dataPath,
                "{\"not_json\n" +
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}\n");

            var ds = Inkling2Dataset.FromFile(dataPath, tokenizer, maxSeqLength: 128);
            Assert.Equal(1, ds.Count);
            var ex = ds.Examples[0];
            Assert.True(ex.InputIds.Length >= 4);
            Assert.Equal(Inkling2Dataset.IgnoreIndex, ex.Labels[^1]);

            int supervisedPositions = 0;
            for (int t = 0; t < ex.InputIds.Length - 1; t++)
            {
                if (ex.Labels[t] != Inkling2Dataset.IgnoreIndex)
                {
                    Assert.Equal(ex.InputIds[t + 1], ex.Labels[t]);
                    supervisedPositions++;
                }
            }
            Assert.True(supervisedPositions > 0);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void LoraBinaryFormat_AndCachedHiddenDataset_RejectCorruptedPayloads()
    {
        string tempDir = CreateTempDir("m36_bin_");
        try
        {
            using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 32, ffnDim: 64, vocabSize: 32, rank: 4);
            using var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f }, device: "cpu");

            string validPath = Path.Combine(tempDir, "valid.bin");
            model.SaveAdapter(validPath);
            model.LoadAdapter(validPath);

            string trailingPath = Path.Combine(tempDir, "trailing.bin");
            File.Copy(validPath, trailingPath);
            File.AppendAllBytes(trailingPath, [0x01, 0x02]);
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(trailingPath));

            string badLayers = Path.Combine(tempDir, "bad_layers.bin");
            byte[] raw = File.ReadAllBytes(validPath);
            BitConverter.TryWriteBytes(raw.AsSpan(8, 4), 999_999);
            File.WriteAllBytes(badLayers, raw);
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(badLayers));

            string nanWeights = Path.Combine(tempDir, "nan_weights.bin");
            byte[] rawNan = File.ReadAllBytes(validPath);
            BitConverter.TryWriteBytes(rawNan.AsSpan(rawNan.Length - 4, 4), float.NaN);
            File.WriteAllBytes(nanWeights, rawNan);
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(nanWeights));

            string cachePath = Path.Combine(tempDir, "hidden.cache");
            var sampleVec = new Half[8];
            for (int i = 0; i < 8; i++) sampleVec[i] = (Half)(i * 0.25f);
            var ds = new Inkling2HeadLoraTrainer.CachedHiddenDataset(8, sampleVec, [3]);
            ds.Save(cachePath);

            var loaded = Inkling2HeadLoraTrainer.CachedHiddenDataset.Load(cachePath);
            Assert.Equal(8, loaded.EmbdDim);
            Assert.Equal(1, loaded.Count);

            string badCachePath = Path.Combine(tempDir, "hidden_bad.cache");
            byte[] cBytes = File.ReadAllBytes(cachePath);
            BitConverter.TryWriteBytes(cBytes.AsSpan(4, 4), -5);
            File.WriteAllBytes(badCachePath, cBytes);
            Assert.Throws<InvalidDataException>(() => Inkling2HeadLoraTrainer.CachedHiddenDataset.Load(badCachePath));

            // Truncated 0x41524F4C payloads must uniformly throw InvalidDataException across LoadAdapter and GgufMerger.Merge
            static byte[] BuildLoraHeader(int tensorCount)
            {
                byte[] hdr = new byte[24];
                BitConverter.TryWriteBytes(hdr.AsSpan(0, 4), 0x41524F4Cu);
                BitConverter.TryWriteBytes(hdr.AsSpan(4, 4), 1u);
                BitConverter.TryWriteBytes(hdr.AsSpan(8, 4), 1);
                BitConverter.TryWriteBytes(hdr.AsSpan(12, 4), 4);
                BitConverter.TryWriteBytes(hdr.AsSpan(16, 4), 8.0f);
                BitConverter.TryWriteBytes(hdr.AsSpan(20, 4), tensorCount);
                return hdr;
            }

            // 1) 33-byte truncated tensor dims (1B len + 'a' + 4B dim0 + 3B partial dim1)
            string truncDimsPath = Path.Combine(tempDir, "trunc_dims_9b.bin");
            using (var ms = new MemoryStream())
            {
                ms.Write(BuildLoraHeader(tensorCount: 1));
                ms.WriteByte(1);
                ms.WriteByte((byte)'a');
                ms.Write(BitConverter.GetBytes(4));
                ms.Write([0x04, 0x00, 0x00]); // 3 of 4 bytes of dim1
                File.WriteAllBytes(truncDimsPath, ms.ToArray());
            }
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(truncDimsPath));
            Assert.Throws<InvalidDataException>(() =>
                GgufMerger.Merge(fixture.GgufPath, truncDimsPath, Path.Combine(tempDir, "out_trunc_dims.gguf")));

            // 2) Truncated mid-varint name length (trailing continuation byte 0x81 after valid tensor #1)
            string truncVarintPath = Path.Combine(tempDir, "trunc_varint.bin");
            using (var ms = new MemoryStream())
            {
                ms.Write(BuildLoraHeader(tensorCount: 2));
                ms.WriteByte(1);
                ms.WriteByte((byte)'x');
                ms.Write(BitConverter.GetBytes(1));
                ms.Write(BitConverter.GetBytes(1));
                ms.Write(BitConverter.GetBytes(0.5f));
                ms.WriteByte(0x81); // truncated varint at EOF for tensor #2
                File.WriteAllBytes(truncVarintPath, ms.ToArray());
            }
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(truncVarintPath));
            Assert.Throws<InvalidDataException>(() =>
                GgufMerger.Merge(fixture.GgufPath, truncVarintPath, Path.Combine(tempDir, "out_trunc_varint.gguf")));

            // 3) 5-byte varint with high bits shifted out (0x10 << 28)
            string wrapVarintPath = Path.Combine(tempDir, "wrap_varint.bin");
            using (var ms = new MemoryStream())
            {
                ms.Write(BuildLoraHeader(tensorCount: 1));
                ms.Write([0x85, 0x80, 0x80, 0x80, 0x10]);
                ms.Write(new byte[16]);
                File.WriteAllBytes(wrapVarintPath, ms.ToArray());
            }
            Assert.Throws<InvalidDataException>(() => model.LoadAdapter(wrapVarintPath));
            Assert.Throws<InvalidDataException>(() =>
                GgufMerger.Merge(fixture.GgufPath, wrapVarintPath, Path.Combine(tempDir, "out_wrap_varint.gguf")));

            // 4) Truncated StatefulAdamW state file must throw InvalidDataException
            var dummyParam = new Tensor<float>(4);
            using var opt = new StatefulAdamW([dummyParam]);
            string truncOptHeader = Path.Combine(tempDir, "trunc_opt_hdr.bin");
            File.WriteAllBytes(truncOptHeader, [0x41, 0x44, 0x4D, 0x57, 1, 0, 0, 0, 0, 0]);
            Assert.Throws<InvalidDataException>(() => opt.LoadState(truncOptHeader));

            string optFull = Path.Combine(tempDir, "full_opt.bin");
            opt.SaveState(optFull);
            byte[] optBytes = File.ReadAllBytes(optFull);
            string truncOptBody = Path.Combine(tempDir, "trunc_opt_body.bin");
            File.WriteAllBytes(truncOptBody, optBytes[..^4]);
            Assert.Throws<InvalidDataException>(() => opt.LoadState(truncOptBody));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    // =========================================================================
    // Feature #6 — CPU SIMD Kernel Parity (SwiGLU, RoPE, CausalAttn, Arbitrary Rank LoRA) + GPU Residency
    // =========================================================================

    [Theory]
    [InlineData(1, 32)]
    [InlineData(3, 64)]
    [InlineData(7, 97)]
    [InlineData(24, 133)]
    public void LoraKernels_ArbitraryRank_ForwardAndBackward_MatchScalarReference(int rank, int features)
    {
        const int seqLen = 5;
        int outFeatures = features + 11;
        var rng = new Random(1234 + rank);

        float[] x = RandomArray(rng, seqLen * features);
        float[] a = RandomArray(rng, features * rank);
        float[] b = RandomArray(rng, rank * outFeatures);
        float[] dy = RandomArray(rng, seqLen * outFeatures);
        float scaling = 2.0f / rank;

        float[] ySimd = new float[seqLen * outFeatures];
        float[] yRef = new float[seqLen * outFeatures];
        float[] gradASimd = new float[a.Length];
        float[] gradARef = new float[a.Length];
        float[] gradBSimd = new float[b.Length];
        float[] gradBRef = new float[b.Length];
        float[] dxSimd = new float[x.Length];
        float[] dxRef = new float[x.Length];

        fixed (float* pX = x, pA = a, pB = b, pDY = dy, pY = ySimd, pGA = gradASimd, pGB = gradBSimd, pDX = dxSimd)
        {
            LoraKernels.ApplyLoraDelta(pX, pA, pB, pY, seqLen, features, outFeatures, rank, scaling);
            LoraKernels.Backward(pX, pDY, pA, pB, pGA, pGB, pDX, seqLen, features, outFeatures, rank, scaling);
        }

        float[] lr = new float[seqLen * rank];
        for (int s = 0; s < seqLen; s++)
            for (int r = 0; r < rank; r++)
                for (int k = 0; k < features; k++)
                    lr[s * rank + r] += x[s * features + k] * a[k * rank + r];

        for (int s = 0; s < seqLen; s++)
            for (int r = 0; r < rank; r++)
                for (int n = 0; n < outFeatures; n++)
                    yRef[s * outFeatures + n] += scaling * lr[s * rank + r] * b[r * outFeatures + n];

        for (int r = 0; r < rank; r++)
            for (int s = 0; s < seqLen; s++)
                for (int n = 0; n < outFeatures; n++)
                    gradBRef[r * outFeatures + n] += scaling * lr[s * rank + r] * dy[s * outFeatures + n];

        float[] dlr = new float[seqLen * rank];
        for (int s = 0; s < seqLen; s++)
            for (int r = 0; r < rank; r++)
                for (int n = 0; n < outFeatures; n++)
                    dlr[s * rank + r] += dy[s * outFeatures + n] * b[r * outFeatures + n];

        for (int k = 0; k < features; k++)
            for (int r = 0; r < rank; r++)
                for (int s = 0; s < seqLen; s++)
                    gradARef[k * rank + r] += scaling * x[s * features + k] * dlr[s * rank + r];

        for (int s = 0; s < seqLen; s++)
            for (int k = 0; k < features; k++)
                for (int r = 0; r < rank; r++)
                    dxRef[s * features + k] += scaling * dlr[s * rank + r] * a[k * rank + r];

        AssertArrayClose(yRef, ySimd, 1e-4f);
        AssertArrayClose(gradARef, gradASimd, 1e-4f);
        AssertArrayClose(gradBRef, gradBSimd, 1e-4f);
        AssertArrayClose(dxRef, dxSimd, 1e-4f);
    }

    [Fact]
    public void SwiGlu_And_RoPE_MatchReferenceValues()
    {
        var rng = new Random(777);
        const int n = 103;
        float[] gate = RandomArray(rng, n);
        float[] up = RandomArray(rng, n);
        float[] dh = RandomArray(rng, n);
        float[] hidden = new float[n];
        float[] dGate = new float[n];
        float[] dUp = new float[n];

        fixed (float* pG = gate, pU = up, pH = hidden, pDH = dh, pDG = dGate, pDU = dUp)
        {
            SwiGluKernel.Forward(pG, pU, pH, n);
            SwiGluKernel.Backward(pDH, pG, pU, pDG, pDU, n);
        }

        for (int i = 0; i < n; i++)
        {
            float sig = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            float silu = gate[i] * sig;
            Assert.True(MathF.Abs(hidden[i] - silu * up[i]) < 1e-5f);
            Assert.True(MathF.Abs(dUp[i] - dh[i] * silu) < 1e-5f);
            float dSilu = sig * (1.0f + gate[i] * (1.0f - sig));
            Assert.True(MathF.Abs(dGate[i] - dh[i] * up[i] * dSilu) < 1e-5f);
        }

        const int seqLen = 6, hq = 2, hkv = 1, headDim = 36;
        float[] q = RandomArray(rng, seqLen * hq * headDim);
        float[] k = RandomArray(rng, seqLen * hkv * headDim);
        float[] qOrig = (float[])q.Clone();
        float[] kOrig = (float[])k.Clone();
        fixed (float* pQ = q, pK = k)
        {
            RoPEKernel.ForwardSequence(pQ, pK, seqLen, hq, hkv, headDim, 10000f);
            RoPEKernel.BackwardSequence(pQ, pK, seqLen, hq, hkv, headDim, 10000f);
        }
        AssertArrayClose(qOrig, q, 1e-5f);
        AssertArrayClose(kOrig, k, 1e-5f);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void GpuLoraEngine_FlashAttention2_SupportsArbitrarySeqLen_AndLossScaleResidentParity()
    {
        if (!GpuContext.IsSupported) return;

        using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(
            layerCount: 2, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 32, rank: 8, alpha: 16f, weightType: GgufType.F32, zeroMeanWeights: true);

        int[] inputTokens = [1, 4, 9, 14, 19];
        int[] targetTokens = [-100, 9, 14, 19, 3];
        const float lossScale = 0.35f;

        float[] cpuGrad;
        float lossCpu;
        using (var cpuModel = fixture.CreateLoraModel(new LoraConfig { Rank = 8, Alpha = 16f, Seed = 55 }, device: "cpu"))
        {
            foreach (var p in cpuModel.Blocks[0].OProj.TrainableParameters)
                for (int i = 0; i < p.ElementCount; i++) p.AsSpan()[i] += 0.03f;
            lossCpu = cpuModel.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false, lossScale: lossScale);
            cpuGrad = cpuModel.Blocks[0].QProj.AdapterB.Grad!.AsSpan().ToArray();
        }

        float[] gpuGrad;
        float lossGpu;
        using (var gpuModel = fixture.CreateLoraModel(new LoraConfig { Rank = 8, Alpha = 16f, Seed = 55 }, device: "gpu"))
        {
            Assert.True(gpuModel.IsGpuAccelerated);
            Assert.True(GpuLoraEngine.Current!.SupportsAttention(20_000, 64));

            foreach (var p in gpuModel.Blocks[0].OProj.TrainableParameters)
                for (int i = 0; i < p.ElementCount; i++) p.AsSpan()[i] += 0.03f;
            lossGpu = gpuModel.ForwardLossAndBackward(inputTokens, targetTokens, checkpointActivations: false, lossScale: lossScale);
            gpuGrad = gpuModel.Blocks[0].QProj.AdapterB.Grad!.AsSpan().ToArray();
        }

        Assert.True(MathF.Abs(lossCpu - lossGpu) < 1e-4f, $"Loss parity mismatch: CPU={lossCpu}, GPU={lossGpu}");
        AssertArrayClose(cpuGrad, gpuGrad, 1e-3f);
    }

    // =========================================================================
    // Feature #7 — StatefulAdamW, DpoTrainer, GgufMerger Q6_K/F16/BF16 & Streaming
    // =========================================================================

    [Fact]
    public void StatefulAdamW_SaveLoadResume_MatchesUninterruptedTrajectory()
    {
        string tempDir = CreateTempDir("m36_adamw_");
        try
        {
            var pFull = new Tensor<float>(8, 4);
            var pSplit = new Tensor<float>(8, 4);
            pFull.Grad = Tensor<float>.Zeros(8, 4);
            pSplit.Grad = Tensor<float>.Zeros(8, 4);
            for (int i = 0; i < 32; i++)
            {
                pFull.AsSpan()[i] = 0.1f * (i - 10);
                pSplit.AsSpan()[i] = 0.1f * (i - 10);
            }

            using var optFull = new StatefulAdamW([pFull], lr: 1e-3f, beta1: 0.9f, beta2: 0.999f, weightDecay: 0.01f);
            using var optPart1 = new StatefulAdamW([pSplit], lr: 1e-3f, beta1: 0.9f, beta2: 0.999f, weightDecay: 0.01f);

            for (int step = 0; step < 5; step++)
            {
                for (int i = 0; i < 32; i++)
                {
                    float g = MathF.Sin(step + i * 0.3f);
                    pFull.Grad.AsSpan()[i] = g;
                    pSplit.Grad.AsSpan()[i] = g;
                }
                optFull.Step();
                optPart1.Step();
            }

            string optPath = Path.Combine(tempDir, "adamw.opt");
            optPart1.SaveState(optPath);

            using var optPart2 = new StatefulAdamW([pSplit], lr: 999f, beta1: 0.9f, beta2: 0.999f, weightDecay: 0.01f);
            optPart2.LoadState(optPath);
            Assert.Equal(5, optPart2.StepCount);
            Assert.Equal(1e-3f, optPart2.LearningRate);

            for (int step = 5; step < 10; step++)
            {
                for (int i = 0; i < 32; i++)
                {
                    float g = MathF.Sin(step + i * 0.3f);
                    pFull.Grad.AsSpan()[i] = g;
                    pSplit.Grad.AsSpan()[i] = g;
                }
                optFull.Step();
                optPart2.Step();
            }

            AssertArrayClose(pFull.AsSpan().ToArray(), pSplit.AsSpan().ToArray(), 0f);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void DpoDataset_Tokenize_And_DpoTrainer_RefCacheAndAccumulationWorkCorrectly()
    {
        string tempDir = CreateTempDir("m36_dpo_");
        try
        {
            string tokPath = Path.Combine(tempDir, "tok.gguf");
            BuildMinimalTokenizerGguf(tokPath);
            using var tokGguf = new GgufFile(tokPath);
            var tokenizer = new BpeTokenizer(tokGguf);

            var pairs = new[]
            {
                new DpoPreferencePair("hi", "ok", "no"),
                new DpoPreferencePair("", "yes", "bad", PromptMessages: [new ChatMessage { Role = "system", Content = "s" }, new ChatMessage { Role = "user", Content = "hi" }])
            };
            var tokenized = DpoDataset.Tokenize(pairs, tokenizer, maxSeqLength: 128);
            Assert.Equal(2, tokenized.Count);
            Assert.Equal(ChatMlDataset.IgnoreIndex, tokenized[0].ChosenLabels[^1]);

            using var fixture = SyntheticGgufFixture.CreateTinyQwenModel(layerCount: 1, hiddenDim: 32, ffnDim: 64, vocabSize: 32, rank: 4);
            using var model = fixture.CreateLoraModel(new LoraConfig { Rank = 4, Alpha = 8f, Seed = 42 }, device: "cpu");

            string cachePath = Path.Combine(tempDir, "ref_logps.bin");
            string outDir = Path.Combine(tempDir, "dpo_out");

            var synPairs = new[]
            {
                new TokenizedPreferencePair([1, 2, 3, 4], [-100, 2, 3, 4], [1, 2, 5, 6], [-100, 2, 5, 6]),
                new TokenizedPreferencePair([1, 3, 4, 5], [-100, 3, 4, 5], [1, 3, 6, 7], [-100, 3, 6, 7])
            };

            using (var dpo1 = new DpoTrainer(model, new DpoTrainingArguments
            {
                Beta = 0.1,
                GradientAccumulationSteps = 2,
                SftAnchorWeight = 0.2,
                ReferenceLogProbsCachePath = cachePath,
                OutputDir = outDir,
                Epochs = 2,
                LearningRate = 1e-3f
            }))
            {
                var res = dpo1.Train(synPairs);
                Assert.Equal(2, res.Count);
                Assert.All(res, r => Assert.True(double.IsFinite(r.Loss)));
            }

            Assert.True(File.Exists(cachePath));
            Assert.True(File.Exists(Path.Combine(outDir, "adapter_model.bin")));
            Assert.True(File.Exists(Path.Combine(outDir, "optimizer_state.bin")));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void GgufMerger_Q6K_F16_BF16_QuantizeRoundtrip_And_MergeQ6K_ProduceValidModel()
    {
        const int count = 256;
        var rng = new Random(99);
        float* src = stackalloc float[count];
        float* dec = stackalloc float[count];
        for (int i = 0; i < count; i++) src[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 3.5);

        BlockQ6_K* q6k = stackalloc BlockQ6_K[1];
        GgufMerger.QuantizeRowQ6_K(src, q6k, count);
        GgufMerger.DequantizeRowQ6_K(q6k, dec, count);
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(src[i] - dec[i]) < 0.08f, $"Q6_K error too large at {i}: src={src[i]}, dec={dec[i]}");
        }

        Half* f16 = stackalloc Half[count];
        GgufMerger.QuantizeRowF16(src, f16, count);
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(src[i] - (float)f16[i]) < 0.004f);
        }

        ushort* bf16 = stackalloc ushort[count];
        GgufMerger.QuantizeRowBF16(src, bf16, count);
        QuantKernels.DequantizeBF16(bf16, dec, count);
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(src[i] - dec[i]) < 0.04f);
        }

        using var fixture = SyntheticGgufFixture.CreateSuperBlockQwenModel(hiddenDim: 256, ffnDim: 512);
        string mergedPath = Path.Combine(fixture.DirectoryPath, "merged_q6k.gguf");
        var mergeResult = GgufMerger.Merge(fixture.GgufPath, fixture.AdapterPath, mergedPath,
            new GgufMerger.MergeOptions { TargetQuantization = TargetQuantization.Q6_K });
        Assert.Equal(7, mergeResult.AdaptedTensorCount);

        using var mergedGguf = new GgufFile(mergedPath);
        Assert.Equal(GgufType.Q6_K, mergedGguf.Tensors["blk.0.attn_q.weight"].Type);
    }

    private static float[] RandomArray(Random rng, int len)
    {
        float[] a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static void AssertArrayClose(float[] expected, float[] actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tol, $"Mismatch at [{i}]: expected {expected[i]}, got {actual[i]} (diff={diff} > tol={tol})");
        }
    }

    private static int NaiveLevenshtein(string a, string b)
    {
        int[,] dp = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++) dp[i, 0] = i;
        for (int j = 0; j <= b.Length; j++) dp[0, j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
            }
        }
        return dp[a.Length, b.Length];
    }

    private static void BuildMinimalTokenizerGguf(string path)
    {
        var vocabList = new List<string>
        {
            "<unk>", "<|im_start|>", "<|im_end|>", "\u010A", "\u0120",
            "<|message_system|>", "<|message_user|>", "<|message_model|>", "<|message_tool|>",
            "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"
        };
        for (char c = '!'; c <= '~'; c++) vocabList.Add(c.ToString());

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);
        static void WriteStr(BinaryWriter w, string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        bw.Write((uint)0x46554747);
        bw.Write((uint)3);
        bw.Write((ulong)0);
        bw.Write((ulong)4);

        WriteStr(bw, "general.architecture"); bw.Write((uint)8); WriteStr(bw, "qwen2");
        WriteStr(bw, "tokenizer.ggml.model"); bw.Write((uint)8); WriteStr(bw, "gpt2");
        WriteStr(bw, "tokenizer.ggml.tokens"); bw.Write((uint)9); bw.Write((uint)8); bw.Write((ulong)vocabList.Count);
        foreach (string t in vocabList) WriteStr(bw, t);

        WriteStr(bw, "tokenizer.ggml.token_type"); bw.Write((uint)9); bw.Write((uint)5); bw.Write((ulong)vocabList.Count);
        foreach (string t in vocabList)
        {
            bw.Write(t.StartsWith("<|", StringComparison.Ordinal) ? 3 : 1);
        }
    }
}
