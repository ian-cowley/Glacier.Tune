namespace Glacier.Tune.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Tensor.Core;
using Glacier.Tune.Config;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;
using Glacier.Tune.Trainer;
using Xunit;

public class CheckpointAndOptimizationTests
{
    // =========================================================================
    // 1. Cosine Decay Scheduling Tests
    // =========================================================================

    [Fact]
    public void CosineScheduler_WarmupLinearGrowth_MatchesAnalyticalValues()
    {
        const int totalSteps = 1000;
        const float baseLr = 2e-4f;
        const float warmupRatio = 0.05f; // 50 steps

        // Step 0: 0.0
        float lr0 = LoraTrainer.ComputeCosineLr(0, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lr0);

        // Step 25: exactly 50% of warmup -> 0.5 * baseLr
        float lr25 = LoraTrainer.ComputeCosineLr(25, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr * 0.5f, lr25, precision: 6);

        // Step 50: end of warmup -> exactly baseLr
        float lr50 = LoraTrainer.ComputeCosineLr(50, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr, lr50, precision: 6);
    }

    [Fact]
    public void CosineScheduler_CosineDecay_MatchesAnalyticalValues()
    {
        const int totalSteps = 1000;
        const float baseLr = 1e-4f;
        const float warmupRatio = 0.0f; // No warmup: pure cosine decay

        // Step 0: progress 0.0 -> baseLr * 0.5 * (1 + cos(0)) = baseLr
        float lr0 = LoraTrainer.ComputeCosineLr(0, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr, lr0, precision: 6);

        // Step 500: progress 0.5 -> baseLr * 0.5 * (1 + cos(pi/2)) = 0.5 * baseLr
        float lr500 = LoraTrainer.ComputeCosineLr(500, totalSteps, baseLr, warmupRatio);
        Assert.Equal(baseLr * 0.5f, lr500, precision: 5);

        // Step 1000: progress 1.0 -> baseLr * 0.5 * (1 + cos(pi)) = 0.0
        float lr1000 = LoraTrainer.ComputeCosineLr(1000, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lr1000, precision: 5);

        // Step 1200: clamped beyond totalSteps -> 0.0
        float lrBeyond = LoraTrainer.ComputeCosineLr(1200, totalSteps, baseLr, warmupRatio);
        Assert.Equal(0.0f, lrBeyond, precision: 5);
    }

    [Fact]
    public void CosineScheduler_AnnealedLearningRate_DecaysProportionately()
    {
        const int totalSteps = 200;
        const float baseLr = 2e-4f;
        const float annealedLr = baseLr * 0.5f; // Round 1 annealed LR: 1e-4f

        float lrInitial = LoraTrainer.ComputeCosineLr(0, totalSteps, annealedLr, 0.0f);
        Assert.Equal(annealedLr, lrInitial, precision: 6);

        float lrMid = LoraTrainer.ComputeCosineLr(100, totalSteps, annealedLr, 0.0f);
        Assert.Equal(annealedLr * 0.5f, lrMid, precision: 6);

        float lrFinal = LoraTrainer.ComputeCosineLr(200, totalSteps, annealedLr, 0.0f);
        Assert.Equal(0.0f, lrFinal, precision: 6);
    }

    // =========================================================================
    // 2. SIMD Gradient Norm Clipping Tests
    // =========================================================================

    [Fact]
    public void GradientNormClipping_BelowMaxNorm_PreservesGradients()
    {
        using var dummyModel = SyntheticGgufHelper.CreateTestModel();
        var trainer = new LoraTrainer(dummyModel, new TrainingArguments { MaxGradNorm = 1.0f });

        // Set parameter gradients to values with total norm < 1.0
        var param = dummyModel.TrainableParameters[0];
        param.Grad!.Fill(0.01f); // 64 * 16 elements * 0.0001 = 0.1024 sumSq -> norm ~ 0.32

        var spanBefore = param.Grad.AsSpan().ToArray();
        float norm = trainer.ClipGradients(1.0f);

        Assert.True(norm < 1.0f, $"Norm should be below 1.0, got {norm}");
        var spanAfter = param.Grad.AsSpan().ToArray();

        for (int i = 0; i < spanBefore.Length; i++)
        {
            Assert.Equal(spanBefore[i], spanAfter[i]);
        }
    }

    [Fact]
    public void GradientNormClipping_AboveMaxNorm_ScalesToUnitNormWithSimdVector256()
    {
        using var dummyModel = SyntheticGgufHelper.CreateTestModel();
        var trainer = new LoraTrainer(dummyModel, new TrainingArguments { MaxGradNorm = 1.0f });

        // Fill all gradients with large values so total norm is well above 1.0
        foreach (var p in dummyModel.TrainableParameters)
        {
            p.Grad!.Fill(2.0f);
        }

        float initialNorm = trainer.ClipGradients(1.0f);
        Assert.True(initialNorm > 1.0f, $"Initial norm should exceed 1.0, got {initialNorm}");

        // Now re-calculate post-clipping norm: should be exactly ~1.0
        double postNormSq = 0.0;
        foreach (var p in dummyModel.TrainableParameters)
        {
            var span = p.Grad!.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                postNormSq += (double)span[i] * span[i];
            }
        }
        float postNorm = MathF.Sqrt((float)postNormSq);
        Assert.True(MathF.Abs(postNorm - 1.0f) < 1e-4f, $"Post-clip norm should be 1.0, got {postNorm}");
    }

    // =========================================================================
    // 3. FusedCrossEntropyLoss Hot-Loop Memory & Zero-Allocation Verification
    // =========================================================================

    [Fact]
    public unsafe void FusedCrossEntropyLoss_HotLoopMemory_ZeroGCAllocations()
    {
        const int seqLen = 64;
        const int hiddenDim = 64;
        const int vocabSize = 32;

        float* pHidden = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim) * sizeof(float), 64);
        float* pDHidden = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim) * sizeof(float), 64);
        float* pHead = (float*)NativeMemory.AlignedAlloc((nuint)(vocabSize * hiddenDim) * sizeof(float), 64);

        try
        {
            new Span<float>(pHidden, seqLen * hiddenDim).Fill(0.01f);
            new Span<float>(pHead, vocabSize * hiddenDim).Fill(0.005f);

            int[] inputTokens = new int[seqLen];
            int[] targetTokens = new int[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                inputTokens[i] = i % vocabSize;
                targetTokens[i] = (i + 1) % vocabSize;
            }

            // Warmup steps to initialize thread-local arena, buffers, and complete JIT compilation
            for (int w = 0; w < 3; w++)
            {
                FusedCrossEntropyLoss.ComputeLossAndGradients(
                    pHidden, inputTokens, targetTokens, GgufType.F32, (byte*)pHead, pDHidden, seqLen, hiddenDim, vocabSize);
            }

            // Verify zero GC heap allocations across subsequent hot-loop steps
            for (int iter = 0; iter < 10; iter++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();

                float loss = FusedCrossEntropyLoss.ComputeLossAndGradients(
                    pHidden, inputTokens, targetTokens, GgufType.F32, (byte*)pHead, pDHidden, seqLen, hiddenDim, vocabSize);

                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

                Assert.True(loss > 0f, "Loss should be positive");
                Assert.Equal(0L, allocated);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pHidden);
            NativeMemory.AlignedFree(pDHidden);
            NativeMemory.AlignedFree(pHead);
        }
    }

    // =========================================================================
    // 4. AutogradScratchWorkspace Verification
    // =========================================================================

    [Fact]
    public unsafe void AutogradScratchWorkspace_WrapOverloads_AddressAndShapeIntegrity()
    {
        const int seqLen = 32;
        const int hiddenDim = 64;

        using var ws = AutogradScratchWorkspace.GetOrCreate(seqLen, hiddenDim, 128, 2, 1, 32);
        Assert.Same(ws, AutogradScratchWorkspace.Current);

        // 2D Wrap overload
        using var tensor2D = AutogradScratchWorkspace.Wrap(ws.HiddenScratchA, seqLen, hiddenDim);
        Assert.Equal(2, tensor2D.Rank);
        Assert.Equal(seqLen, tensor2D.Shape[0]);
        Assert.Equal(hiddenDim, tensor2D.Shape[1]);
        Assert.Equal((IntPtr)ws.HiddenScratchA, (IntPtr)tensor2D.DataPointer);

        // 1D Wrap overload
        using var tensor1D = AutogradScratchWorkspace.Wrap(ws.FfnScratchA, seqLen * hiddenDim);
        Assert.Equal(1, tensor1D.Rank);
        Assert.Equal(seqLen * hiddenDim, tensor1D.Shape[0]);
        Assert.Equal((IntPtr)ws.FfnScratchA, (IntPtr)tensor1D.DataPointer);
    }

    // =========================================================================
    // 5. Round-Trip SaveAdapter -> LoadAdapter Mathematical Exactness (All 7 Projections)
    // =========================================================================

    [Fact]
    public void SaveAdapter_LoadAdapter_All7Projections_MathematicalExactness_RoundTrip()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "GlacierLoraTest_" + Guid.NewGuid().ToString("N"));
        try
        {
            using var model = SyntheticGgufHelper.CreateTestModel();
            Assert.True(model.Blocks.Count >= 1);

            var block = model.Blocks[0];

            // 1. Fill all 7 projection matrices (AdapterA and AdapterB) with unique known deterministic floats
            var rng = new Random(1337);

            FillTensor(block.QProj.AdapterA, rng);
            FillTensor(block.QProj.AdapterB, rng);
            FillTensor(block.KProj.AdapterA, rng);
            FillTensor(block.KProj.AdapterB, rng);
            FillTensor(block.VProj.AdapterA, rng);
            FillTensor(block.VProj.AdapterB, rng);
            FillTensor(block.OProj.AdapterA, rng);
            FillTensor(block.OProj.AdapterB, rng);
            FillTensor(block.GateProj.AdapterA, rng);
            FillTensor(block.GateProj.AdapterB, rng);
            FillTensor(block.UpProj.AdapterA, rng);
            FillTensor(block.UpProj.AdapterB, rng);
            FillTensor(block.DownProj.AdapterA, rng);
            FillTensor(block.DownProj.AdapterB, rng);

            // Snapshot expected values
            float[] expectedQA = block.QProj.AdapterA.AsSpan().ToArray();
            float[] expectedQB = block.QProj.AdapterB.AsSpan().ToArray();
            float[] expectedKA = block.KProj.AdapterA.AsSpan().ToArray();
            float[] expectedKB = block.KProj.AdapterB.AsSpan().ToArray();
            float[] expectedVA = block.VProj.AdapterA.AsSpan().ToArray();
            float[] expectedVB = block.VProj.AdapterB.AsSpan().ToArray();
            float[] expectedOA = block.OProj.AdapterA.AsSpan().ToArray();
            float[] expectedOB = block.OProj.AdapterB.AsSpan().ToArray();
            float[] expectedGateA = block.GateProj.AdapterA.AsSpan().ToArray();
            float[] expectedGateB = block.GateProj.AdapterB.AsSpan().ToArray();
            float[] expectedUpA = block.UpProj.AdapterA.AsSpan().ToArray();
            float[] expectedUpB = block.UpProj.AdapterB.AsSpan().ToArray();
            float[] expectedDownA = block.DownProj.AdapterA.AsSpan().ToArray();
            float[] expectedDownB = block.DownProj.AdapterB.AsSpan().ToArray();

            // 2. Save adapter
            model.SaveAdapter(tempDir);
            string binFile = Path.Combine(tempDir, "adapter_model.bin");
            Assert.True(File.Exists(binFile), "adapter_model.bin must exist on disk");

            // 3. Clear/Corrupt all matrices in memory
            block.QProj.AdapterA.Fill(0f);
            block.QProj.AdapterB.Fill(0f);
            block.KProj.AdapterA.Fill(0f);
            block.KProj.AdapterB.Fill(0f);
            block.VProj.AdapterA.Fill(0f);
            block.VProj.AdapterB.Fill(0f);
            block.OProj.AdapterA.Fill(0f);
            block.OProj.AdapterB.Fill(0f);
            block.GateProj.AdapterA.Fill(0f);
            block.GateProj.AdapterB.Fill(0f);
            block.UpProj.AdapterA.Fill(0f);
            block.UpProj.AdapterB.Fill(0f);
            block.DownProj.AdapterA.Fill(0f);
            block.DownProj.AdapterB.Fill(0f);

            // 4. Reload adapter from file and verify exact mathematical equivalence across all 7 projections
            model.LoadAdapter(tempDir);

            AssertTensorsEqual(expectedQA, block.QProj.AdapterA);
            AssertTensorsEqual(expectedQB, block.QProj.AdapterB);
            AssertTensorsEqual(expectedKA, block.KProj.AdapterA);
            AssertTensorsEqual(expectedKB, block.KProj.AdapterB);
            AssertTensorsEqual(expectedVA, block.VProj.AdapterA);
            AssertTensorsEqual(expectedVB, block.VProj.AdapterB);
            AssertTensorsEqual(expectedOA, block.OProj.AdapterA);
            AssertTensorsEqual(expectedOB, block.OProj.AdapterB);
            AssertTensorsEqual(expectedGateA, block.GateProj.AdapterA);
            AssertTensorsEqual(expectedGateB, block.GateProj.AdapterB);
            AssertTensorsEqual(expectedUpA, block.UpProj.AdapterA);
            AssertTensorsEqual(expectedUpB, block.UpProj.AdapterB);
            AssertTensorsEqual(expectedDownA, block.DownProj.AdapterA);
            AssertTensorsEqual(expectedDownB, block.DownProj.AdapterB);

            // 5. Test direct binary file path loading
            block.QProj.AdapterA.Fill(-999f);
            model.LoadAdapter(binFile);
            AssertTensorsEqual(expectedQA, block.QProj.AdapterA);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    // =========================================================================
    // 6. LoraTrainer Checkpoint Resumption Tests
    // =========================================================================

    [Fact]
    public void LoraTrainer_ResumeFromCheckpoint_SetsAnnealedLrAndCustomStartingStep()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "GlacierLoraResume_" + Guid.NewGuid().ToString("N"));
        try
        {
            using var model = SyntheticGgufHelper.CreateTestModel();
            model.SaveAdapter(tempDir);

            var args = new TrainingArguments
            {
                LearningRate = 2e-4f,
                WarmupRatio = 0.05f
            };

            using var trainer = new LoraTrainer(model, args);
            Assert.Equal(0, trainer.StartingStep);

            // Resume with annealed learning rate and custom starting step
            const float annealedLr = 1e-4f;
            const int startingStep = 450;
            trainer.ResumeFromCheckpoint(tempDir, annealedLearningRate: annealedLr, startingStep: startingStep);

            Assert.Equal(startingStep, trainer.StartingStep);
            Assert.Equal(annealedLr, trainer.ResumedLearningRate);
            Assert.Equal(annealedLr, trainer.Optimizer.LearningRate);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    private static void FillTensor(Tensor<float> tensor, Random rng)
    {
        var span = tensor.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
        }
    }

    private static void AssertTensorsEqual(float[] expected, Tensor<float> actualTensor)
    {
        var actual = actualTensor.AsSpan();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }
}

/// <summary>
/// Generates synthetic, standalone GGUF model files for in-memory testing without 8GB physical models.
/// </summary>
internal static class SyntheticGgufHelper
{
    public static GgufLoraModel CreateTestModel(int rank = 16, float alpha = 32f)
    {
        string tempGguf = Path.Combine(Path.GetTempPath(), "SyntheticTest_" + Guid.NewGuid().ToString("N") + ".gguf");
        BuildSyntheticGguf(tempGguf, hiddenDim: 64, ffnDim: 128, nHeadsQ: 2, nHeadsKv: 1, vocabSize: 16);
        var model = GgufLoraModel.Load(tempGguf, new LoraConfig { Rank = rank, Alpha = alpha }, device: "cpu");
        return model;
    }

    public static void BuildSyntheticGguf(string path, int hiddenDim, int ffnDim, int nHeadsQ, int nHeadsKv, int vocabSize)
    {
        int headDim = hiddenDim / nHeadsQ;
        int kvDim = nHeadsKv * headDim;

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // 1. Header
        bw.Write((uint)0x46554747); // 'GGUF'
        bw.Write((uint)3);          // Version 3

        var tensors = new List<(string Name, ulong[] Dims, ulong ByteSize)>
        {
            ("token_embd.weight", [(ulong)hiddenDim, (ulong)vocabSize], (ulong)(hiddenDim * vocabSize * 4)),
            ("output.weight", [(ulong)hiddenDim, (ulong)vocabSize], (ulong)(hiddenDim * vocabSize * 4)),
            ("output_norm.weight", [(ulong)hiddenDim], (ulong)(hiddenDim * 4)),
            ("blk.0.attn_norm.weight", [(ulong)hiddenDim], (ulong)(hiddenDim * 4)),
            ("blk.0.attn_q.weight", [(ulong)hiddenDim, (ulong)hiddenDim], (ulong)(hiddenDim * hiddenDim * 4)),
            ("blk.0.attn_k.weight", [(ulong)hiddenDim, (ulong)kvDim], (ulong)(hiddenDim * kvDim * 4)),
            ("blk.0.attn_v.weight", [(ulong)hiddenDim, (ulong)kvDim], (ulong)(hiddenDim * kvDim * 4)),
            ("blk.0.attn_output.weight", [(ulong)hiddenDim, (ulong)hiddenDim], (ulong)(hiddenDim * hiddenDim * 4)),
            ("blk.0.ffn_norm.weight", [(ulong)hiddenDim], (ulong)(hiddenDim * 4)),
            ("blk.0.ffn_gate.weight", [(ulong)hiddenDim, (ulong)ffnDim], (ulong)(hiddenDim * ffnDim * 4)),
            ("blk.0.ffn_up.weight", [(ulong)hiddenDim, (ulong)ffnDim], (ulong)(hiddenDim * ffnDim * 4)),
            ("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)hiddenDim], (ulong)(ffnDim * hiddenDim * 4)),
        };

        bw.Write((ulong)tensors.Count); // TensorCount

        // 2. Metadata KV pairs
        var metadata = new List<Action>();

        void AddString(string key, string val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)8); // GgufValueType.String
            WriteString(bw, val);
        });

        void AddUInt32(string key, uint val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)4); // GgufValueType.UInt32
            bw.Write(val);
        });

        void AddSingle(string key, float val) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)6); // GgufValueType.Float32
            bw.Write(val);
        });

        void AddStringArray(string key, string[] items) => metadata.Add(() =>
        {
            WriteString(bw, key);
            bw.Write((uint)9); // GgufValueType.Array
            bw.Write((uint)8); // Element type: String
            bw.Write((ulong)items.Length);
            foreach (var s in items) WriteString(bw, s);
        });

        AddString("general.architecture", "qwen2");
        AddUInt32("qwen2.block_count", 1);
        AddUInt32("qwen2.context_length", 512);
        AddUInt32("qwen2.embedding_length", (uint)hiddenDim);
        AddUInt32("qwen2.feed_forward_length", (uint)ffnDim);
        AddUInt32("qwen2.attention.head_count", (uint)nHeadsQ);
        AddUInt32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        AddSingle("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        AddSingle("qwen2.rope.freq_base", 10000.0f);
        AddUInt32("tokenizer.ggml.bos_token_id", 1);
        AddUInt32("tokenizer.ggml.eos_token_id", 2);

        string[] tokens = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++) tokens[i] = $"[tok_{i}]";
        AddStringArray("tokenizer.ggml.tokens", tokens);

        bw.Write((ulong)metadata.Count);
        foreach (var action in metadata) action();

        // 3. Tensor Directory
        ulong currentOffset = 0;
        foreach (var (name, dims, byteSize) in tensors)
        {
            WriteString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (var d in dims) bw.Write(d);
            bw.Write((uint)0); // GgufType.F32 = 0
            bw.Write(currentOffset);
            currentOffset += byteSize;
        }

        // 4. Align to 32 bytes before tensor data
        long pos = fs.Position;
        long rem = pos % 32;
        if (rem != 0)
        {
            int pad = (int)(32 - rem);
            bw.Write(new byte[pad]);
        }

        // 5. Tensor Data (zeros with non-zero diag for stability)
        foreach (var (_, _, byteSize) in tensors)
        {
            byte[] buf = new byte[byteSize];
            // Initialize small float pattern
            for (int i = 0; i < buf.Length; i += 4)
            {
                buf[i] = 0x3D; // ~0.046f
            }
            bw.Write(buf);
        }
    }

    private static void WriteString(BinaryWriter bw, string str)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(str);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }
}
