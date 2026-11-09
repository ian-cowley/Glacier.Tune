namespace Glacier.Tune.Tests;

using System;
using System.Threading.Tasks;
using Glacier.Tune.Expansion;
using Glacier.Tune.Sleep;
using Xunit;

public class SleepConsolidationTests
{
    [Fact]
    public void PreallocatedMoeExpansionBank_DescriptorsAndMemory_Are64ByteAligned()
    {
        unsafe
        {
            Assert.Equal(64, sizeof(MoeExpertSlotDescriptor));

            using var bank = new PreallocatedMoeExpansionBank(dimension: 32, rank: 8, maxExperts: 4);
            Assert.Equal(32, bank.Dimension);
            Assert.Equal(8, bank.Rank);
            Assert.Equal(4, bank.MaxExperts);
            Assert.Equal(0, bank.ActiveExpertCount);

            float[] initCentroid = new float[32];
            initCentroid[0] = 1.0f;
            int slot = bank.UnlockNextExpertSlot(initCentroid);
            Assert.Equal(0, slot);
            Assert.Equal(1, bank.ActiveExpertCount);

            float* aPtr = bank.GetMatrixAPtr(0);
            float* bPtr = bank.GetMatrixBPtr(0);
            float* cPtr = bank.GetCentroidPtr(0);

            Assert.Equal(0UL, (ulong)aPtr % 64UL);
            Assert.Equal(0UL, (ulong)bPtr % 64UL);
            Assert.Equal(0UL, (ulong)cPtr % 64UL);

            // Verify Matrix A starts strictly zero (0 initial perturbation)
            for (int i = 0; i < 32 * 8; i++)
            {
                Assert.Equal(0.0f, aPtr[i]);
            }
        }
    }

    [Fact]
    public void PreallocatedMoeExpansionBank_ForwardDelta_ProducesZeroWhenDormantAndAccurateWhenActive()
    {
        const int dim = 16;
        const int rank = 4;
        using var bank = new PreallocatedMoeExpansionBank(dim, rank, maxExperts: 2);

        float[] input = new float[dim];
        float[] delta = new float[dim];
        float[] scratch = new float[rank];
        input[0] = 1.0f;

        // No active slots: deltaOutput must be strictly 0
        bank.ForwardDelta(input, delta, scratch);
        for (int d = 0; d < dim; d++) Assert.Equal(0.0f, delta[d]);

        // Unlock slot 0 with centroid [1, 0, ...]
        int slot = bank.UnlockNextExpertSlot(input);
        bank.ActivateConsolidatedSlot(slot);

        unsafe
        {
            // Set A_0 = identity-like, B_0 = identity-like
            float* aPtr = bank.GetMatrixAPtr(slot);
            float* bPtr = bank.GetMatrixBPtr(slot);

            // A[d, r] = 1 if d == r, B[r, d] = 1 if r == d
            for (int r = 0; r < rank; r++)
            {
                aPtr[r * rank + r] = 1.0f;
                bPtr[r * dim + r] = 1.0f;
            }
        }

        bank.ForwardDelta(input, delta, scratch);
        Assert.Equal(1.0f, delta[0]);
    }

    [Fact]
    public void DreamQualityFilter_AcceptsHighFidelityDreamsAndRejectsHallucinations()
    {
        string refText = "Glacier.Tune optimizes low-rank adapters via GPU-accelerated SIMD kernels.";
        string validDream = "Glacier.Tune optimizes low-rank adapters via GPU-accelerated SIMD kernels.";
        string corruptedDream = "Completely unrelated hallucinatory token sequence with random noise 12345.";

        float[] refEmb = [1.0f, 0.0f, 0.0f, 0.0f];
        float[] validEmb = [0.99f, 0.05f, 0.0f, 0.0f];
        float[] corruptEmb = [-0.80f, 0.20f, 0.10f, 0.0f];

        float validScore = DreamQualityFilter.ComputeScore(validDream, refText, validEmb, refEmb, alpha: 0.65f);
        float corruptScore = DreamQualityFilter.ComputeScore(corruptedDream, refText, corruptEmb, refEmb, alpha: 0.65f);

        Assert.True(validScore > 0.95f, $"Expected valid dream score > 0.95, got {validScore:F4}");
        Assert.True(corruptScore < 0.30f, $"Expected corrupt dream score < 0.30, got {corruptScore:F4}");
    }

    [Fact]
    public async Task SleepConsolidationTrainer_EndToEndLifecycle_ConsolidatesKnowledgeAndExecutesSynapticPruning()
    {
        const int dim = 32;
        const int rank = 8;
        using var bank = new PreallocatedMoeExpansionBank(dim, rank, maxExperts: 4);

        bool pruningExecuted = false;
        using var trainer = new SleepConsolidationTrainer(bank, synapticPruningCallback: () =>
        {
            pruningExecuted = true;
        });

        Random rng = new Random(42);
        float[][] keys = CreateOrthogonalBasis(rng, dim, 8);
        float[] centroid = (float[])keys[0].Clone();
        int slot = trainer.ParameterExpansion(centroid);

        // Generate 8 dream candidates (6 high-fidelity with distinct keys, 2 hallucinated)
        DreamCandidate[] dreamBatch = new DreamCandidate[8];
        for (int i = 0; i < 8; i++)
        {
            bool isHallucinated = (i >= 6);
            float[] q = keys[i];
            float[] target = CreateRandomUnitVector(rng, dim);
            string prompt = $"EpisodicContext_Item_{i}";
            string dream = isHallucinated ? "Garbage_Hallucination_Error_99999" : prompt;
            float composite = isHallucinated ? 0.15f : 0.92f;

            dreamBatch[i] = new DreamCandidate(
                prompt,
                dream,
                q,
                target,
                SemanticSimilarity: composite,
                LevenshteinRatio: composite,
                CompositeScore: composite);
        }

        ConsolidationResult result = await trainer.ConsolidateKnowledgeAsync(
            slot,
            dreamBatch,
            distillationSteps: 120,
            learningRate: 0.25f,
            qualityThreshold: 0.75f);

        Assert.Equal(slot, result.ExpertSlotIndex);
        Assert.Equal(8, result.TotalDreamsGenerated);
        Assert.Equal(6, result.TotalDreamsAccepted);
        Assert.True(result.AcceptanceRate == 0.75f);
        Assert.True(result.FinalGkdLoss < result.InitialGkdLoss * 0.10f, $"Expected GKD loss reduction by at least 90%, initial={result.InitialGkdLoss:F6}, final={result.FinalGkdLoss:F6}");
        Assert.True(result.SynapticPruningSucceeded);
        Assert.True(pruningExecuted);

        // Verify inference now routes through the consolidated MoE expert
        float[] delta = new float[dim];
        float[] scratch = new float[rank];
        bank.ForwardDelta(centroid, delta, scratch);

        float deltaNorm = 0.0f;
        for (int d = 0; d < dim; d++) deltaNorm += delta[d] * delta[d];
        Assert.True(deltaNorm > 0.01f, "Expected non-zero expert output contribution after consolidation.");
    }

    private static float[] CreateRandomUnitVector(Random rng, int dim)
    {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        NormalizeInPlace(v);
        return v;
    }

    private static void NormalizeInPlace(Span<float> v)
    {
        float mag = 0f;
        for (int i = 0; i < v.Length; i++) mag += v[i] * v[i];
        mag = MathF.Sqrt(mag);
        if (mag > 1e-12f)
        {
            for (int i = 0; i < v.Length; i++) v[i] /= mag;
        }
    }

    private static float[][] CreateOrthogonalBasis(Random rng, int dim, int count)
    {
        float[][] basis = new float[count][];
        for (int i = 0; i < count; i++)
        {
            float[] v = CreateRandomUnitVector(rng, dim);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int j = 0; j < i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dim; d++) dot += v[d] * basis[j][d];
                    for (int d = 0; d < dim; d++) v[d] -= dot * basis[j][d];
                }
                NormalizeInPlace(v);
            }
            basis[i] = v;
        }
        return basis;
    }
}

