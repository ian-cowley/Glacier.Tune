namespace Glacier.Tune.Expansion;

using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Status of a pre-allocated low-rank Mixture-of-Experts (MoE) slot in the expansion bank.
/// </summary>
public enum MoeSlotStatus : byte
{
    /// <summary>
    /// Slot is dormant in reserved unmanaged memory and does not participate in routing.
    /// </summary>
    Dormant = 0,

    /// <summary>
    /// Slot is currently undergoing sleep consolidation (GKD knowledge distillation).
    /// </summary>
    Consolidating = 1,

    /// <summary>
    /// Slot is active and routes live inference tokens.
    /// </summary>
    Active = 2,

    /// <summary>
    /// Slot is frozen against further gradient updates.
    /// </summary>
    Frozen = 3
}

/// <summary>
/// Fixed 64-byte cache-line aligned descriptor for a single low-rank MoE expert slot.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64)]
public unsafe struct MoeExpertSlotDescriptor
{
    [FieldOffset(0)] public int SlotIndex;
    [FieldOffset(4)] public MoeSlotStatus Status;
    [FieldOffset(5)] public byte ReservedByte;
    [FieldOffset(6)] public ushort Dimension;
    [FieldOffset(8)] public ushort Rank;
    [FieldOffset(10)] public ushort ReservedShort;
    [FieldOffset(12)] public float RouterTemperature;
    [FieldOffset(16)] public ulong MatrixAPointer;
    [FieldOffset(24)] public ulong MatrixBPointer;
    [FieldOffset(32)] public ulong CentroidPointer;
    [FieldOffset(40)] public ulong ReservedPtr1;
    [FieldOffset(48)] public ulong ReservedPtr2;
    [FieldOffset(56)] public ulong ReservedPtr3;
}

/// <summary>
/// Pre-allocated low-rank Mixture-of-Experts (MoE) parameter expansion bank for continual learning.
/// Enables unlocking dormant low-rank expert slots ($A_e \in \mathbb{R}^{d \times r}, B_e \in \mathbb{R}^{r \times d}$)
/// during offline sleep cycles without reallocating buffers or mutating frozen foundational base weights.
/// </summary>
public sealed unsafe class PreallocatedMoeExpansionBank : IDisposable
{
    private const nuint CacheLineAlignment = 64;

    private readonly int _dimension;
    private readonly int _rank;
    private readonly int _maxExperts;
    private readonly long _matrixABytes;
    private readonly long _matrixBBytes;
    private readonly long _centroidBytes;
    private readonly long _perExpertBytes;
    private readonly long _totalUnmanagedBytes;

    private byte* _buffer;
    private MoeExpertSlotDescriptor* _descriptors;
    private int _activeExpertCount;
    private float _topKGateThreshold;
    private bool _disposed;

    public int Dimension => _dimension;
    public int Rank => _rank;
    public int MaxExperts => _maxExperts;
    public int ActiveExpertCount => _activeExpertCount;
    public long TotalUnmanagedBytes => _totalUnmanagedBytes;

    public float TopKGateThreshold
    {
        get => _topKGateThreshold;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _topKGateThreshold = value;
        }
    }

    public PreallocatedMoeExpansionBank(
        int dimension,
        int rank,
        int maxExperts = 8,
        float topKGateThreshold = 0.05f)
    {
        if (dimension <= 0 || dimension > ushort.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Dimension must be between 1 and 65535.");
        }
        if (rank <= 0 || rank > ushort.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), rank, "Rank must be between 1 and 65535.");
        }
        if (maxExperts <= 0 || maxExperts > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(maxExperts), maxExperts, "MaxExperts must be between 1 and 256.");
        }
        ArgumentOutOfRangeException.ThrowIfNegative(topKGateThreshold);

        _dimension = dimension;
        _rank = rank;
        _maxExperts = maxExperts;
        _topKGateThreshold = topKGateThreshold;
        _activeExpertCount = 0;

        _matrixABytes = AlignUp((long)dimension * rank * sizeof(float), (long)CacheLineAlignment);
        _matrixBBytes = AlignUp((long)rank * dimension * sizeof(float), (long)CacheLineAlignment);
        _centroidBytes = AlignUp((long)dimension * sizeof(float), (long)CacheLineAlignment);
        _perExpertBytes = _matrixABytes + _matrixBBytes + _centroidBytes;

        long descriptorTableBytes = AlignUp((long)maxExperts * sizeof(MoeExpertSlotDescriptor), (long)CacheLineAlignment);
        _totalUnmanagedBytes = descriptorTableBytes + ((long)maxExperts * _perExpertBytes);

        _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)_totalUnmanagedBytes, CacheLineAlignment);
        NativeMemory.Clear(_buffer, (nuint)_totalUnmanagedBytes);

        _descriptors = (MoeExpertSlotDescriptor*)_buffer;
        byte* expertDataStart = _buffer + descriptorTableBytes;

        for (int e = 0; e < maxExperts; e++)
        {
            byte* expertBase = expertDataStart + (e * _perExpertBytes);
            float* aPtr = (float*)expertBase;
            float* bPtr = (float*)(expertBase + _matrixABytes);
            float* cPtr = (float*)(expertBase + _matrixABytes + _matrixBBytes);

            _descriptors[e].SlotIndex = e;
            _descriptors[e].Status = MoeSlotStatus.Dormant;
            _descriptors[e].Dimension = (ushort)dimension;
            _descriptors[e].Rank = (ushort)rank;
            _descriptors[e].RouterTemperature = 1.0f;
            _descriptors[e].MatrixAPointer = (ulong)aPtr;
            _descriptors[e].MatrixBPointer = (ulong)bPtr;
            _descriptors[e].CentroidPointer = (ulong)cPtr;
        }
    }

    /// <summary>
    /// Unlocks the next dormant low-rank MoE slot for the upcoming Sleep consolidation cycle.
    /// Matrix A is initialized to 0 (ensuring 0 initial deviation), Matrix B is initialized
    /// with normalized random Gaussian projections, and the routing centroid is assigned.
    /// </summary>
    public int UnlockNextExpertSlot(ReadOnlySpan<float> initialCentroid, Random? rng = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_activeExpertCount >= _maxExperts)
        {
            throw new InvalidOperationException($"Cannot unlock slot: All {_maxExperts} expert slots are active.");
        }
        if (initialCentroid.Length < _dimension)
        {
            throw new ArgumentException($"Initial centroid length ({initialCentroid.Length}) must be at least Dimension ({_dimension}).", nameof(initialCentroid));
        }

        int slotIndex = _activeExpertCount;
        ref MoeExpertSlotDescriptor desc = ref _descriptors[slotIndex];

        float* aPtr = (float*)desc.MatrixAPointer;
        float* bPtr = (float*)desc.MatrixBPointer;
        float* cPtr = (float*)desc.CentroidPointer;

        // Clear Matrix A to strictly 0 (zero initial perturbation)
        NativeMemory.Clear(aPtr, (nuint)_matrixABytes);

        // Initialize Matrix B with scaled Gaussian random projections: std = sqrt(2 / dim)
        rng ??= new Random(42 + slotIndex * 1337);
        float std = MathF.Sqrt(2.0f / _dimension);
        int bCount = _rank * _dimension;
        for (int i = 0; i < bCount; i++)
        {
            bPtr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * std);
        }

        // Copy and normalize centroid
        fixed (float* pInit = initialCentroid)
        {
            Buffer.MemoryCopy(pInit, cPtr, _dimension * sizeof(float), _dimension * sizeof(float));
            NormalizeVectorSimd(cPtr, _dimension);
        }

        desc.Status = MoeSlotStatus.Consolidating;
        _activeExpertCount++;
        return slotIndex;
    }

    /// <summary>
    /// Marks an in-consolidation expert slot as fully active for live inference routing.
    /// </summary>
    public void ActivateConsolidatedSlot(int slotIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateSlotIndex(slotIndex);
        _descriptors[slotIndex].Status = MoeSlotStatus.Active;
    }

    /// <summary>
    /// Gets a pointer to Matrix A of the specified expert slot (shape: [Dimension x Rank]).
    /// </summary>
    public float* GetMatrixAPtr(int slotIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateSlotIndex(slotIndex);
        return (float*)_descriptors[slotIndex].MatrixAPointer;
    }

    /// <summary>
    /// Gets a pointer to Matrix B of the specified expert slot (shape: [Rank x Dimension]).
    /// </summary>
    public float* GetMatrixBPtr(int slotIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateSlotIndex(slotIndex);
        return (float*)_descriptors[slotIndex].MatrixBPointer;
    }

    /// <summary>
    /// Gets a pointer to the routing centroid vector of the specified expert slot (length: Dimension).
    /// </summary>
    public float* GetCentroidPtr(int slotIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateSlotIndex(slotIndex);
        return (float*)_descriptors[slotIndex].CentroidPointer;
    }

    /// <summary>
    /// Evaluates the low-rank MoE forward contribution:
    /// <c>deltaOutput = sum_{e \in Active} g_e(input) * (A_e (B_e input))</c>
    /// with zero managed allocations and tiered SIMD dot products and GEMVs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ForwardDelta(
        ReadOnlySpan<float> input,
        Span<float> deltaOutput,
        Span<float> rankScratch)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (input.Length < _dimension) throw new ArgumentException("Input too small.", nameof(input));
        if (deltaOutput.Length < _dimension) throw new ArgumentException("deltaOutput too small.", nameof(deltaOutput));
        if (rankScratch.Length < _rank) throw new ArgumentException("rankScratch too small.", nameof(rankScratch));

        deltaOutput.Slice(0, _dimension).Clear();
        if (_activeExpertCount == 0) return;

        fixed (float* pIn = input)
        fixed (float* pDelta = deltaOutput)
        fixed (float* pRank = rankScratch)
        {
            // Find Top-1 / Softmax expert with highest affinity above threshold
            int bestExpert = -1;
            float bestAffinity = _topKGateThreshold;

            for (int e = 0; e < _activeExpertCount; e++)
            {
                if (_descriptors[e].Status != MoeSlotStatus.Active) continue;

                float* cPtr = (float*)_descriptors[e].CentroidPointer;
                float dot = DotProductSimd(pIn, cPtr, _dimension);
                float affinity = MathF.Abs(dot); // cosine affinity with normalized keys/centroids

                if (affinity > bestAffinity)
                {
                    bestAffinity = affinity;
                    bestExpert = e;
                }
            }

            if (bestExpert >= 0)
            {
                float* aPtr = (float*)_descriptors[bestExpert].MatrixAPointer;
                float* bPtr = (float*)_descriptors[bestExpert].MatrixBPointer;

                // 1. rankScratch = B_e * input  ([Rank x Dimension] * [Dimension x 1])
                for (int r = 0; r < _rank; r++)
                {
                    float* bRow = bPtr + (r * _dimension);
                    pRank[r] = DotProductSimd(bRow, pIn, _dimension);
                }

                // 2. deltaOutput = A_e * rankScratch ([Dimension x Rank] * [Rank x 1])
                for (int d = 0; d < _dimension; d++)
                {
                    float* aRow = aPtr + (d * _rank);
                    float acc = 0.0f;
                    for (int r = 0; r < _rank; r++)
                    {
                        acc += aRow[r] * pRank[r];
                    }
                    pDelta[d] = acc;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void NormalizeVectorSimd(float* vec, int dim)
    {
        float dot = DotProductSimd(vec, vec, dim);
        float mag = MathF.Sqrt(dot);
        if (mag > 1e-12f)
        {
            float invMag = 1.0f / mag;
            for (int i = 0; i < dim; i++)
            {
                vec[i] *= invMag;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(float* a, float* b, int dim)
    {
        int j = 0;
        float sum = 0.0f;

        if (Avx512F.IsSupported && dim >= 16)
        {
            Vector512<float> acc512 = Vector512<float>.Zero;
            for (; j <= dim - 16; j += 16)
            {
                Vector512<float> va = Avx512F.LoadVector512(a + j);
                Vector512<float> vb = Avx512F.LoadVector512(b + j);
                acc512 = Avx512F.FusedMultiplyAdd(va, vb, acc512);
            }
            sum += Vector512.Sum(acc512);
        }

        if (Avx2.IsSupported && j <= dim - 8)
        {
            Vector256<float> acc256 = Vector256<float>.Zero;
            for (; j <= dim - 8; j += 8)
            {
                Vector256<float> va = Avx.LoadVector256(a + j);
                Vector256<float> vb = Avx.LoadVector256(b + j);
                acc256 = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc256) : Avx.Add(acc256, Avx.Multiply(va, vb));
            }
            sum += Vector256.Sum(acc256);
        }

        for (; j < dim; j++)
        {
            sum += a[j] * b[j];
        }

        return sum;
    }

    private void ValidateSlotIndex(int slotIndex)
    {
        if ((uint)slotIndex >= (uint)_maxExperts)
        {
            throw new ArgumentOutOfRangeException(nameof(slotIndex), slotIndex, $"Slot index must be between 0 and {_maxExperts - 1}.");
        }
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) & ~(alignment - 1);

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_buffer != null)
            {
                NativeMemory.AlignedFree(_buffer);
                _buffer = null;
                _descriptors = null;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~PreallocatedMoeExpansionBank()
    {
        Dispose();
    }
}
