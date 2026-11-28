namespace Glacier.Tune.Export;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Tune.Diagnostics;
using Glacier.Tune.Kernels;
using Glacier.Tune.Model;

/// <summary>
/// Target quantization representation for adapted LoRA projections.
/// </summary>
public enum TargetQuantization
{
    /// <summary>
    /// Auto-select quantization preserving base tensor type (Q6_K, F16, BF16, F32, Q4_0, Q8_0; others default to Q4_K).
    /// </summary>
    Auto,

    /// <summary>
    /// Super-block 4-bit k-quantization (256 weights in 144 bytes, 4.5 bpw).
    /// </summary>
    Q4_K,

    /// <summary>
    /// Super-block 6-bit k-quantization (256 weights in 210 bytes, 6.5625 bpw).
    /// </summary>
    Q6_K,

    /// <summary>
    /// Standard 4-bit quantization (32 weights in 18 bytes, 4.5 bpw).
    /// </summary>
    Q4_0,

    /// <summary>
    /// Standard 8-bit quantization (32 weights in 34 bytes, 8.5 bpw).
    /// </summary>
    Q8_0,

    /// <summary>
    /// IEEE 754 binary16 half-precision representation (16 bpw).
    /// </summary>
    F16,

    /// <summary>
    /// Brain floating-point 16-bit representation (16 bpw).
    /// </summary>
    BF16,

    /// <summary>
    /// Uncompressed IEEE 754 32-bit floating-point representation (32 bpw).
    /// </summary>
    F32
}

/// <summary>
/// Result metadata of GGUF LoRA model merge.
/// </summary>
public sealed class MergeResult
{
    public string OutputPath { get; init; } = string.Empty;
    public long OutputSizeBytes { get; init; }
    public int AdaptedTensorCount { get; init; }
    public int PreservedTensorCount { get; init; }
    public TimeSpan Elapsed { get; init; }
}

/// <summary>
/// High-performance standalone GGUF model merger.
/// Fuses trained LoRA low-rank delta adapters directly into base model GGUF weights,
/// converting adapted layers into lossless F32/Q8_0 or compact Q4_K/Q4_0 quantized representations
/// while preserving non-adapted layers, tokenizer vocabulary, and all GGUF metadata intact.
/// 100% pure C# .NET 10 with AVX-512/AVX2 multithreading.
/// </summary>
public static unsafe class GgufMerger
{
    public sealed class MergeOptions
    {
        public Action<string>? ProgressCallback { get; init; }
        public int MaxDegreeOfParallelism { get; init; } = Environment.ProcessorCount;
        public TargetQuantization TargetQuantization { get; init; } = TargetQuantization.Auto;
    }

    /// <summary>
    /// Merges binary LoRA adapter weights directly into a base GGUF model and writes
    /// a standalone, production-ready GGUF model file.
    /// </summary>
    public static MergeResult Merge(string baseGgufPath, string adapterBinPath, string outputGgufPath, MergeOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseGgufPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(adapterBinPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputGgufPath);

        var progress = options?.ProgressCallback ?? (msg => GlacierDiagnostics.LogInformation(msg));
        var targetQuant = options?.TargetQuantization ?? TargetQuantization.Auto;

        if (!File.Exists(baseGgufPath))
            throw new FileNotFoundException($"Base GGUF model not found: {baseGgufPath}");

        if (!File.Exists(adapterBinPath))
            throw new FileNotFoundException($"LoRA adapter file not found: {adapterBinPath}");

        string fullBasePath = Path.GetFullPath(baseGgufPath);
        string fullOutPath = Path.GetFullPath(outputGgufPath);
        if (string.Equals(fullBasePath, fullOutPath, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Target output GGUF path '{outputGgufPath}' resolves to the base GGUF model path. In-place overwrite of memory-mapped base weights is prohibited.");
        }

        var sw = Stopwatch.StartNew();
        progress($"[1/4] Loading unmanaged base GGUF model and LoRA adapter...");
        using var baseGguf = new GgufFile(baseGgufPath);
        var adapter = ParsedLoraAdapter.Load(adapterBinPath);

        if (adapter.Rank <= 0)
        {
            throw new InvalidDataException($"LoRA adapter rank must be positive, got {adapter.Rank}.");
        }

        progress($"  Base Model:    {baseGguf.Architecture} ({baseGguf.BlockCount} blocks, {baseGguf.TensorCount} tensors)");
        progress($"  LoRA Adapter:  {adapter.LayerCount} layers, rank={adapter.Rank}, alpha={adapter.Alpha}, scaling={adapter.Scaling:F2}");
        progress($"  Base Size:     {new FileInfo(baseGgufPath).Length:N0} bytes");

        // 1. Calculate Tensor Directory, Validate Adapter Dimensions, and Compute Aligned Data Layout
        progress($"[2/4] Computing merged tensor directory and aligned offsets...");
        int tensorCount = (int)baseGguf.TensorCount;
        var newTypes = new GgufType[tensorCount];
        var newOffsets = new ulong[tensorCount];
        var newByteSizes = new ulong[tensorCount];
        var mergePlans = new List<RowSliceMerge>?[tensorCount];

        int adaptedCount = 0;
        int nonAdaptedCount = 0;

        long tensorDirByteCount = 0;
        for (int i = 0; i < tensorCount; i++)
        {
            var t = baseGguf.TensorList[i];
            int nameBytesLen = Encoding.UTF8.GetByteCount(t.Name);
            tensorDirByteCount += 8 + nameBytesLen; // string: uint64 len + utf8 bytes
            tensorDirByteCount += 4;                // n_dims: uint32
            tensorDirByteCount += (long)t.DimensionsCount * 8; // dims: nDims * uint64
            tensorDirByteCount += 4;                // type: uint32
            tensorDirByteCount += 8;                // offset: uint64
        }

        ulong totalHeaderBytes = baseGguf.MetadataEndOffset + (ulong)tensorDirByteCount;
        ulong remHeader = totalHeaderBytes % baseGguf.Alignment;
        ulong tensorDataOffset = remHeader == 0 ? totalHeaderBytes : totalHeaderBytes + (baseGguf.Alignment - remHeader);

        ulong currentPayloadOffset = 0;
        for (int i = 0; i < tensorCount; i++)
        {
            var t = baseGguf.TensorList[i];
            var slices = ResolveMergeSlices(baseGguf, t, adapter);
            bool isAdapted = slices != null && slices.Count > 0;
            mergePlans[i] = slices;

            GgufType newType;
            ulong newSize;
            if (isAdapted)
            {
                adaptedCount++;
                int inDim = (int)t.Dimensions[0];
                int outDim = (int)t.Dimensions[1];
                newType = ResolveTargetType(targetQuant, t.Type, inDim);
                int rowBytes = GetRowByteSize(newType, inDim);
                newSize = (ulong)outDim * (ulong)rowBytes;
            }
            else
            {
                nonAdaptedCount++;
                newType = t.Type;
                newSize = t.GetByteSize();
            }

            ulong remAlign = currentPayloadOffset % baseGguf.Alignment;
            if (remAlign != 0)
            {
                currentPayloadOffset += (baseGguf.Alignment - remAlign);
            }

            newOffsets[i] = currentPayloadOffset;
            newTypes[i] = newType;
            newByteSizes[i] = newSize;

            currentPayloadOffset += newSize;
        }

        string? outDir = Path.GetDirectoryName(outputGgufPath);
        if (!string.IsNullOrEmpty(outDir) && !Directory.Exists(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        progress($"  Adapted Projections: {adaptedCount} (fused into target {targetQuant})");
        progress($"  Non-Adapted Tensors: {nonAdaptedCount} (preserved bit-exact)");
        progress($"  Header End Offset:   {totalHeaderBytes:N0} bytes");
        progress($"  Tensor Data Offset:  {tensorDataOffset:N0} bytes (aligned to {baseGguf.Alignment} bytes)");

        // 2. Open Atomic Temp Output Stream and Write Header + Metadata + Directory
        progress($"[3/4] Writing merged GGUF header, metadata, and tensor directory...");
        string tmpOutPath = LoraBinaryFormat.CreateTempSiblingPath(outputGgufPath);
        try
        {
            using (var outFs = new FileStream(tmpOutPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1024 * 1024))
            using (var bw = new BinaryWriter(outFs, Encoding.UTF8, leaveOpen: true))
            {
                // Copy raw Header + Metadata byte-for-byte from base GGUF (preserves tokenizer & KV metadata intact)
                byte* pBase = baseGguf.BasePointer;
                ulong metadataBytes = baseGguf.MetadataEndOffset;
                outFs.Write(new ReadOnlySpan<byte>(pBase, checked((int)metadataBytes)));

                // Write Tensor Directory entries
                for (int i = 0; i < tensorCount; i++)
                {
                    var t = baseGguf.TensorList[i];
                    byte[] nameBytes = Encoding.UTF8.GetBytes(t.Name);
                    bw.Write((ulong)nameBytes.Length);
                    bw.Write(nameBytes);
                    bw.Write(t.DimensionsCount);
                    for (int d = 0; d < (int)t.DimensionsCount; d++)
                    {
                        bw.Write(t.Dimensions[d]);
                    }
                    bw.Write((uint)newTypes[i]);
                    bw.Write(newOffsets[i]);
                }
                bw.Flush();

                long currentPos = outFs.Position;
                int headerPad = (int)(tensorDataOffset - (ulong)currentPos);
                if (headerPad > 0)
                {
                    WriteZeroPadding(outFs, headerPad);
                }

                // 3. Process and Stream Tensor Data via Bounded <= 16 MB Row Strips
                progress($"[4/4] Streaming and fusing weights into {outputGgufPath}...");
                int lastReportedLayer = -1;
                var parallelOpts = new ParallelOptions { MaxDegreeOfParallelism = options?.MaxDegreeOfParallelism ?? Environment.ProcessorCount };
                const int maxStripBytes = 16 * 1024 * 1024;

                for (int i = 0; i < tensorCount; i++)
                {
                    var t = baseGguf.TensorList[i];
                    long targetPos = (long)(tensorDataOffset + newOffsets[i]);
                    long padNeeded = targetPos - outFs.Position;
                    if (padNeeded > 0)
                    {
                        WriteZeroPadding(outFs, (int)padNeeded);
                    }

                    var slices = mergePlans[i];
                    if (slices == null || slices.Count == 0)
                    {
                        byte* srcPtr = baseGguf.GetTensorPointer(t);
                        ulong bytesRemaining = newByteSizes[i];
                        while (bytesRemaining > 0)
                        {
                            int toCopy = (int)Math.Min((ulong)maxStripBytes, bytesRemaining);
                            outFs.Write(new ReadOnlySpan<byte>(srcPtr, toCopy));
                            srcPtr += toCopy;
                            bytesRemaining -= (ulong)toCopy;
                        }
                    }
                    else
                    {
                        if (TryParseLayerName(t.Name, out int layer, out _) && layer != lastReportedLayer)
                        {
                            lastReportedLayer = layer;
                            progress($"  Fusing Transformer Block {layer:D2}/{baseGguf.BlockCount - 1}...");
                        }

                        int inDim = (int)t.Dimensions[0];
                        int outDim = (int)t.Dimensions[1];
                        GgufType targetType = newTypes[i];
                        int rowBytes = GetRowByteSize(targetType, inDim);
                        int stripRows = Math.Max(1, Math.Min(outDim, maxStripBytes / Math.Max(1, rowBytes)));
                        nuint stripAllocBytes = checked((nuint)stripRows * (nuint)rowBytes);

                        byte* pStrip = (byte*)NativeMemory.Alloc(stripAllocBytes);
                        try
                        {
                            int rank = adapter.Rank;
                            float scaling = adapter.Scaling;
                            IntPtr ptrBaseData = (IntPtr)baseGguf.GetTensorPointer(t);
                            IntPtr ptrStrip = (IntPtr)pStrip;
                            GgufType baseType = t.Type;

                            var pinnedSlices = new PinnedRowSlice[slices.Count];
                            var transposedABufs = new List<float[]>();
                            for (int s = 0; s < slices.Count; s++)
                            {
                                var sl = slices[s];
                                float[] aArr = sl.Slot.A!;
                                bool isAPeft = sl.IsAPeftRowMajor;
                                if (isAPeft)
                                {
                                    // Convert [rank, inDim] -> [inDim, rank] once per adapted tensor so both layouts run identical SIMD DotSimd (bit-exact output).
                                    float[] nativeA = ArrayPool<float>.Shared.Rent(checked(inDim * rank));
                                    transposedABufs.Add(nativeA);
                                    for (int c = 0; c < inDim; c++)
                                    {
                                        int dstOff = c * rank;
                                        for (int k = 0; k < rank; k++)
                                        {
                                            nativeA[dstOff + k] = aArr[k * inDim + c];
                                        }
                                    }
                                    aArr = nativeA;
                                    isAPeft = false;
                                }

                                var hA = GCHandle.Alloc(aArr, GCHandleType.Pinned);
                                var hB = GCHandle.Alloc(sl.Slot.B!, GCHandleType.Pinned);
                                pinnedSlices[s] = new PinnedRowSlice(
                                    sl.RowStart,
                                    sl.RowCount,
                                    hA,
                                    hB,
                                    hA.AddrOfPinnedObject(),
                                    hB.AddrOfPinnedObject(),
                                    isAPeft,
                                    sl.IsBPeftRowMajor);
                            }

                            try
                            {
                                for (int rowBase = 0; rowBase < outDim; rowBase += stripRows)
                                {
                                    int curStripRows = Math.Min(stripRows, outDim - rowBase);
                                    int rowBaseLocal = rowBase;

                                    Parallel.For(0, curStripRows, parallelOpts, () => ArrayPool<float>.Shared.Rent(inDim + rank), (localRowIdx, loopState, workArr) =>
                                    {
                                        int r = rowBaseLocal + localRowIdx;
                                        fixed (float* pWork = workArr)
                                        {
                                            float* rowFloats = pWork;
                                            float* b_r = pWork + inDim;
                                            byte* curBase = (byte*)ptrBaseData;
                                            byte* curDst = (byte*)ptrStrip;

                                            // 1. Dequantize base weight row r -> rowFloats[0 .. inDim - 1]
                                            QuantKernels.ExtractEmbedding(baseType, curBase, r, rowFloats, inDim);

                                            // 2 & 3. Find covering LoRA slice (if any) and accumulate LoRA delta
                                            for (int s = 0; s < pinnedSlices.Length; s++)
                                            {
                                                var slice = pinnedSlices[s];
                                                if (r >= slice.RowStart && r < slice.RowStart + slice.RowCount)
                                                {
                                                    int localRow = r - slice.RowStart;
                                                    int sliceOutDim = slice.RowCount;
                                                    float* curA = (float*)slice.PtrA;
                                                    float* curB = (float*)slice.PtrB;

                                                    if (slice.IsBPeftRowMajor)
                                                    {
                                                        float* bRow = curB + (long)localRow * rank;
                                                        for (int k = 0; k < rank; k++)
                                                        {
                                                            b_r[k] = scaling * bRow[k];
                                                        }
                                                    }
                                                    else
                                                    {
                                                        for (int k = 0; k < rank; k++)
                                                        {
                                                            b_r[k] = scaling * curB[(long)k * sliceOutDim + localRow];
                                                        }
                                                    }

                                                    for (int c = 0; c < inDim; c++)
                                                    {
                                                        rowFloats[c] += LoraKernels.DotSimd(b_r, curA + (long)c * rank, rank);
                                                    }
                                                    break;
                                                }
                                            }

                                            // 4. Quantize or copy fused row to target representation
                                            byte* dstRow = curDst + (long)localRowIdx * rowBytes;
                                            switch (targetType)
                                            {
                                                case GgufType.Q4_K:
                                                    QuantizeRowQ4_K(rowFloats, (BlockQ4_K*)dstRow, inDim);
                                                    break;
                                                case GgufType.Q6_K:
                                                    QuantizeRowQ6_K(rowFloats, (BlockQ6_K*)dstRow, inDim);
                                                    break;
                                                case GgufType.Q4_0:
                                                    QuantizeRowQ4_0(rowFloats, (BlockQ4_0*)dstRow, inDim);
                                                    break;
                                                case GgufType.Q8_0:
                                                    QuantizeRowQ8_0(rowFloats, (BlockQ8_0*)dstRow, inDim);
                                                    break;
                                                case GgufType.F16:
                                                    QuantizeRowF16(rowFloats, (Half*)dstRow, inDim);
                                                    break;
                                                case GgufType.BF16:
                                                    QuantizeRowBF16(rowFloats, (ushort*)dstRow, inDim);
                                                    break;
                                                case GgufType.F32:
                                                    Buffer.MemoryCopy(rowFloats, dstRow, (ulong)rowBytes, (ulong)rowBytes);
                                                    break;
                                                default:
                                                    throw new NotSupportedException($"Unsupported target quantization: {targetType}");
                                            }
                                        }
                                        return workArr;
                                    },
                                    workArr => ArrayPool<float>.Shared.Return(workArr));

                                    int bytesToWrite = checked(curStripRows * rowBytes);
                                    outFs.Write(new ReadOnlySpan<byte>(pStrip, bytesToWrite));
                                }
                            }
                            finally
                            {
                                for (int s = 0; s < pinnedSlices.Length; s++)
                                {
                                    pinnedSlices[s].HandleA.Free();
                                    pinnedSlices[s].HandleB.Free();
                                }
                                for (int iBuf = 0; iBuf < transposedABufs.Count; iBuf++)
                                {
                                    ArrayPool<float>.Shared.Return(transposedABufs[iBuf]);
                                }
                            }
                        }
                        finally
                        {
                            NativeMemory.Free(pStrip);
                        }
                    }
                }

                outFs.Flush(true);
            }

            File.Move(tmpOutPath, outputGgufPath, overwrite: true);
        }
        catch
        {
            try { if (File.Exists(tmpOutPath)) File.Delete(tmpOutPath); }
            catch (Exception cleanupEx) when (cleanupEx is IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"[GgufMerger] Failed to clean temp file '{tmpOutPath}': {cleanupEx.Message}");
            }
            throw;
        }
        sw.Stop();

        long finalLength = new FileInfo(outputGgufPath).Length;
        progress($"\n[SUCCESS] Standalone GGUF Model Exported Successfully!");
        progress($"  Output Path: {outputGgufPath}");
        progress($"  Final Size:  {finalLength:N0} bytes ({finalLength / (1024.0 * 1024.0 * 1024.0):F2} GB)");
        progress($"  Time Taken:  {sw.Elapsed.TotalMinutes:F2} min ({sw.Elapsed.TotalSeconds:F1} s)");

        return new MergeResult
        {
            OutputPath = outputGgufPath,
            OutputSizeBytes = finalLength,
            AdaptedTensorCount = adaptedCount,
            PreservedTensorCount = nonAdaptedCount,
            Elapsed = sw.Elapsed
        };
    }

    private static void ValidateAndResolveSlotLayout(
        string tensorName,
        float[]? A,
        float[]? B,
        int slotInDim,
        int slotOutDim,
        int inDim,
        int outDim,
        int rank,
        out bool isAPeftRowMajor,
        out bool isBPeftRowMajor)
    {
        if (A == null || B == null)
        {
            throw new InvalidDataException($"Incomplete LoRA adapter pair for tensor '{tensorName}': both lora_a and lora_b must be present.");
        }

        long expectedALen = (long)inDim * rank;
        long expectedBLen = (long)rank * outDim;
        if (A.LongLength != expectedALen)
        {
            throw new InvalidDataException(
                $"LoRA adapter A dimension mismatch for '{tensorName}': expected {inDim}x{rank} ({expectedALen} elements), got {A.LongLength} elements (dim0={slotInDim}).");
        }
        if (B.LongLength != expectedBLen)
        {
            throw new InvalidDataException(
                $"LoRA adapter B dimension mismatch for '{tensorName}': expected {rank}x{outDim} ({expectedBLen} elements), got {B.LongLength} elements (dim1={slotOutDim}).");
        }

        if (slotInDim == inDim)
        {
            isAPeftRowMajor = false;
        }
        else if (slotInDim == rank)
        {
            isAPeftRowMajor = true;
        }
        else
        {
            throw new InvalidDataException(
                $"LoRA adapter A shape mismatch for '{tensorName}': dim0={slotInDim} matches neither inDim={inDim} nor rank={rank}.");
        }

        if (slotOutDim == outDim)
        {
            isBPeftRowMajor = false;
        }
        else if (slotOutDim == rank)
        {
            isBPeftRowMajor = true;
        }
        else
        {
            throw new InvalidDataException(
                $"LoRA adapter B shape mismatch for '{tensorName}': dim1={slotOutDim} matches neither outDim={outDim} nor rank={rank}.");
        }
    }

    private static void WriteZeroPadding(FileStream fs, int padBytes)
    {
        Span<byte> zeroChunk = padBytes <= 256 ? stackalloc byte[padBytes] : stackalloc byte[256];
        zeroChunk.Clear();
        int remaining = padBytes;
        while (remaining > 0)
        {
            int n = Math.Min(remaining, zeroChunk.Length);
            fs.Write(zeroChunk.Slice(0, n));
            remaining -= n;
        }
    }

    private static float BlockMaxAbs32(float* blockSrc)
    {
        if (Vector512.IsHardwareAccelerated)
        {
            var v0 = Vector512.Abs(Vector512.Load(blockSrc));
            var v1 = Vector512.Abs(Vector512.Load(blockSrc + 16));
            var vmax = Vector512.Max(v0, v1);
            float m = 0f;
            for (int k = 0; k < 16; k++)
            {
                float e = vmax[k];
                if (e > m) m = e;
            }
            return m;
        }
        if (Vector256.IsHardwareAccelerated)
        {
            var v0 = Vector256.Abs(Vector256.Load(blockSrc));
            var v1 = Vector256.Abs(Vector256.Load(blockSrc + 8));
            var v2 = Vector256.Abs(Vector256.Load(blockSrc + 16));
            var v3 = Vector256.Abs(Vector256.Load(blockSrc + 24));
            var vmax = Vector256.Max(Vector256.Max(v0, v1), Vector256.Max(v2, v3));
            float m = 0f;
            for (int k = 0; k < 8; k++)
            {
                float e = vmax[k];
                if (e > m) m = e;
            }
            return m;
        }
        float maxAbs = 0f;
        for (int i = 0; i < 32; i++)
        {
            float val = MathF.Abs(blockSrc[i]);
            if (val > maxAbs) maxAbs = val;
        }
        return maxAbs;
    }

    private static void BlockMinMax32(float* blockSrc, out float minVal, out float maxVal)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var x0 = Vector256.Load(blockSrc);
            var x1 = Vector256.Load(blockSrc + 8);
            var x2 = Vector256.Load(blockSrc + 16);
            var x3 = Vector256.Load(blockSrc + 24);
            var vmin = Vector256.Min(Vector256.Min(x0, x1), Vector256.Min(x2, x3));
            var vmax = Vector256.Max(Vector256.Max(x0, x1), Vector256.Max(x2, x3));
            minVal = vmin[0];
            maxVal = vmax[0];
            for (int k = 1; k < 8; k++)
            {
                if (vmin[k] < minVal) minVal = vmin[k];
                if (vmax[k] > maxVal) maxVal = vmax[k];
            }
            return;
        }
        minVal = blockSrc[0];
        maxVal = blockSrc[0];
        for (int i = 1; i < 32; i++)
        {
            float v = blockSrc[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void QuantizeRowQ8_0(float* src, BlockQ8_0* dst, int n)
    {
        int numBlocks = n / 32;
        for (int b = 0; b < numBlocks; b++)
        {
            float* blockSrc = src + b * 32;
            BlockQ8_0* blockDst = dst + b;

            float maxAbs = BlockMaxAbs32(blockSrc);

            if (maxAbs == 0f)
            {
                blockDst->Delta = (Half)0f;
                new Span<sbyte>(blockDst->Qs, 32).Clear();
                continue;
            }

            float scale = maxAbs / 127.0f;
            float invScale = 127.0f / maxAbs;
            blockDst->Delta = (Half)scale;

            for (int i = 0; i < 32; i++)
            {
                int q = (int)MathF.Round(blockSrc[i] * invScale);
                blockDst->Qs[i] = (sbyte)Math.Clamp(q, -128, 127);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void QuantizeRowF16(float* src, Half* dst, int n)
    {
        for (int i = 0; i < n; i++)
        {
            dst[i] = (Half)src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeRowBF16(float* src, ushort* dst, int n)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var vOne = Vector256.Create(1u);
            var vRoundBase = Vector256.Create(0x7FFFu);
            var vAbsMask = Vector256.Create(0x7FFFFFFFu);
            var vInf = Vector256.Create(0x7F800000u);
            var vQNaN = Vector256.Create(0x0040u);

            for (; i <= n - 8; i += 8)
            {
                var bits = Vector256.Load((uint*)(src + i));
                var absBits = bits & vAbsMask;
                var isNan = Vector256.GreaterThan(absBits, vInf);
                var lsb = (bits >> 16) & vOne;
                var rounded = (bits + vRoundBase + lsb) >> 16;
                var nanVal = (bits >> 16) | vQNaN;
                var bf32 = Vector256.ConditionalSelect(isNan, nanVal, rounded);
                for (int k = 0; k < 8; k++)
                {
                    dst[i + k] = (ushort)bf32[k];
                }
            }
        }
        for (; i < n; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(src[i]);
            if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            {
                dst[i] = (ushort)((bits >> 16) | 0x0040u);
            }
            else
            {
                uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
                dst[i] = (ushort)((bits + roundingBias) >> 16);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeRowQ6_K(float* src, BlockQ6_K* dst, int n)
    {
        int numSuperBlocks = n / 256;
        float* subScale = stackalloc float[16];
        sbyte* sc = stackalloc sbyte[16];
        byte* qBias = stackalloc byte[256];

        for (int sb = 0; sb < numSuperBlocks; sb++)
        {
            float* sbSrc = src + sb * 256;
            BlockQ6_K* sbDst = dst + sb;

            float maxScale = 0f;
            for (int ib = 0; ib < 16; ib++)
            {
                float* bSrc = sbSrc + ib * 16;
                float bMaxAbs = 0f;
                for (int i = 0; i < 16; i++)
                {
                    float v = MathF.Abs(bSrc[i]);
                    if (v > bMaxAbs) bMaxAbs = v;
                }
                float scale = bMaxAbs / 31.0f;
                subScale[ib] = scale;
                if (scale > maxScale) maxScale = scale;
            }

            if (maxScale == 0f)
            {
                sbDst->Delta = (Half)0f;
                new Span<sbyte>(sbDst->Scales, 16).Clear();
                new Span<byte>(sbDst->Ql, 128).Clear();
                new Span<byte>(sbDst->Qh, 64).Clear();
                continue;
            }

            float superD = maxScale / 127.0f;
            float invSuperScale = 127.0f / maxScale;
            sbDst->Delta = (Half)superD;
            float actualD = (float)sbDst->Delta;

            for (int ib = 0; ib < 16; ib++)
            {
                int scQ = (int)MathF.Round(subScale[ib] * invSuperScale);
                sc[ib] = (sbyte)Math.Clamp(scQ, 0, 127);
                sbDst->Scales[ib] = sc[ib];

                float effScale = actualD * sc[ib];
                float invEff = effScale > 0f ? 1.0f / effScale : 0f;
                float* bSrc = sbSrc + ib * 16;
                byte* bQ = qBias + ib * 16;

                for (int i = 0; i < 16; i++)
                {
                    int q = invEff > 0f ? (int)MathF.Round(bSrc[i] * invEff) : 0;
                    q = Math.Clamp(q, -32, 31);
                    bQ[i] = (byte)(q + 32);
                }
            }

            for (int half = 0; half < 2; half++)
            {
                byte* qHalf = qBias + half * 128;
                byte* ql = sbDst->Ql + half * 64;
                byte* qh = sbDst->Qh + half * 32;

                for (int l = 0; l < 32; l++)
                {
                    int q1 = qHalf[l + 0];
                    int q2 = qHalf[l + 32];
                    int q3 = qHalf[l + 64];
                    int q4 = qHalf[l + 96];

                    ql[l + 0] = (byte)((q1 & 0x0F) | ((q3 & 0x0F) << 4));
                    ql[l + 32] = (byte)((q2 & 0x0F) | ((q4 & 0x0F) << 4));
                    qh[l] = (byte)(((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DequantizeRowQ6_K(BlockQ6_K* src, float* dst, int n)
    {
        QuantKernels.DequantizeQ6_K(src, dst, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GgufType ResolveTargetType(TargetQuantization target, GgufType baseType, int inDim)
    {
        GgufType resolved = target switch
        {
            TargetQuantization.Q4_K => GgufType.Q4_K,
            TargetQuantization.Q6_K => GgufType.Q6_K,
            TargetQuantization.Q4_0 => GgufType.Q4_0,
            TargetQuantization.Q8_0 => GgufType.Q8_0,
            TargetQuantization.F16 => GgufType.F16,
            TargetQuantization.BF16 => GgufType.BF16,
            TargetQuantization.F32 => GgufType.F32,
            TargetQuantization.Auto => baseType switch
            {
                GgufType.Q6_K => GgufType.Q6_K,
                GgufType.Q4_0 => GgufType.Q4_0,
                GgufType.Q8_0 => GgufType.Q8_0,
                GgufType.F16 => GgufType.F16,
                GgufType.BF16 => GgufType.BF16,
                GgufType.F32 => GgufType.F32,
                _ => GgufType.Q4_K
            },
            _ => GgufType.Q4_K
        };

        if (resolved == GgufType.Q6_K && inDim % 256 != 0)
        {
            resolved = inDim % 32 == 0 ? GgufType.Q8_0 : GgufType.F16;
        }

        if (resolved == GgufType.Q4_K && inDim % 256 != 0)
        {
            resolved = GgufType.Q4_0;
        }

        if ((resolved == GgufType.Q4_0 || resolved == GgufType.Q8_0) && inDim % 32 != 0)
        {
            resolved = GgufType.F32;
        }

        return resolved;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetRowByteSize(GgufType type, int inDim)
    {
        return type switch
        {
            GgufType.Q4_K => (inDim / 256) * sizeof(BlockQ4_K),
            GgufType.Q6_K => (inDim / 256) * sizeof(BlockQ6_K),
            GgufType.Q4_0 => (inDim / 32) * sizeof(BlockQ4_0),
            GgufType.Q8_0 => (inDim / 32) * sizeof(BlockQ8_0),
            GgufType.F16 => inDim * sizeof(ushort),
            GgufType.BF16 => inDim * sizeof(ushort),
            GgufType.F32 => inDim * sizeof(float),
            _ => throw new NotSupportedException($"Target quantization type {type} is not supported for LoRA fusion.")
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void QuantizeRowQ4_0(float* src, BlockQ4_0* dst, int n)
    {
        int numBlocks = n / 32;
        for (int b = 0; b < numBlocks; b++)
        {
            float* blockSrc = src + b * 32;
            BlockQ4_0* blockDst = dst + b;

            float amax = 0.0f;
            float max = 0.0f;
            for (int i = 0; i < 32; i++)
            {
                float v = blockSrc[i];
                float abs = MathF.Abs(v);
                if (abs > amax)
                {
                    amax = abs;
                    max = v;
                }
            }

            if (amax == 0.0f)
            {
                blockDst->Delta = (Half)0f;
                for (int l = 0; l < 16; l++)
                {
                    blockDst->Qs[l] = 0x88;
                }
                continue;
            }

            float d = max / -8.0f;
            float id = d != 0f ? 1.0f / d : 0.0f;
            blockDst->Delta = (Half)d;

            for (int l = 0; l < 16; l++)
            {
                float x0 = blockSrc[l] * id;
                float x1 = blockSrc[l + 16] * id;
                int q0 = Math.Clamp((int)MathF.Round(x0) + 8, 0, 15);
                int q1 = Math.Clamp((int)MathF.Round(x1) + 8, 0, 15);
                blockDst->Qs[l] = (byte)(q0 | (q1 << 4));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DequantizeRowQ4_0(BlockQ4_0* src, float* dst, int n)
    {
        int numBlocks = n / 32;
        for (int b = 0; b < numBlocks; b++)
        {
            float d = (float)src[b].Delta;
            for (int l = 0; l < 16; l++)
            {
                byte v = src[b].Qs[l];
                dst[b * 32 + l] = ((v & 0x0F) - 8) * d;
                dst[b * 32 + l + 16] = ((v >> 4) - 8) * d;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void QuantizeRowQ4_K(float* src, BlockQ4_K* dst, int n)
    {
        int numSuperBlocks = n / 256;
        float* subScale = stackalloc float[8];
        float* subMin = stackalloc float[8];
        byte* sc = stackalloc byte[8];
        byte* m = stackalloc byte[8];
        byte* L = stackalloc byte[256];

        for (int sb = 0; sb < numSuperBlocks; sb++)
        {
            float* sbSrc = src + sb * 256;
            BlockQ4_K* sbDst = dst + sb;

            float maxScale = 0f;
            float maxMin = 0f;

            for (int ib = 0; ib < 8; ib++)
            {
                float* bSrc = sbSrc + ib * 32;
                BlockMinMax32(bSrc, out float bMin, out float bMax);

                if (bMin > 0f) bMin = 0f;
                float range = bMax - bMin;
                float scale = range / 15.0f;
                subScale[ib] = scale;
                if (scale > maxScale) maxScale = scale;

                float minOffset = -bMin;
                subMin[ib] = minOffset;
                if (minOffset > maxMin) maxMin = minOffset;
            }

            float invScale = maxScale > 0f ? 63.0f / maxScale : 0f;
            float invMin = maxMin > 0f ? 63.0f / maxMin : 0f;

            for (int ib = 0; ib < 8; ib++)
            {
                int scQuant = (int)MathF.Round(subScale[ib] * invScale);
                sc[ib] = (byte)Math.Clamp(scQuant, 0, 63);

                int mQuant = (int)MathF.Round(subMin[ib] * invMin);
                m[ib] = (byte)Math.Clamp(mQuant, 0, 63);
            }

            new Span<byte>(sbDst->Scales, 12).Clear();
            for (int j = 0; j < 8; j++)
            {
                SetScaleMinK4(j, sbDst->Scales, sc[j], m[j]);
            }

            sbDst->Delta = (Half)(maxScale / 63.0f);
            sbDst->DeltaMin = (Half)(maxMin / 63.0f);

            float dSuper = (float)sbDst->Delta;
            float dminSuper = (float)sbDst->DeltaMin;

            for (int ib = 0; ib < 8; ib++)
            {
                float d = dSuper * sc[ib];
                float dm = dminSuper * m[ib];
                float invD = d > 0f ? 1.0f / d : 0f;

                float* bSrc = sbSrc + ib * 32;
                byte* bL = L + ib * 32;

                for (int i = 0; i < 32; i++)
                {
                    int q = invD > 0f ? (int)MathF.Round((bSrc[i] + dm) * invD) : 0;
                    bL[i] = (byte)Math.Clamp(q, 0, 15);
                }
            }

            byte* qDst = sbDst->Qs;
            for (int g = 0; g < 256; g += 64)
            {
                for (int l = 0; l < 32; l++)
                {
                    qDst[l] = (byte)(L[g + l] | (L[g + l + 32] << 4));
                }
                qDst += 32;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DequantizeRowQ4_K(BlockQ4_K* src, float* dst, int n)
    {
        QuantKernels.DequantizeQ4_K(src, dst, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetScaleMinK4(int j, byte* scales, byte sc, byte m)
    {
        if (j < 4)
        {
            scales[j] = (byte)((scales[j] & 0xC0) | (sc & 0x3F));
            scales[j + 4] = (byte)((scales[j + 4] & 0xC0) | (m & 0x3F));
        }
        else
        {
            scales[j + 4] = (byte)((sc & 0x0F) | ((m & 0x0F) << 4));
            scales[j - 4] = (byte)((scales[j - 4] & 0x3F) | ((sc >> 4) << 6));
            scales[j]     = (byte)((scales[j] & 0x3F)     | ((m >> 4) << 6));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4)   | ((q[j] >> 6) << 4));
        }
    }

    private static bool TryParseLayerName(string tensorName, out int layerIndex, out string rawProj)
    {
        layerIndex = -1;
        rawProj = string.Empty;
        var parts = tensorName.Split('.');
        if (parts.Length == 4 && parts[0] == "blk" && int.TryParse(parts[1], out layerIndex) && parts[3] == "weight")
        {
            rawProj = parts[2];
            return true;
        }
        return false;
    }

    private static List<RowSliceMerge>? ResolveMergeSlices(GgufFile baseGguf, GgufTensorInfo t, ParsedLoraAdapter adapter)
    {
        if (!TryParseLayerName(t.Name, out int layer, out string rawProj) || layer < 0 || layer >= adapter.LayerCount)
        {
            return null;
        }

        string? canonical = CanonicalizeProjection(rawProj);
        if (canonical != null && adapter.TryGetSlot(layer, canonical, out var directSlot) && (directSlot.A != null || directSlot.B != null))
        {
            if (t.DimensionsCount != 2)
            {
                throw new InvalidDataException($"LoRA target tensor '{t.Name}' must be 2D, got {t.DimensionsCount}D.");
            }
            int inDim = (int)t.Dimensions[0];
            int outDim = (int)t.Dimensions[1];
            ValidateAndResolveSlotLayout(t.Name, directSlot.A, directSlot.B, directSlot.InDim, directSlot.OutDim, inDim, outDim, adapter.Rank, out bool isAPeft, out bool isBPeft);
            return [new RowSliceMerge(0, outDim, directSlot, isAPeft, isBPeft)];
        }

        // Composite / fused GGUF tensor handling
        if (rawProj == "attn_qkv" && !baseGguf.Tensors.ContainsKey($"blk.{layer}.ssm_out.weight"))
        {
            bool hasQ = adapter.TryGetSlot(layer, "attn_q", out var qSlot) && (qSlot.A != null || qSlot.B != null);
            bool hasK = adapter.TryGetSlot(layer, "attn_k", out var kSlot) && (kSlot.A != null || kSlot.B != null);
            bool hasV = adapter.TryGetSlot(layer, "attn_v", out var vSlot) && (vSlot.A != null || vSlot.B != null);
            if (!hasQ && !hasK && !hasV) return null;

            if (t.DimensionsCount != 2)
                throw new InvalidDataException($"LoRA target tensor '{t.Name}' must be 2D, got {t.DimensionsCount}D.");

            int inDim = (int)t.Dimensions[0];
            int outDim = (int)t.Dimensions[1];
            int headDim = (baseGguf.TryGetTensor($"blk.{layer}.attn_q_norm.weight", out var qNorm) && qNorm != null && qNorm.Dimensions.Length > 0)
                ? (int)qNorm.Dimensions[0]
                : (baseGguf.HeadDim > 0 ? baseGguf.HeadDim : (baseGguf.HeadCount > 0 ? baseGguf.EmbeddingLength / baseGguf.HeadCount : 128));
            int layerHeadsKv = (baseGguf.HeadCountKvPattern != null && layer < baseGguf.HeadCountKvPattern.Length)
                ? baseGguf.HeadCountKvPattern[layer]
                : baseGguf.HeadCountKv;
            int qDim = baseGguf.HeadCount * headDim;
            int kvDim = layerHeadsKv * headDim;
            int qRows = outDim == 2 * qDim + 2 * kvDim ? 2 * qDim : (outDim == qDim + 2 * kvDim ? qDim : -1);
            if (qRows <= 0 || qRows + 2 * kvDim != outDim)
            {
                throw new NotSupportedException($"Cannot split fused full-attention '{t.Name}' (outDim={outDim}) into Q({qDim}/{2 * qDim})+K({kvDim})+V({kvDim}).");
            }

            var slices = new List<RowSliceMerge>(3);
            if (hasQ)
            {
                ValidateAndResolveSlotLayout(t.Name + "[q]", qSlot.A, qSlot.B, qSlot.InDim, qSlot.OutDim, inDim, qRows, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(0, qRows, qSlot, isAPeft, isBPeft));
            }
            if (hasK)
            {
                ValidateAndResolveSlotLayout(t.Name + "[k]", kSlot.A, kSlot.B, kSlot.InDim, kSlot.OutDim, inDim, kvDim, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(qRows, kvDim, kSlot, isAPeft, isBPeft));
            }
            if (hasV)
            {
                ValidateAndResolveSlotLayout(t.Name + "[v]", vSlot.A, vSlot.B, vSlot.InDim, vSlot.OutDim, inDim, kvDim, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(qRows + kvDim, kvDim, vSlot, isAPeft, isBPeft));
            }
            return slices;
        }

        if (rawProj == "attn_qkv_gate")
        {
            bool hasQkv = adapter.TryGetSlot(layer, "attn_qkv", out var qkvSlot) && (qkvSlot.A != null || qkvSlot.B != null);
            bool hasGate = adapter.TryGetSlot(layer, "attn_gate", out var gateSlot) && (gateSlot.A != null || gateSlot.B != null);
            if (!hasQkv && !hasGate) return null;

            if (t.DimensionsCount != 2)
                throw new InvalidDataException($"LoRA target tensor '{t.Name}' must be 2D, got {t.DimensionsCount}D.");

            int inDim = (int)t.Dimensions[0];
            int outDim = (int)t.Dimensions[1];
            int nZ = baseGguf.SsmInnerSize > 0 ? baseGguf.SsmInnerSize : baseGguf.SsmStateSize * baseGguf.SsmTimeStepRank;
            int nQkv = outDim - nZ;
            if (nQkv <= 0 || nZ <= 0)
            {
                throw new NotSupportedException($"Cannot split fused GDN '{t.Name}' (outDim={outDim}, nZ={nZ}).");
            }

            var slices = new List<RowSliceMerge>(2);
            if (hasQkv)
            {
                ValidateAndResolveSlotLayout(t.Name + "[qkv]", qkvSlot.A, qkvSlot.B, qkvSlot.InDim, qkvSlot.OutDim, inDim, nQkv, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(0, nQkv, qkvSlot, isAPeft, isBPeft));
            }
            if (hasGate)
            {
                ValidateAndResolveSlotLayout(t.Name + "[gate]", gateSlot.A, gateSlot.B, gateSlot.InDim, gateSlot.OutDim, inDim, nZ, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(nQkv, nZ, gateSlot, isAPeft, isBPeft));
            }
            return slices;
        }

        if (rawProj is "ssm_beta_alpha" or "ssm_ba")
        {
            bool hasBeta = adapter.TryGetSlot(layer, "ssm_beta", out var betaSlot) && (betaSlot.A != null || betaSlot.B != null);
            bool hasAlpha = adapter.TryGetSlot(layer, "ssm_alpha", out var alphaSlot) && (alphaSlot.A != null || alphaSlot.B != null);
            if (!hasBeta && !hasAlpha) return null;

            if (t.DimensionsCount != 2)
                throw new InvalidDataException($"LoRA target tensor '{t.Name}' must be 2D, got {t.DimensionsCount}D.");

            int inDim = (int)t.Dimensions[0];
            int outDim = (int)t.Dimensions[1];
            int nBeta = outDim / 2;
            int nAlpha = outDim - nBeta;
            if (nBeta <= 0 || nAlpha <= 0) return null;

            var slices = new List<RowSliceMerge>(2);
            if (hasBeta)
            {
                ValidateAndResolveSlotLayout(t.Name + "[beta]", betaSlot.A, betaSlot.B, betaSlot.InDim, betaSlot.OutDim, inDim, nBeta, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(0, nBeta, betaSlot, isAPeft, isBPeft));
            }
            if (hasAlpha)
            {
                ValidateAndResolveSlotLayout(t.Name + "[alpha]", alphaSlot.A, alphaSlot.B, alphaSlot.InDim, alphaSlot.OutDim, inDim, nAlpha, adapter.Rank, out bool isAPeft, out bool isBPeft);
                slices.Add(new RowSliceMerge(nBeta, nAlpha, alphaSlot, isAPeft, isBPeft));
            }
            return slices;
        }

        return null;
    }

    private static string? CanonicalizeProjection(string projName)
    {
        return projName switch
        {
            "attn_q" or "q_proj" => "attn_q",
            "attn_k" or "k_proj" => "attn_k",
            "attn_v" or "v_proj" => "attn_v",
            "attn_output" or "o_proj" => "attn_output",
            "ffn_gate" or "gate_proj" => "ffn_gate",
            "ffn_up" or "up_proj" => "ffn_up",
            "ffn_down" or "down_proj" => "ffn_down",
            "attn_qkv" or "in_proj_qkv" or "qkv_proj" => "attn_qkv",
            "attn_gate" or "in_proj_z" => "attn_gate",
            "ssm_alpha" or "in_proj_a" or "alpha_proj" => "ssm_alpha",
            "ssm_beta" or "in_proj_b" or "beta_proj" => "ssm_beta",
            "ssm_out" or "out_proj" => "ssm_out",
            _ => null
        };
    }

    private readonly record struct RowSliceMerge(int RowStart, int RowCount, AdapterSlot Slot, bool IsAPeftRowMajor, bool IsBPeftRowMajor);

    private readonly record struct PinnedRowSlice(
        int RowStart,
        int RowCount,
        GCHandle HandleA,
        GCHandle HandleB,
        IntPtr PtrA,
        IntPtr PtrB,
        bool IsAPeftRowMajor,
        bool IsBPeftRowMajor);

    private sealed class AdapterSlot
    {
        public float[]? A { get; set; }
        public float[]? B { get; set; }
        public int InDim { get; set; }
        public int OutDim { get; set; }
    }

    private sealed class ParsedLoraAdapter
    {
        public int LayerCount { get; init; }
        public int Rank { get; init; }
        public float Alpha { get; init; }
        public float Scaling => Rank > 0 ? Alpha / Rank : 1.0f;

        private readonly Dictionary<(int Layer, string Proj), AdapterSlot> _slots = [];

        public bool TryGetSlot(int layer, string canonicalProj, out AdapterSlot slot)
            => _slots.TryGetValue((layer, canonicalProj), out slot!);

        public static ParsedLoraAdapter Load(string adapterFilePath)
        {
            string binPath = adapterFilePath;
            if (Directory.Exists(adapterFilePath))
            {
                string candidate = Path.Combine(adapterFilePath, "adapter_model.bin");
                if (File.Exists(candidate)) binPath = candidate;
            }

            if (!File.Exists(binPath))
                throw new FileNotFoundException($"LoRA adapter file not found: {adapterFilePath}");

            using var fs = File.OpenRead(binPath);
            using var br = new BinaryReader(fs);

            LoraBinaryFormat.ValidateAndReadHeader(fs, br, out int layerCount, out int rank, out float alpha, out int tensorCount);

            var parsed = new ParsedLoraAdapter
            {
                LayerCount = layerCount,
                Rank = rank,
                Alpha = alpha
            };

            for (int t = 0; t < tensorCount; t++)
            {
                string name = LoraBinaryFormat.ReadBoundedString(fs);
                if (fs.Length - fs.Position < sizeof(int) * 2)
                    throw new InvalidDataException($"Truncated LoRA tensor dimensions for '{name}'.");

                int dim0 = br.ReadInt32();
                int dim1 = br.ReadInt32();
                if (dim0 <= 0 || dim1 <= 0)
                    throw new InvalidDataException($"Invalid LoRA tensor dimensions [{dim0}, {dim1}] for '{name}'.");

                long elemCountLong = checked((long)dim0 * dim1);
                if (elemCountLong > int.MaxValue / sizeof(float))
                    throw new InvalidDataException($"LoRA tensor '{name}' element count {elemCountLong} exceeds maximum supported buffer size.");

                long byteCount = checked(elemCountLong * sizeof(float));
                if (byteCount > fs.Length - fs.Position)
                    throw new InvalidDataException($"Truncated LoRA binary payload for '{name}': expected {byteCount} bytes, only {fs.Length - fs.Position} remaining.");

                int elemCount = (int)elemCountLong;
                var parts = name.Split('.');
                if (parts.Length >= 5 && parts[0] == "blk" && int.TryParse(parts[1], out int layerIdx))
                {
                    if (layerIdx < 0 || layerIdx >= layerCount)
                        throw new InvalidDataException($"LoRA tensor '{name}' references out-of-range layer index {layerIdx} (layerCount={layerCount}).");

                    string? canonicalProj = CanonicalizeProjection(parts[2]);
                    string matrix = parts[3];
                    if (canonicalProj != null && (matrix == "lora_a" || matrix == "lora_b"))
                    {
                        float[] data = new float[elemCount];
                        fs.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
                        LoraBinaryFormat.ValidateFinite(data, $"Merged adapter tensor '{name}'");

                        if (!parsed._slots.TryGetValue((layerIdx, canonicalProj), out var slot))
                        {
                            slot = new AdapterSlot();
                            parsed._slots[(layerIdx, canonicalProj)] = slot;
                        }

                        if (matrix == "lora_a")
                        {
                            slot.A = data;
                            slot.InDim = dim0;
                        }
                        else
                        {
                            slot.B = data;
                            slot.OutDim = dim1;
                        }
                        continue;
                    }
                }

                fs.Seek(byteCount, SeekOrigin.Current);
            }

            if (fs.Position != fs.Length)
                throw new InvalidDataException($"Unexpected trailing bytes ({fs.Length - fs.Position}) in LoRA adapter '{binPath}'.");

            return parsed;
        }
    }
}
