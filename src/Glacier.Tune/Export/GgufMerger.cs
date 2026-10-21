namespace Glacier.Tune.Export;

using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Tune.Diagnostics;

/// <summary>
/// Target quantization representation for adapted LoRA projections.
/// </summary>
public enum TargetQuantization
{
    /// <summary>
    /// Auto-select quantization based on base tensor type (defaults to Q4_K for Q4_K/Q6_K).
    /// </summary>
    Auto,

    /// <summary>
    /// Super-block 4-bit k-quantization (256 weights in 144 bytes, 4.5 bpw).
    /// </summary>
    Q4_K,

    /// <summary>
    /// Standard 4-bit quantization (32 weights in 18 bytes, 4.5 bpw).
    /// </summary>
    Q4_0,

    /// <summary>
    /// Standard 8-bit quantization (32 weights in 34 bytes, 8.5 bpw).
    /// </summary>
    Q8_0
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
/// converting adapted layers into lossless Q8_0 or compact Q4_K/Q4_0 quantized representations
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
        var progress = options?.ProgressCallback ?? (msg => GlacierDiagnostics.LogInformation(msg));
        var targetQuant = options?.TargetQuantization ?? TargetQuantization.Auto;

        if (!File.Exists(baseGgufPath))
            throw new FileNotFoundException($"Base GGUF model not found: {baseGgufPath}");

        if (!File.Exists(adapterBinPath))
            throw new FileNotFoundException($"LoRA adapter file not found: {adapterBinPath}");

        string outFileName = Path.GetFileName(outputGgufPath);
        if (outFileName.Equals("qwen2.5-coder-7b-enterprise-q8_0.gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("CRITICAL SAFETY GUARD: Target cannot be qwen2.5-coder-7b-enterprise-q8_0.gguf! That is the preserved Python model.");
        }

        string? outDir = Path.GetDirectoryName(outputGgufPath);
        if (!string.IsNullOrEmpty(outDir) && !Directory.Exists(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        var sw = Stopwatch.StartNew();
        progress($"[1/4] Loading unmanaged base GGUF model and LoRA adapter...");
        using var baseGguf = new GgufFile(baseGgufPath);
        var adapter = LoraAdapterWeights.Load(adapterBinPath);

        progress($"  Base Model:    {baseGguf.Architecture} ({baseGguf.BlockCount} blocks, {baseGguf.TensorCount} tensors)");
        progress($"  LoRA Adapter:  {adapter.LayerCount} layers, rank={adapter.Rank}, alpha={adapter.Alpha}, scaling={adapter.Scaling:F2}");
        progress($"  Base Size:     {new FileInfo(baseGgufPath).Length:N0} bytes");

        // 1. Calculate Tensor Directory and Data Layout
        progress($"[2/4] Computing merged tensor directory and aligned offsets...");
        int tensorCount = (int)baseGguf.TensorCount;
        var newTypes = new GgufType[tensorCount];
        var newOffsets = new ulong[tensorCount];
        var newByteSizes = new ulong[tensorCount];
        var isAdaptedList = new bool[tensorCount];

        int adaptedCount = 0;
        int nonAdaptedCount = 0;

        // Calculate size of Tensor Directory entries
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

        // Header + Metadata + Tensor Directory
        ulong totalHeaderBytes = baseGguf.MetadataEndOffset + (ulong)tensorDirByteCount;
        ulong remHeader = totalHeaderBytes % baseGguf.Alignment;
        ulong tensorDataOffset = remHeader == 0 ? totalHeaderBytes : totalHeaderBytes + (baseGguf.Alignment - remHeader);

        // Compute new aligned data offsets
        ulong currentPayloadOffset = 0;
        for (int i = 0; i < tensorCount; i++)
        {
            var t = baseGguf.TensorList[i];
            bool isAdapted = TryGetLoraProjection(t.Name, out int layer, out LoraProjection proj)
                             && layer < adapter.LayerCount
                             && adapter.GetProjections(layer).Get(proj).A != null;

            isAdaptedList[i] = isAdapted;

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

        progress($"  Adapted Projections: {adaptedCount} (fused into target {targetQuant})");
        progress($"  Non-Adapted Tensors: {nonAdaptedCount} (preserved bit-exact)");
        progress($"  Header End Offset:   {totalHeaderBytes:N0} bytes");
        progress($"  Tensor Data Offset:  {tensorDataOffset:N0} bytes (aligned to {baseGguf.Alignment} bytes)");

        // 2. Open Output Stream and Write Header + Metadata + Directory
        progress($"[3/4] Writing merged GGUF header, metadata, and tensor directory...");
        using var outFs = new FileStream(outputGgufPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1024 * 1024);
        using var bw = new BinaryWriter(outFs, Encoding.UTF8, leaveOpen: true);

        // Copy raw Header + Metadata exactly from base GGUF
        byte* pBase = baseGguf.BasePointer;
        ulong metadataBytes = baseGguf.MetadataEndOffset;
        outFs.Write(new ReadOnlySpan<byte>(pBase, (int)metadataBytes));

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

        // Write padding bytes up to tensorDataOffset
        long currentPos = outFs.Position;
        int headerPad = (int)(tensorDataOffset - (ulong)currentPos);
        if (headerPad > 0)
        {
            outFs.Write(new byte[headerPad]);
        }

        // 3. Process and Stream Tensor Data
        progress($"[4/4] Streaming and fusing weights into {outputGgufPath}...");
        int lastReportedLayer = -1;
        var parallelOpts = new ParallelOptions { MaxDegreeOfParallelism = options?.MaxDegreeOfParallelism ?? Environment.ProcessorCount };

        for (int i = 0; i < tensorCount; i++)
        {
            var t = baseGguf.TensorList[i];
            long targetPos = (long)(tensorDataOffset + newOffsets[i]);
            long padNeeded = targetPos - outFs.Position;
            if (padNeeded > 0)
            {
                outFs.Write(new byte[(int)padNeeded]);
            }

            if (!isAdaptedList[i])
            {
                // Non-adapted tensor: direct zero-copy block transfer from base GGUF
                byte* srcPtr = baseGguf.GetTensorPointer(t);
                ulong bytesRemaining = newByteSizes[i];
                byte[] copyBuffer = new byte[Math.Min(16 * 1024 * 1024, (int)Math.Min((ulong)int.MaxValue, bytesRemaining))];
                while (bytesRemaining > 0)
                {
                    int toCopy = (int)Math.Min((ulong)copyBuffer.Length, bytesRemaining);
                    Marshal.Copy((IntPtr)srcPtr, copyBuffer, 0, toCopy);
                    outFs.Write(copyBuffer, 0, toCopy);
                    srcPtr += toCopy;
                    bytesRemaining -= (ulong)toCopy;
                }
            }
            else
            {
                // Adapted projection: multithreaded dequantize + LoRA delta fusion + target quantize
                TryGetLoraProjection(t.Name, out int layer, out LoraProjection proj);
                if (layer != lastReportedLayer)
                {
                    lastReportedLayer = layer;
                    progress($"  Fusing Transformer Block {layer:D2}/{baseGguf.BlockCount - 1}...");
                }

                int inDim = (int)t.Dimensions[0];
                int outDim = (int)t.Dimensions[1];
                GgufType targetType = newTypes[i];
                int rowBytes = GetRowByteSize(targetType, inDim);
                ulong totalTensorBytes = (ulong)outDim * (ulong)rowBytes;

                byte* pDst = (byte*)NativeMemory.Alloc((nuint)totalTensorBytes);
                try
                {
                    var (A, B, _, _) = adapter.GetProjections(layer).Get(proj);
                    int rank = adapter.Rank;
                    float scaling = adapter.Scaling;
                    IntPtr ptrBaseData = (IntPtr)baseGguf.GetTensorPointer(t);
                    IntPtr ptrDst = (IntPtr)pDst;
                    GgufType baseType = t.Type;

                    fixed (float* pA = A, pB = B)
                    {
                        IntPtr ptrA = (IntPtr)pA;
                        IntPtr ptrB = (IntPtr)pB;

                        Parallel.For(0, outDim, parallelOpts, () => ArrayPool<float>.Shared.Rent(inDim), (r, loopState, rowArr) =>
                        {
                            fixed (float* rowFloats = rowArr)
                            {
                                float* curA = (float*)ptrA;
                                float* curB = (float*)ptrB;
                                byte* curBase = (byte*)ptrBaseData;
                                byte* curDst = (byte*)ptrDst;

                                // 1. Dequantize base weight row r
                                QuantKernels.ExtractEmbedding(baseType, curBase, r, rowFloats, inDim);

                                // 2. Compute scaled column of B for row r: b_r[k] = scaling * B[k, r]
                                float* b_r = stackalloc float[rank];
                                for (int k = 0; k < rank; k++)
                                {
                                    b_r[k] = scaling * curB[k * outDim + r];
                                }

                                // 3. Add LoRA delta: delta[c] = sum_{k=0..rank-1} b_r[k] * A[c, k]
                                if (Vector256.IsHardwareAccelerated && rank == 16)
                                {
                                    Vector256<float> br0 = Vector256.Load(b_r);
                                    Vector256<float> br1 = Vector256.Load(b_r + 8);
                                    for (int c = 0; c < inDim; c++)
                                    {
                                        float* aRow = curA + (long)c * 16;
                                        Vector256<float> a0 = Vector256.Load(aRow);
                                        Vector256<float> a1 = Vector256.Load(aRow + 8);
                                        Vector256<float> prod0 = br0 * a0;
                                        Vector256<float> prod1 = br1 * a1;
                                        rowFloats[c] += Vector256.Sum(prod0 + prod1);
                                    }
                                }
                                else
                                {
                                    for (int c = 0; c < inDim; c++)
                                    {
                                        float* aRow = curA + (long)c * rank;
                                        float delta = 0f;
                                        for (int k = 0; k < rank; k++)
                                        {
                                            delta += b_r[k] * aRow[k];
                                        }
                                        rowFloats[c] += delta;
                                    }
                                }

                                // 4. Quantize fused row to target blocks
                                byte* dstRow = curDst + (long)r * rowBytes;
                                switch (targetType)
                                {
                                    case GgufType.Q4_K:
                                        QuantizeRowQ4_K(rowFloats, (BlockQ4_K*)dstRow, inDim);
                                        break;
                                    case GgufType.Q4_0:
                                        QuantizeRowQ4_0(rowFloats, (BlockQ4_0*)dstRow, inDim);
                                        break;
                                    case GgufType.Q8_0:
                                        QuantizeRowQ8_0(rowFloats, (BlockQ8_0*)dstRow, inDim);
                                        break;
                                    default:
                                        throw new NotSupportedException($"Unsupported target quantization: {targetType}");
                                }
                            }
                            return rowArr;
                        },
                        rowArr => ArrayPool<float>.Shared.Return(rowArr));
                    }

                    // Write fused tensor bytes
                    byte* pWrite = pDst;
                    ulong remaining = totalTensorBytes;
                    byte[] streamBuf = new byte[Math.Min(16 * 1024 * 1024, (int)Math.Min((ulong)int.MaxValue, remaining))];
                    while (remaining > 0)
                    {
                        int toWrite = (int)Math.Min((ulong)streamBuf.Length, remaining);
                        Marshal.Copy((IntPtr)pWrite, streamBuf, 0, toWrite);
                        outFs.Write(streamBuf, 0, toWrite);
                        pWrite += toWrite;
                        remaining -= (ulong)toWrite;
                    }
                }
                finally
                {
                    NativeMemory.Free(pDst);
                }
            }
        }

        outFs.Flush();
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void QuantizeRowQ8_0(float* src, BlockQ8_0* dst, int n)
    {
        int numBlocks = n / 32;
        for (int b = 0; b < numBlocks; b++)
        {
            float* blockSrc = src + b * 32;
            BlockQ8_0* blockDst = dst + b;

            float maxAbs = 0f;
            for (int i = 0; i < 32; i++)
            {
                float val = MathF.Abs(blockSrc[i]);
                if (val > maxAbs) maxAbs = val;
            }

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
    public static GgufType ResolveTargetType(TargetQuantization target, GgufType baseType, int inDim)
    {
        GgufType resolved = target switch
        {
            TargetQuantization.Q4_K => GgufType.Q4_K,
            TargetQuantization.Q4_0 => GgufType.Q4_0,
            TargetQuantization.Q8_0 => GgufType.Q8_0,
            TargetQuantization.Auto => baseType switch
            {
                GgufType.Q4_0 => GgufType.Q4_0,
                GgufType.Q8_0 => GgufType.Q8_0,
                _ => GgufType.Q4_K
            },
            _ => GgufType.Q4_K
        };

        if (resolved == GgufType.Q4_K && inDim % 256 != 0)
        {
            resolved = GgufType.Q4_0;
        }

        return resolved;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetRowByteSize(GgufType type, int inDim)
    {
        return type switch
        {
            GgufType.Q4_K => (inDim / 256) * sizeof(BlockQ4_K),
            GgufType.Q4_0 => (inDim / 32) * sizeof(BlockQ4_0),
            GgufType.Q8_0 => (inDim / 32) * sizeof(BlockQ8_0),
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
                float bMin = bSrc[0];
                float bMax = bSrc[0];
                for (int i = 1; i < 32; i++)
                {
                    float v = bSrc[i];
                    if (v < bMin) bMin = v;
                    if (v > bMax) bMax = v;
                }

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

            // Pack 8 6-bit scales and 8 6-bit mins into Scales[12]
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

            // Pack 256 4-bit quants into 128 bytes (pairs of 32-element sub-blocks)
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

    private static bool TryGetLoraProjection(string tensorName, out int layerIndex, out LoraProjection projection)
    {
        layerIndex = -1;
        projection = (LoraProjection)(-1);

        var parts = tensorName.Split('.');
        if (parts.Length == 4 && parts[0] == "blk" && int.TryParse(parts[1], out layerIndex) && parts[3] == "weight")
        {
            projection = parts[2] switch
            {
                "attn_q" => LoraProjection.Q,
                "attn_k" => LoraProjection.K,
                "attn_v" => LoraProjection.V,
                "attn_output" => LoraProjection.AttnOut,
                "ffn_gate" => LoraProjection.Gate,
                "ffn_up" => LoraProjection.Up,
                "ffn_down" => LoraProjection.Down,
                _ => (LoraProjection)(-1)
            };
            return (int)projection >= 0;
        }

        return false;
    }
}
