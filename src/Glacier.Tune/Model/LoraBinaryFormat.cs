namespace Glacier.Tune.Model;

using System;
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text;

/// <summary>
/// Shared validation and atomic-write helpers for the <c>0x41524F4C</c> (<c>'LORA'</c>) binary format.
/// </summary>
internal static class LoraBinaryFormat
{
    public const uint Magic = 0x41524F4C; // 'LORA'
    public const uint Version = 1;
    public const int HeaderBytes = 24;
    public const int MaxLayerCount = 4096;
    public const int MaxRank = 16384;
    public const int MaxTensorNameUtf8Bytes = 4096;

    public static void ValidateAndReadHeader(
        Stream fs,
        BinaryReader br,
        out int layerCount,
        out int rank,
        out float alpha,
        out int tensorCount)
    {
        if (fs.Length - fs.Position < HeaderBytes)
        {
            throw new InvalidDataException($"Truncated LoRA adapter header: expected at least {HeaderBytes} bytes, found {fs.Length - fs.Position}.");
        }

        uint magic = br.ReadUInt32();
        if (magic != Magic)
        {
            throw new InvalidDataException($"Invalid LoRA magic header: 0x{magic:X8} (expected 0x{Magic:X8} 'LORA').");
        }

        uint version = br.ReadUInt32();
        if (version != Version)
        {
            throw new InvalidDataException($"Unsupported LoRA adapter version: {version}.");
        }

        layerCount = br.ReadInt32();
        if (layerCount <= 0 || layerCount > MaxLayerCount)
        {
            throw new InvalidDataException($"Invalid LoRA layer count: {layerCount} (must be in 1..{MaxLayerCount}).");
        }

        rank = br.ReadInt32();
        if (rank < 0 || rank > MaxRank)
        {
            throw new InvalidDataException($"Invalid LoRA rank: {rank} (must be in 0..{MaxRank}).");
        }

        alpha = br.ReadSingle();
        if (!float.IsFinite(alpha) || (rank > 0 ? alpha <= 0f : alpha < 0f))
        {
            throw new InvalidDataException($"Invalid LoRA adapter alpha value: {alpha}.");
        }

        tensorCount = br.ReadInt32();
        if (tensorCount < 0)
        {
            throw new InvalidDataException($"Invalid negative LoRA tensor count: {tensorCount}.");
        }

        if (tensorCount > 0 && rank <= 0)
        {
            throw new InvalidDataException($"LoRA adapter specifies {tensorCount} tensors with rank {rank}.");
        }

        long remaining = fs.Length - fs.Position;
        // Minimum bytes per tensor record: 1B string length + 1B name + 4B dim0 + 4B dim1 + 4B float = 14B (conservative lower bound 9B)
        if ((long)tensorCount * 9L > remaining)
        {
            throw new InvalidDataException($"LoRA tensor count {tensorCount} exceeds remaining stream capacity ({remaining} bytes).");
        }
    }

    public static string ReadBoundedString(Stream fs, int maxUtf8Bytes = MaxTensorNameUtf8Bytes)
    {
        int byteLen = 0;
        int shift = 0;
        while (true)
        {
            if (shift > 28)
            {
                throw new InvalidDataException("Corrupted 7-bit string length prefix in LoRA binary.");
            }

            int b = fs.ReadByte();
            if (b < 0)
            {
                throw new InvalidDataException("Unexpected end of stream reading LoRA tensor name length.");
            }

            if (shift == 28 && (b & 0x78) != 0)
            {
                throw new InvalidDataException("Corrupted 7-bit string length prefix in LoRA binary.");
            }

            byteLen |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                break;
            }
            shift += 7;
        }

        if (byteLen <= 0 || byteLen > maxUtf8Bytes)
        {
            throw new InvalidDataException($"Invalid LoRA tensor name byte length: {byteLen} (must be in 1..{maxUtf8Bytes}).");
        }

        long remaining = fs.Length - fs.Position;
        if (byteLen > remaining)
        {
            throw new InvalidDataException($"Truncated LoRA tensor name: expected {byteLen} bytes, only {remaining} remaining.");
        }

        if (byteLen <= 512)
        {
            Span<byte> stackBuf = stackalloc byte[byteLen];
            fs.ReadExactly(stackBuf);
            return Encoding.UTF8.GetString(stackBuf);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(byteLen);
        try
        {
            var span = rented.AsSpan(0, byteLen);
            fs.ReadExactly(span);
            return Encoding.UTF8.GetString(span);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ValidateFinite(ReadOnlySpan<float> values, string context)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && values.Length >= Vector512<float>.Count)
        {
            int step = Vector512<float>.Count;
            int limit = values.Length - step;
            for (; i <= limit; i += step)
            {
                var v = Vector512.LoadUnsafe(ref Unsafe.AsRef(in values[i]));
                // NaN/Inf check: (v - v) == 0 is true iff all lanes are finite
                if ((v - v) != Vector512<float>.Zero)
                {
                    ThrowNonFinite(values.Slice(i, step), i, context);
                }
            }
        }
        if (Vector256.IsHardwareAccelerated && values.Length - i >= Vector256<float>.Count)
        {
            int step = Vector256<float>.Count;
            int limit = values.Length - step;
            for (; i <= limit; i += step)
            {
                var v = Vector256.LoadUnsafe(ref Unsafe.AsRef(in values[i]));
                if ((v - v) != Vector256<float>.Zero)
                {
                    ThrowNonFinite(values.Slice(i, step), i, context);
                }
            }
        }
        for (; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                throw new InvalidDataException($"Non-finite float value ({values[i]}) encountered in {context} at index {i}.");
            }
        }
    }

    private static void ThrowNonFinite(ReadOnlySpan<float> slice, int baseIdx, string context)
    {
        for (int k = 0; k < slice.Length; k++)
        {
            if (!float.IsFinite(slice[k]))
            {
                throw new InvalidDataException($"Non-finite float value ({slice[k]}) encountered in {context} at index {baseIdx + k}.");
            }
        }
    }

    public static string CreateTempSiblingPath(string targetPath)
    {
        string full = Path.GetFullPath(targetPath);
        string? dir = Path.GetDirectoryName(full);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }
        return $"{full}.tmp.{Guid.NewGuid():N}";
    }
}
