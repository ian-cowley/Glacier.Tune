using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Tensor.Core;

namespace Glacier.Tune.Model;

/// <summary>
/// Unified layer block interface supporting both full-attention <see cref="TransformerBlock"/>
/// and Gated DeltaNet <see cref="GdnBlock"/> layers in hybrid models.
/// </summary>
public interface ITuneBlock : IDisposable
{
    int LayerIndex { get; }
    IReadOnlyList<Tensor<float>> TrainableParameters { get; }

    Tensor<float> Forward(Tensor<float> x, int seqLen, bool saveActivations, out IDisposable? state);
    Tensor<float> Backward(Tensor<float> dOutput, IDisposable state, Tensor<float> xInput, int seqLen, AutogradScratchWorkspace? workspace = null);
    void ConnectGpuWeights(Glacier.Tune.Gpu.GpuLoraEngine engine);
    LoraAdapter? FindAdapter(string projectionName);
    int CountEnabledTensors();
    void WriteAdapterTensors(BinaryWriter bw);
    void WriteManifestEntries(StreamWriter writer);
}

internal static class TuneBlockSerialization
{
    public static void WriteAdapterPair(BinaryWriter bw, int layerIndex, string ggufProj, LoraAdapter? adapter)
    {
        if (adapter == null || !adapter.IsEnabled) return;
        WriteTensor(bw, $"blk.{layerIndex}.{ggufProj}.lora_a.weight", adapter.AdapterA);
        WriteTensor(bw, $"blk.{layerIndex}.{ggufProj}.lora_b.weight", adapter.AdapterB);
    }

    public static void WriteManifestEntry(StreamWriter writer, int layerIndex, string manifestProj, LoraAdapter? adapter)
    {
        if (adapter == null || !adapter.IsEnabled) return;
        writer.WriteLine(
            $"blk.{layerIndex}.{manifestProj}: A=[{adapter.AdapterA.Shape[0]},{adapter.AdapterA.Shape[1]}], B=[{adapter.AdapterB.Shape[0]},{adapter.AdapterB.Shape[1]}]");
    }

    public static void WriteTensor(BinaryWriter bw, string name, Tensor<float> tensor)
    {
        bw.Write(name);
        bw.Write(tensor.Shape[0]);
        bw.Write(tensor.Shape[1]);
        var span = tensor.AsSpan();
        bw.Write(MemoryMarshal.AsBytes(span));
    }
}
