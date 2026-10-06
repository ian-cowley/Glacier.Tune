namespace Glacier.Tune.Tests;

using System;
using System.Collections.Generic;
using Glacier.Tune.Diagnostics;
using Xunit;

public class DiagnosticsTests : IDisposable
{
    public DiagnosticsTests()
    {
        GlacierDiagnostics.Reset();
    }

    public void Dispose()
    {
        GlacierDiagnostics.Reset();
    }

    [Fact]
    public void GlacierDiagnostics_Default_UsesNullGlacierLogger()
    {
        GlacierDiagnostics.Reset();
        Assert.NotNull(GlacierDiagnostics.Logger);
        Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Trace));
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Debug));
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Information));
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Warning));
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Error));
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Critical));

        // Must remain completely silent and not throw when invoked
        GlacierDiagnostics.LogTrace("Trace message");
        GlacierDiagnostics.LogDebug("Debug message");
        GlacierDiagnostics.LogInformation("Info message");
        GlacierDiagnostics.LogWarning("Warning message", new InvalidOperationException("Test warning"));
        GlacierDiagnostics.LogError("Error message", new Exception("Test failure"));
        GlacierDiagnostics.LogCritical("Critical message");
    }

    [Fact]
    public void GlacierDiagnostics_AssignNull_RevertsToNullGlacierLogger()
    {
        GlacierDiagnostics.Logger = null!;
        Assert.NotNull(GlacierDiagnostics.Logger);
        Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);
    }

    [Fact]
    public void DelegateGlacierLogger_RedirectsAndCapturesAllLogLevels()
    {
        var logs = new List<(LogLevel Level, string Message, Exception? Exception)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            logs.Add((level, msg, ex));
        }, LogLevel.Trace);

        var testEx = new InvalidOperationException("Test exception");

        GlacierDiagnostics.LogTrace("Trace entry");
        GlacierDiagnostics.LogDebug("Debug entry");
        GlacierDiagnostics.LogInformation("Info entry");
        GlacierDiagnostics.LogWarning("Warning entry", testEx);
        GlacierDiagnostics.LogError("Error entry", testEx);
        GlacierDiagnostics.LogCritical("Critical entry");

        Assert.Equal(6, logs.Count);

        Assert.Equal(LogLevel.Trace, logs[0].Level);
        Assert.Equal("Trace entry", logs[0].Message);
        Assert.Null(logs[0].Exception);

        Assert.Equal(LogLevel.Debug, logs[1].Level);
        Assert.Equal("Debug entry", logs[1].Message);
        Assert.Null(logs[1].Exception);

        Assert.Equal(LogLevel.Information, logs[2].Level);
        Assert.Equal("Info entry", logs[2].Message);
        Assert.Null(logs[2].Exception);

        Assert.Equal(LogLevel.Warning, logs[3].Level);
        Assert.Equal("Warning entry", logs[3].Message);
        Assert.Same(testEx, logs[3].Exception);

        Assert.Equal(LogLevel.Error, logs[4].Level);
        Assert.Equal("Error entry", logs[4].Message);
        Assert.Same(testEx, logs[4].Exception);

        Assert.Equal(LogLevel.Critical, logs[5].Level);
        Assert.Equal("Critical entry", logs[5].Message);
        Assert.Null(logs[5].Exception);
    }

    [Fact]
    public void DelegateGlacierLogger_SingleStringDelegate_CapturesFormattedMessages()
    {
        var lines = new List<string>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger(lines.Add, LogLevel.Information);

        GlacierDiagnostics.LogDebug("Should be filtered out");
        GlacierDiagnostics.LogInformation("Important training step info");
        GlacierDiagnostics.LogWarning("Watch out", new Exception("WarnEx"));

        Assert.Equal(2, lines.Count);
        Assert.Contains("[Information] Important training step info", lines[0]);
        Assert.Contains("[Warning] Watch out: System.Exception: WarnEx", lines[1]);
    }

    [Fact]
    public void DelegateGlacierLogger_RespectsMinimumLevelFilter()
    {
        var logs = new List<(LogLevel Level, string Message, Exception? Exception)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            logs.Add((level, msg, ex));
        }, LogLevel.Warning);

        GlacierDiagnostics.LogTrace("Trace filtered");
        GlacierDiagnostics.LogDebug("Debug filtered");
        GlacierDiagnostics.LogInformation("Info filtered");
        GlacierDiagnostics.LogWarning("Warning kept");
        GlacierDiagnostics.LogError("Error kept");
        GlacierDiagnostics.LogCritical("Critical kept");

        Assert.Equal(3, logs.Count);
        Assert.Equal(LogLevel.Warning, logs[0].Level);
        Assert.Equal("Warning kept", logs[0].Message);
        Assert.Equal(LogLevel.Error, logs[1].Level);
        Assert.Equal("Error kept", logs[1].Message);
        Assert.Equal(LogLevel.Critical, logs[2].Level);
        Assert.Equal("Critical kept", logs[2].Message);
    }

    [Fact]
    public void ConsoleGlacierLogger_HonorsLevelsAndDoesNotThrow()
    {
        var loggerWithoutColors = new ConsoleGlacierLogger(LogLevel.Information, useColors: false);
        Assert.True(loggerWithoutColors.IsEnabled(LogLevel.Information));
        Assert.True(loggerWithoutColors.IsEnabled(LogLevel.Warning));
        Assert.False(loggerWithoutColors.IsEnabled(LogLevel.Debug));
        Assert.False(loggerWithoutColors.IsEnabled(LogLevel.None));

        // Ensure logging methods execute cleanly without throwing
        loggerWithoutColors.Log(LogLevel.Information, "Console test info");
        loggerWithoutColors.Log(LogLevel.Warning, "Console test warning", new Exception("Test warn"));

        var loggerWithColors = new ConsoleGlacierLogger(LogLevel.Debug, useColors: true);
        Assert.True(loggerWithColors.IsEnabled(LogLevel.Debug));
        loggerWithColors.Log(LogLevel.Debug, "Console test debug with color");
        loggerWithColors.Log(LogLevel.Error, "Console test error with color", new Exception("Test err"));
    }

    [Fact]
    public void FallbackWarningTelemetry_CapturedWithExceptionDetails()
    {
        var capturedLogs = new List<(LogLevel Level, string Message, Exception? Ex)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            capturedLogs.Add((level, msg, ex));
        }, LogLevel.Trace);

        var simulatedGpuEx = new DllNotFoundException("CUDA driver not installed");
        GlacierDiagnostics.LogWarning($"[GPU WARNING] Failed to initialize GPU VRAM base engine ({simulatedGpuEx.Message}). Falling back to CPU memory-mapped base execution.", simulatedGpuEx);

        Assert.Single(capturedLogs);
        var entry = capturedLogs[0];
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Contains("[GPU WARNING]", entry.Message);
        Assert.Contains("CUDA driver not installed", entry.Message);
        Assert.NotNull(entry.Ex);
        Assert.IsType<DllNotFoundException>(entry.Ex);
        Assert.Equal("CUDA driver not installed", entry.Ex.Message);
    }
}
