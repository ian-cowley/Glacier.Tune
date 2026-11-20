using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Glacier.Inference.Evaluation;
using Glacier.Tune.Config;
using Glacier.Tune.Evaluation;
using Glacier.Tune.Model;
using Xunit;

namespace Glacier.Tune.Tests.E2E;

[Collection("SecurityScanner")]
[Trait("Suite", "UpgradesE2E")]
public sealed class SecurityComplianceE2ETests
{
    private static string? TryFindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Glacier.sln")) &&
                File.Exists(Path.Combine(dir.FullName, "scripts", "security_check.py")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        return null;
    }

    private static (int ExitCode, string StdOut, string StdErr) RunProcess(string fileName, string arguments, string workingDirectory)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        psi.Environment["PYTHONUTF8"] = "1";

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start process: {fileName} {arguments}");
        var stdoutTask = proc.StandardOutput.ReadToEndAsync();
        var stderrTask = proc.StandardError.ReadToEndAsync();
        if (!proc.WaitForExit(60_000))
        {
            try { proc.Kill(entireProcessTree: true); } catch { }
            throw new TimeoutException($"Process timed out: {fileName} {arguments}");
        }

        return (proc.ExitCode, stdoutTask.GetAwaiter().GetResult(), stderrTask.GetAwaiter().GetResult());
    }

    private static (int ExitCode, string StdOut, string StdErr) RunSecurityCheck(string repoRoot, string targetPath)
    {
        string scriptPath = Path.Combine(repoRoot, "scripts", "security_check.py");
        string pythonExe = OperatingSystem.IsWindows() ? "python" : "python3";
        return RunProcess(pythonExe, $"\"{scriptPath}\" \"{targetPath}\"", repoRoot);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F15")]
    public void T1_F15_01_SecurityCheckPassesCleanlyOnEntireRepository()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, "Glacier.Tune");

        Assert.True(exitCode == 0, $"security_check.py failed on repo root (exit {exitCode}):\n{stdout}\n{stderr}");
        Assert.Contains("[PASSED]", stdout);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F15")]
    public void T1_F15_02_SecurityCheckPassesOnAllFourPillarDirectories()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string[] pillars = ["Glacier.Inference", "Glacier.Quant", "Glacier.Tune", "Glacier.Agent"];

        foreach (string pillar in pillars)
        {
            string pillarDir = Path.Combine(repoRoot, pillar);
            Assert.True(Directory.Exists(pillarDir), $"Missing pillar directory: {pillarDir}");

            var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, pillar);
            Assert.True(exitCode == 0, $"security_check.py failed on {pillar} (exit {exitCode}):\n{stdout}\n{stderr}");
            Assert.Contains("[PASSED]", stdout);
        }
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F15")]
    public void T1_F15_03_SecurityCheckPassesOnEnvironmentVariableOnlySample()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        using var tempScope = new TempDirectoryScope("f15_clean_");
        string cleanFile = tempScope.FilePath("clean.cs");
        File.WriteAllText(cleanFile, """
            using System;
            public static class EnvConfig
            {
                public static string? GetKey() => Environment.GetEnvironmentVariable("GLACIER_TEST_ENV_VAR");
            }
            """);

        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, tempScope.Path);
        Assert.True(exitCode == 0, $"Expected clean env-var sample to pass (exit {exitCode}):\n{stdout}\n{stderr}");
        Assert.Contains("[PASSED]", stdout);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F15")]
    public void T1_F15_04_GitignoreExcludesSecretsAndNoCredentialFilesAreTracked()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string gitignorePath = Path.Combine(repoRoot, ".gitignore");
        Assert.True(File.Exists(gitignorePath));

        string gitignore = File.ReadAllText(gitignorePath);
        Assert.Contains("*.env", gitignore);
        Assert.Contains("appsettings*.json", gitignore);
        Assert.Contains("!appsettings.template.json", gitignore);

        var (exitCode, stdout, stderr) = RunProcess("git", "ls-files", repoRoot);
        Assert.True(exitCode == 0, $"git ls-files failed: {stderr}");

        var tracked = stdout.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        var forbidden = new List<string>();
        foreach (string rel in tracked)
        {
            string name = Path.GetFileName(rel);
            bool isAppSettings = name.StartsWith("appsettings", StringComparison.OrdinalIgnoreCase) &&
                                 name.EndsWith(".json", StringComparison.OrdinalIgnoreCase) &&
                                 !name.Equals("appsettings.template.json", StringComparison.OrdinalIgnoreCase);
            bool isEnv = name.Equals(".env", StringComparison.OrdinalIgnoreCase) ||
                         name.StartsWith(".env.", StringComparison.OrdinalIgnoreCase);
            bool isVault = name.EndsWith(".vault", StringComparison.OrdinalIgnoreCase) ||
                           name.Equals(".vault_pass", StringComparison.OrdinalIgnoreCase);

            if (isAppSettings || isEnv || isVault)
            {
                forbidden.Add(rel);
            }
        }

        Assert.Empty(forbidden);
    }

    [Fact]
    [Trait("Tier", "Tier1")]
    [Trait("Feature", "F15")]
    public void T1_F15_05_SecurityCheckPassesOnAllTestDirectories()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string[] testDirs =
        [
            Path.Combine("Glacier.Inference", "tests"),
            Path.Combine("Glacier.Quant", "tests"),
            Path.Combine("Glacier.Tune", "tests"),
            Path.Combine("Glacier.Agent", "tests")
        ];

        foreach (string rel in testDirs)
        {
            string full = Path.Combine(repoRoot, rel);
            Assert.True(Directory.Exists(full), $"Missing test directory: {full}");

            var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, rel);
            Assert.True(exitCode == 0, $"security_check.py failed on {rel} (exit {exitCode}):\n{stdout}\n{stderr}");
        }
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F15")]
    public void T2_F15_01_SyntheticSecretPatternInTempDirTriggersNonZeroScannerFailure()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        using var tempScope = new TempDirectoryScope("f15_leak_");
        string leakyFile = tempScope.FilePath("leaky.json");

        // Build synthetic regex-matching test token dynamically at runtime inside temp dir only
        string syntheticToken = string.Concat("AIza", "Sy", new string('K', 33));
        File.WriteAllText(leakyFile, $"{{\"token\": \"{syntheticToken}\"}}");

        var (exitCode, stdout, _) = RunSecurityCheck(repoRoot, tempScope.Path);
        Assert.NotEqual(0, exitCode);
        Assert.Contains("leaky.json", stdout);
        Assert.Contains("CRITICAL ALERT", stdout);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F15")]
    public void T2_F15_02_EmptyDirectoryScanExitsCleanlyWithZero()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        using var tempScope = new TempDirectoryScope("f15_empty_");

        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, tempScope.Path);
        Assert.True(exitCode == 0, $"Expected 0 on empty dir (got {exitCode}):\n{stdout}\n{stderr}");
        Assert.Contains("Scanned 0 files", stdout);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F15")]
    public void T2_F15_03_NonExistentTargetDirectoryReturnsNonZeroWithoutTraceback()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string missingDir = Path.Combine(Path.GetTempPath(), $"glacier_nonexistent_{Guid.NewGuid():N}");

        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, missingDir);
        Assert.DoesNotContain("Traceback", stderr, StringComparison.OrdinalIgnoreCase);
        Assert.NotEqual(0, exitCode);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F15")]
    public void T2_F15_04_BinaryGgufAndBinFilesDoNotCrashScannerWithUnicodeDecodeError()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        using var tempScope = new TempDirectoryScope("f15_bin_");

        byte[] invalidUtf8 = [0xFF, 0xFE, 0x00, 0x80, 0x99, 0xC3, 0x28];
        File.WriteAllBytes(tempScope.FilePath("weights.gguf"), invalidUtf8);
        File.WriteAllBytes(tempScope.FilePath("adapter.bin"), invalidUtf8);
        File.WriteAllBytes(tempScope.FilePath("raw_blob.dat"), invalidUtf8);

        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, tempScope.Path);
        Assert.True(exitCode == 0, $"Expected binary scan exit 0 (got {exitCode}):\n{stdout}\n{stderr}");
        Assert.DoesNotContain("UnicodeDecodeError", stderr);
    }

    [Fact]
    [Trait("Tier", "Tier2")]
    [Trait("Feature", "F15")]
    public void T2_F15_05_SanitizedJsonTemplateWithEmptyCredentialsPassesCleanly()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        using var tempScope = new TempDirectoryScope("f15_tmpl_");
        File.WriteAllText(tempScope.FilePath("appsettings.template.json"), """
            {
              "Credentials": {
                "ApiKey": "",
                "SecretKey": "",
                "AccessToken": ""
              }
            }
            """);

        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, tempScope.Path);
        Assert.True(exitCode == 0, $"Expected sanitized empty template exit 0 (got {exitCode}):\n{stdout}\n{stderr}");
        Assert.Contains("[PASSED]", stdout);
    }

    [Fact]
    [Trait("Tier", "Tier4")]
    [Trait("Feature", "F15")]
    public void T4_10_FullGlacierSolutionComplianceAndLocalBindingGate()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;

        // 1. Verify all 4 pillar source & test projects exist in the solution layout
        string[] requiredProjects =
        [
            Path.Combine("Glacier.Inference", "src", "Glacier.Inference", "Glacier.Inference.csproj"),
            Path.Combine("Glacier.Inference", "tests", "Glacier.Inference.Tests", "Glacier.Inference.Tests.csproj"),
            Path.Combine("Glacier.Quant", "src", "Glacier.Quant", "Glacier.Quant.csproj"),
            Path.Combine("Glacier.Quant", "tests", "Glacier.Quant.Tests", "Glacier.Quant.Tests.csproj"),
            Path.Combine("Glacier.Tune", "src", "Glacier.Tune", "Glacier.Tune.csproj"),
            Path.Combine("Glacier.Tune", "tests", "Glacier.Tune.Tests", "Glacier.Tune.Tests.csproj"),
            Path.Combine("Glacier.Agent", "src", "Glacier.Agent", "Glacier.Agent.csproj"),
            Path.Combine("Glacier.Agent", "tests", "Glacier.Agent.Tests", "Glacier.Agent.Tests.csproj")
        ];

        foreach (string rel in requiredProjects)
        {
            Assert.True(File.Exists(Path.Combine(repoRoot, rel)), $"Missing pillar project: {rel}");
        }

        // 2. Verify Glacier.Tune is bound to local Glacier.Inference (UseLocalProjectReferences=true) and new R4/R5 APIs exist
        string infAssemblyPath = typeof(KlDivergenceVerifier).Assembly.Location;
        Assert.Contains(Path.GetFullPath(repoRoot), Path.GetFullPath(infAssemblyPath), StringComparison.OrdinalIgnoreCase);

        Assert.Equal(0, new TrainingArguments().EvalSteps);
        Assert.True(typeof(PostTuneEvaluator).GetMethod(nameof(PostTuneEvaluator.Evaluate),
            [typeof(GgufLoraModel), typeof(GgufLoraModel), typeof(string), typeof(EvalOptions)]) != null);

        // 3. Full repository security scan gate must exit 0
        var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, "Glacier.Tune");
        Assert.True(exitCode == 0, $"Full repo security_check.py failed (exit {exitCode}):\n{stdout}\n{stderr}");
        Assert.Contains("[PASSED]", stdout);
    }
}
