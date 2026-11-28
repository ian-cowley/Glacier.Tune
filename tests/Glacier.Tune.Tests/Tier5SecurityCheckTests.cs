using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Xunit;

namespace Glacier.Tune.Tests;

[Collection("SecurityScanner")]
[Trait("Tier", "Tier5")]
public sealed class Tier5SecurityCheckTests
{
    private static string? TryFindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if ((File.Exists(Path.Combine(dir.FullName, "Glacier.sln")) || File.Exists(Path.Combine(dir.FullName, "Glacier.slnx"))) &&
                File.Exists(Path.Combine(dir.FullName, "scripts", "security_check.py")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return null;
    }

    private static (int ExitCode, string StdOut, string StdErr) RunSecurityCheck(string repoRoot, string targetPath)
    {
        string scriptPath = Path.Combine(repoRoot, "scripts", "security_check.py");
        string pythonExe = OperatingSystem.IsWindows() ? "python" : "python3";
        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"\"{scriptPath}\" \"{targetPath}\"",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        psi.Environment["PYTHONUTF8"] = "1";

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start python process.");
        var stdoutTask = proc.StandardOutput.ReadToEndAsync();
        var stderrTask = proc.StandardError.ReadToEndAsync();
        if (!proc.WaitForExit(60_000))
        {
            try { proc.Kill(entireProcessTree: true); } catch { }
            throw new TimeoutException("security_check.py timed out.");
        }
        return (proc.ExitCode, stdoutTask.GetAwaiter().GetResult(), stderrTask.GetAwaiter().GetResult());
    }

    private static (int ExitCode, string StdOut, string StdErr)? TryRunGlacierCliSecurityCheck(string repoRoot, string targetPath)
    {
        string preferredCfg = AppContext.BaseDirectory.Contains($"{Path.DirectorySeparatorChar}Release{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase)
            ? "Release"
            : "Debug";
        string? cliDll = null;
        foreach (string cfg in new[] { preferredCfg, preferredCfg == "Release" ? "Debug" : "Release" })
        {
            string candidate = Path.Combine(repoRoot, "Glacier.Cli", "src", "Glacier.Cli", "bin", cfg, "net10.0", "glacier.dll");
            if (File.Exists(candidate) && File.Exists(Path.ChangeExtension(candidate, ".runtimeconfig.json")))
            {
                cliDll = candidate;
                break;
            }
        }
        if (cliDll is null) return null;

        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        psi.ArgumentList.Add(cliDll);
        psi.ArgumentList.Add("security-check");
        psi.ArgumentList.Add(targetPath);

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start dotnet glacier.dll process.");
        var stdoutTask = proc.StandardOutput.ReadToEndAsync();
        var stderrTask = proc.StandardError.ReadToEndAsync();
        if (!proc.WaitForExit(60_000))
        {
            try { proc.Kill(entireProcessTree: true); } catch { }
            throw new TimeoutException("glacier security-check timed out.");
        }
        return (proc.ExitCode, stdoutTask.GetAwaiter().GetResult(), stderrTask.GetAwaiter().GetResult());
    }

    private static string CreateTempDir(string prefix)
    {
        string dir = Path.Combine(Path.GetTempPath(), prefix + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void Tier5_SecurityCheck_PrunesBenchmarkDotNetArtifactsCaseInsensitively()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string tempDir = CreateTempDir("glacier_t5_bdn_");
        try
        {
            string bdnDir = Path.Combine(tempDir, "BenchmarkDotNet.Artifacts");
            Directory.CreateDirectory(bdnDir);
            string syntheticToken = string.Concat("AIza", "Sy", new string('K', 33));
            File.WriteAllText(Path.Combine(bdnDir, "report.txt"), $"token={syntheticToken}\n");

            var (exitCode, stdout, stderr) = RunSecurityCheck(repoRoot, tempDir);
            Assert.True(exitCode == 0, $"Expected BenchmarkDotNet.Artifacts to be pruned (exit {exitCode}):\n{stdout}\n{stderr}");
            Assert.Contains("Scanned 0 files.", stdout);

            if (TryRunGlacierCliSecurityCheck(repoRoot, tempDir) is { } cli)
            {
                Assert.True(cli.ExitCode == 0, $"Expected glacier security-check to prune BenchmarkDotNet.Artifacts (exit {cli.ExitCode}):\n{cli.StdOut}\n{cli.StdErr}");
                Assert.Contains("Scanned 0 files.", cli.StdOut);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_SecurityCheck_DoesNotBypassLinesContainingSecretPatternsOrRegexSubstrings()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string tempDir = CreateTempDir("glacier_t5_bypass_");
        try
        {
            string skToken = string.Concat("sk-", new string('a', 24));
            string aizaToken = string.Concat("AIza", "Sy", new string('M', 33));
            string bypassContent =
                $"API_KEY = \"{skToken}\"  # SECRET_PATTERNS\n" +
                $"OTHER = \"AIza[0-9A-Za-z\" ; real_key = \"{aizaToken}\"\n";
            File.WriteAllText(Path.Combine(tempDir, "bypass.py"), bypassContent);

            var (exitCode, stdout, _) = RunSecurityCheck(repoRoot, tempDir);
            Assert.NotEqual(0, exitCode);
            Assert.Contains("bypass.py:1", stdout);
            Assert.Contains("bypass.py:2", stdout);

            if (TryRunGlacierCliSecurityCheck(repoRoot, tempDir) is { } cli)
            {
                Assert.NotEqual(0, cli.ExitCode);
                Assert.Contains("bypass.py:1", cli.StdOut);
                Assert.Contains("bypass.py:2", cli.StdOut);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Theory]
    [InlineData("utf16le")]
    [InlineData("utf16be")]
    public void Tier5_SecurityCheck_DetectsSecretsInUtf16EncodedFiles(string encodingKind)
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string tempDir = CreateTempDir("glacier_t5_utf16_");
        try
        {
            string aizaToken = string.Concat("AIza", "Sy", new string('Z', 33));
            Encoding enc = encodingKind == "utf16le" ? Encoding.Unicode : Encoding.BigEndianUnicode;
            File.WriteAllText(Path.Combine(tempDir, "win_config.ps1"), $"$KEY = \"{aizaToken}\"\n", enc);

            var (exitCode, stdout, _) = RunSecurityCheck(repoRoot, tempDir);
            Assert.NotEqual(0, exitCode);
            Assert.Contains("win_config.ps1", stdout);

            if (TryRunGlacierCliSecurityCheck(repoRoot, tempDir) is { } cli)
            {
                Assert.NotEqual(0, cli.ExitCode);
                Assert.Contains("win_config.ps1", cli.StdOut);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_SecurityCheck_DetectsUnquotedEnvAndYamlSecretAssignments()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string tempDir = CreateTempDir("glacier_t5_unquoted_");
        try
        {
            string dbPass = string.Concat("RealUnquoted", "ProdPassword98765");
            string apiVal = string.Concat("unquoted_prod_", "key_1234567890");
            File.WriteAllText(Path.Combine(tempDir, "prod.env"), $"DATABASE_PASSWORD={dbPass}\n");
            File.WriteAllText(Path.Combine(tempDir, "secrets.yaml"), $"api_key: {apiVal}\n");

            var (exitCode, stdout, _) = RunSecurityCheck(repoRoot, tempDir);
            Assert.NotEqual(0, exitCode);
            Assert.Contains("prod.env", stdout);
            Assert.Contains("secrets.yaml", stdout);

            if (TryRunGlacierCliSecurityCheck(repoRoot, tempDir) is { } cli)
            {
                Assert.NotEqual(0, cli.ExitCode);
                Assert.Contains("prod.env", cli.StdOut);
                Assert.Contains("secrets.yaml", cli.StdOut);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Tier5_SecurityCheck_DoesNotAllowlistProviderPrefixedTokensContainingGenericPlaceholderWords()
    {
        if (TryFindRepoRoot() is not { } repoRoot) return;
        string tempDir = CreateTempDir("glacier_t5_provider_");
        try
        {
            string skFake = string.Concat("sk-", "proj1234567890fake1234567890");
            string ghpExample = string.Concat("ghp_", "realProdSecretWithExample123456789012");
            string aizaFake = string.Concat("AIza", "SyB2Fake9Xy123456789012345678901234");
            string hfDummy = string.Concat("hf_", "dummyToken1234567890123456789012345678");

            File.WriteAllText(Path.Combine(tempDir, "provider_leaks.py"),
                $"OPENAI = \"{skFake}\"\n" +
                $"GITHUB = \"{ghpExample}\"\n" +
                $"GEMINI = \"{aizaFake}\"\n" +
                $"HF = \"{hfDummy}\"\n");

            var (exitCode, stdout, _) = RunSecurityCheck(repoRoot, tempDir);
            Assert.NotEqual(0, exitCode);
            Assert.Contains("provider_leaks.py:1", stdout);
            Assert.Contains("provider_leaks.py:2", stdout);
            Assert.Contains("provider_leaks.py:3", stdout);
            Assert.Contains("provider_leaks.py:4", stdout);

            if (TryRunGlacierCliSecurityCheck(repoRoot, tempDir) is { } cliFail)
            {
                Assert.NotEqual(0, cliFail.ExitCode);
                Assert.Contains("provider_leaks.py:1", cliFail.StdOut);
                Assert.Contains("provider_leaks.py:2", cliFail.StdOut);
                Assert.Contains("provider_leaks.py:3", cliFail.StdOut);
                Assert.Contains("provider_leaks.py:4", cliFail.StdOut);
            }

            string safeDir = CreateTempDir("glacier_t5_safe_");
            try
            {
                string aizaDummyFixture = string.Concat("AIza", "SyDummy-123456789012345678901234567");
                File.WriteAllText(Path.Combine(safeDir, "fixtures.cs"),
                    "var a = \"dummy_api_key_sample\";\n" +
                    "const string ApiKey = \"dummy_api_key_sample\";\n" +
                    "const string Placeholder = \"<your_api_key>\";\n" +
                    $"const string Explicit = \"{aizaDummyFixture}\";\n");
                var (safeExit, safeOut, safeErr) = RunSecurityCheck(repoRoot, safeDir);
                Assert.True(safeExit == 0, $"Expected explicit fixtures to stay allowlisted (exit {safeExit}):\n{safeOut}\n{safeErr}");

                if (TryRunGlacierCliSecurityCheck(repoRoot, safeDir) is { } cliSafe)
                {
                    Assert.True(cliSafe.ExitCode == 0, $"Expected glacier security-check fixtures allowlisted (exit {cliSafe.ExitCode}):\n{cliSafe.StdOut}\n{cliSafe.StdErr}");
                }
            }
            finally
            {
                Directory.Delete(safeDir, recursive: true);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
