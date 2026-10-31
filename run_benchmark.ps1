# CST435 reproducible benchmark runner for Windows PowerShell.
# Creates .venv, installs benchmark dependencies, pre-caches MNIST,
# runs the exact benchmark suite, and regenerates both figures.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Write-Host "CST435 MNIST scalability benchmark"
Write-Host "Repository: $RepoRoot"
Write-Host ""

# 1. Locate Python 3.12.
$PythonExe = $null
$PythonArgs = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = "py"
        $PythonArgs = @("-3.12")
    }
}

if (-not $PythonExe) {
    foreach ($Candidate in @("python3.12", "python")) {
        if (Get-Command $Candidate -ErrorAction SilentlyContinue) {
            & $Candidate -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonExe = $Candidate
                break
            }
        }
    }
}

if (-not $PythonExe) {
    throw "Python 3.12 was not found. Install it from https://www.python.org/downloads/ and rerun this script."
}

Write-Host "[1/5] Python 3.12 found: $(& $PythonExe @PythonArgs --version)"

# 2. Create or validate the virtual environment.
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The existing .venv is not Python 3.12. Delete .venv and rerun this script."
    }
    Write-Host "[2/5] Using existing .venv"
}
else {
    Write-Host "[2/5] Creating .venv"
    & $PythonExe @PythonArgs -m venv $VenvDir
    Assert-LastExitCode "Virtual environment creation"
}

# 3. Install benchmark dependencies.
$Requirements = Join-Path $RepoRoot "requirements-benchmark.txt"
Write-Host "[3/5] Installing benchmark dependencies"
& $VenvPython -m pip install --disable-pip-version-check --requirement $Requirements
Assert-LastExitCode "Dependency installation"

# 4. Pre-cache MNIST so download time is outside the timed runs.
Write-Host "[4/5] Pre-caching MNIST"
& $VenvPython -c "from src.dataset import get_mnist_datasets; train, test = get_mnist_datasets(); print(f'MNIST ready: {len(train)} train / {len(test)} test')"
Assert-LastExitCode "MNIST pre-cache"

# 5. Run the exact benchmark and regenerate the figures.
Write-Host "[5/5] Running benchmark (CNN, 1 epoch, batch 128, p=1 2 4 8)"
& $VenvPython (Join-Path $RepoRoot "src\benchmark.py") --model cnn --epochs 1 --batch-size 128 --process-list 1 2 4 8
Assert-LastExitCode "Benchmark suite"

Write-Host "Regenerating figures"
& $VenvPython (Join-Path $RepoRoot "src\plot_scalability.py")
Assert-LastExitCode "Figure generation"

$Artifacts = @(
    (Join-Path $RepoRoot "docs\benchmark_results.md"),
    (Join-Path $RepoRoot "figures\speedup_curve.png"),
    (Join-Path $RepoRoot "figures\scalability_visualization.png")
)

foreach ($Artifact in $Artifacts) {
    if (-not (Test-Path -LiteralPath $Artifact)) {
        throw "Missing artifact: $Artifact"
    }
    if ((Get-Item -LiteralPath $Artifact).Length -le 0) {
        throw "Empty artifact: $Artifact"
    }
}

$Stopwatch.Stop()

Write-Host ""
Write-Host "Benchmark complete."
try {
    $Cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    Write-Host "CPU: $($Cpu.Name) ($($Cpu.NumberOfCores) cores / $($Cpu.NumberOfLogicalProcessors) logical processors)"
}
catch {
    Write-Host "CPU: (unavailable)"
}
Write-Host "Python: $(& $VenvPython --version)"
Write-Host "Elapsed: $([math]::Round($Stopwatch.Elapsed.TotalMinutes, 2)) minutes"
Write-Host "Results: $($Artifacts[0])"
Write-Host "Figures: $($Artifacts[1])"
Write-Host "         $($Artifacts[2])"
