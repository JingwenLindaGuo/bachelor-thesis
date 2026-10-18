param(
    [ValidateSet("diagnosis", "control")]
    [string]$Mode = "diagnosis",

    [string]$PythonExe = "python",

    [string]$InputPath = "",
    [string]$OutputDir = "diagnosis_pipeline_outputs",
    [switch]$Recursive,
    [switch]$UseLLM,
    [switch]$CollectWindowPosts,
    [switch]$GroupPostsByUser,
    [int]$DaysBack = 183,
    [int]$MinWords = 500,
    [string]$LLMModel = "gpt-4o-mini",
    [Nullable[int]]$MaxHits = $null,

    [string[]]$ExtraArgs
)

Write-Host "============================================================"
Write-Host "Building and running Reddit pipeline"
Write-Host "Mode: $Mode"
Write-Host "============================================================"

function Fail-And-Exit {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

if (!(Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment..."
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Fail-And-Exit "Failed to create virtual environment."
    }
} else {
    Write-Host "[1/4] Virtual environment already exists."
}

Write-Host "[2/4] Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Fail-And-Exit "Failed to install dependencies."
}

Write-Host "[4/4] Running selected pipeline..."

if ($Mode -eq "diagnosis") {
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        Fail-And-Exit "Diagnosis mode requires -InputPath."
    }

    $cmd = @(
        "diagnosis_pipeline.py",
        "--input-path", $InputPath,
        "--output-dir", $OutputDir,
        "--days-back", "$DaysBack",
        "--min-words", "$MinWords",
        "--llm-model", $LLMModel
    )

    if ($Recursive) {
        $cmd += "--recursive"
    }

    if ($UseLLM) {
        if (-not $env:OPENAI_API_KEY) {
            Fail-And-Exit "OPENAI_API_KEY is not set, but -UseLLM was requested."
        }
        $cmd += "--use-llm"
    }

    if ($CollectWindowPosts) {
        $cmd += "--collect-window-posts"
    }

    if ($GroupPostsByUser) {
        $cmd += "--group-posts-by-user"
    }

    if ($null -ne $MaxHits) {
        $cmd += @("--max-hits", "$MaxHits")
    }

    if ($ExtraArgs) {
        $cmd += $ExtraArgs
    }

    Write-Host "Running command:"
    Write-Host ("python " + ($cmd -join " "))
    python @cmd
}
elseif ($Mode -eq "control") {
    $cmd = @("diagnosis_pipeline_control.py")

    if ($ExtraArgs) {
        $cmd += $ExtraArgs
    }

    Write-Host "Running command:"
    Write-Host ("python " + ($cmd -join " "))
    Write-Host "Note: the current control pipeline is configuration-driven."
    Write-Host "Please ensure the settings at the top of diagnosis_pipeline_control.py are correct before running."

    python @cmd
}

if ($LASTEXITCODE -ne 0) {
    Fail-And-Exit "Pipeline execution failed."
}

Write-Host "============================================================"
Write-Host "Pipeline completed successfully."
Write-Host "============================================================"
