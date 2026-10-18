param(
    [ValidateSet("depression", "control")]
    [string]$Mode = "control",

    [Parameter(Mandatory = $true)]
    [string]$Input,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$TextField = "text",
    [int]$MinWords = 500,
    [string]$PythonExe = "python",

    # Depression-mode options
    [switch]$NoLLM,
    [string]$Model = "gpt-4o-mini",

    # Control-mode options
    [string]$EntityField,
    [switch]$NoEntityScreening,
    [string]$AcceptedOutput,
    [string]$RejectedOutput
)

Write-Host "============================================================"
Write-Host "Building and running JSON filtering pipeline"
Write-Host "Mode: $Mode"
Write-Host "============================================================"

if (!(Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment..."
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "[1/4] Virtual environment already exists."
}

Write-Host "[2/4] Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements_processfiles.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit 1
}

Write-Host "[4/4] Running pipeline..."

if ($Mode -eq "depression") {
    $scriptName = "processFiles_depression.py"
    $cmd = @(
        $scriptName,
        "--input", $Input,
        "--output", $Output,
        "--text-field", $TextField,
        "--min-words", "$MinWords",
        "--model", $Model
    )

    if ($NoLLM) {
        $cmd += "--no-llm"
    } else {
        if (-not $env:OPENAI_API_KEY) {
            Write-Error "OPENAI_API_KEY is not set, but depression mode is running with LLM verification enabled."
            exit 1
        }
    }
}
else {
    $scriptName = "processFiles_control.py"
    $cmd = @(
        $scriptName,
        "--input", $Input,
        "--output", $Output,
        "--text-field", $TextField,
        "--min-words", "$MinWords"
    )

    if ($EntityField) {
        $cmd += @("--entity-field", $EntityField)
    }

    if ($NoEntityScreening) {
        $cmd += "--no-entity-screening"
    }

    if ($AcceptedOutput) {
        $cmd += @("--accepted-output", $AcceptedOutput)
    }

    if ($RejectedOutput) {
        $cmd += @("--rejected-output", $RejectedOutput)
    }
}

Write-Host "Running command:"
Write-Host ("python " + ($cmd -join " "))

python @cmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline execution failed."
    exit 1
}

Write-Host "============================================================"
Write-Host "Pipeline completed successfully."
Write-Host "============================================================"
