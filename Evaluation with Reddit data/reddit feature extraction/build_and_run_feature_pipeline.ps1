param(
    [Parameter(Mandatory=$true)]
    [string]$Input,

    [Parameter(Mandatory=$true)]
    [string]$Output,

    [string]$TextField = "text",
    [int]$SaveEvery = 5,
    [switch]$Quiet,
    [string]$PythonExe = "python"
)

Write-Host "============================================================"
Write-Host "Building and running Generic JSON-Based NLP Feature Pipeline"
Write-Host "============================================================"

# Step 1: create virtual environment if it does not exist
if (!(Test-Path ".venv")) {
    Write-Host "[1/5] Creating virtual environment..."
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "[1/5] Virtual environment already exists."
}

# Step 2: activate virtual environment
Write-Host "[2/5] Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

# Step 3: install dependencies
Write-Host "[3/5] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements_feature_pipeline.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit 1
}

# Step 4: download required NLP resources
Write-Host "[4/5] Downloading required NLP resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
python -c "import benepar; benepar.download('benepar_en3')"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "One or more resource download commands reported an error. The pipeline may still run if the resources are already available."
}

# Step 5: run the pipeline
Write-Host "[5/5] Running pipeline..."

$cmd = @(
    "pipeline.py",
    "--input", $Input,
    "--output", $Output,
    "--text-field", $TextField,
    "--save-every", "$SaveEvery"
)

if ($Quiet) {
    $cmd += "--quiet"
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
