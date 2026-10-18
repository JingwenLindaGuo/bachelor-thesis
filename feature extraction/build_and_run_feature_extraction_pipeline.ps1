param(
    [Parameter(Mandatory = $true)]
    [string]$Input,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$TextField = "text",
    [int]$SaveEvery = 5,
    [string]$ScriptPath = "feature_extraction_pipeline.py",
    [string]$PythonExe = "python"
)

Write-Host "============================================================"
Write-Host "Building and running NLP feature extraction pipeline"
Write-Host "============================================================"

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

Write-Host "[2/5] Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "[3/5] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements_feature_extraction_pipeline.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit 1
}

Write-Host "[4/5] Downloading required NLP resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download required NLTK resources."
    exit 1
}

python -c "import benepar; benepar.download('benepar_en3')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download benepar resource benepar_en3."
    exit 1
}

Write-Host "[5/5] Running feature extraction pipeline..."
python $ScriptPath --input $Input --output $Output --text-field $TextField --save-every $SaveEvery
if ($LASTEXITCODE -ne 0) {
    Write-Error "Feature extraction pipeline failed."
    exit 1
}

Write-Host "============================================================"
Write-Host "Pipeline completed successfully."
Write-Host "Output: $Output"
Write-Host "============================================================"
