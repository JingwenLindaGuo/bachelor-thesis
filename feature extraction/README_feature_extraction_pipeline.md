# Generic JSON-Based NLP Feature Extraction Pipeline

## Overview

This tool reads JSON records, extracts a predefined set of NLP features from a specified text field, and writes the successful records back to JSON with an additional `features` dictionary.

The current script is:

- `feature_extraction_pipeline.py`

The pipeline supports:

- a list of JSON objects
- a single JSON object

If any required feature extractor fails for a record, that entire record is skipped. Only successful records are written to the output.

## Expected project files

The following files should be available in the same project directory, or otherwise importable from the active Python environment:

- `feature_extraction_pipeline.py`
- `lexical_features.py`
- `local_coh.py`
- `speechgraph_feature.py`
- `ambiguous_pronoun.py`
- `basic_meas.py`
- `coh_rep.py`

## Python version

- Python 3.10 or higher

## Installation

### 1. Create and activate a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements_feature_extraction_pipeline.txt
```

## External resource setup

Before running the pipeline, ensure the required NLP resources are available.

```bash
python -c "import benepar; benepar.download('benepar_en3')"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

If your environment does not already contain the `speechgraph` package, install it with:

```bash
git clone https://github.com/facuzeta/speechgraph.git
pip install git+https://github.com/facuzeta/speechgraph/
```

## Input format

The script expects JSON data in one of the following forms:

### Option 1: list of records

```json
[
  {
    "author": "user_1",
    "text": "This is a sample text. It has multiple sentences."
  },
  {
    "author": "user_2",
    "text": "Another text example."
  }
]
```

### Option 2: single record

```json
{
  "author": "user_1",
  "text": "This is a sample text. It has multiple sentences."
}
```

## Output format

The output preserves the original record fields and appends a `features` dictionary.

Example:

```json
{
  "author": "user_1",
  "text": "This is a sample text. It has multiple sentences.",
  "features": {
    "word_count": 9,
    "sentence_count": 2,
    "semantic_coherence": 0.42
  }
}
```

## Manual usage

### Basic example

```bash
python feature_extraction_pipeline.py --input normal.json --output normal_with_features.json
```

### Depression group example

```bash
python feature_extraction_pipeline.py --input depression.json --output depression_with_features.json
```

### Use a different text field name

```bash
python feature_extraction_pipeline.py --input input.json --output output.json --text-field cleaned_text
```

### Save checkpoints every 10 records

```bash
python feature_extraction_pipeline.py --input input.json --output output.json --save-every 10
```

## Command-line arguments

- `--input`  
  Path to the input JSON file.

- `--output`  
  Path to the output JSON file.

- `--text-field`  
  Name of the text field in each record. Default: `text`

- `--save-every`  
  Save a checkpoint every N input records. Default: `5`

## Build-and-run script

A PowerShell helper script is provided:

- `build_and_run_feature_extraction_pipeline.ps1`

Example:

```powershell
.\build_and_run_feature_extraction_pipeline.ps1 -Input normal.json -Output normal_with_features.json
```

## Troubleshooting

### `ModuleNotFoundError`

Make sure the local feature modules are present in the same project directory or otherwise importable.

### `OPENAI_API_KEY` is not relevant here

This pipeline does not require an OpenAI API key.

### `benepar` / `nltk` resource errors

Run the resource download commands listed above before starting the pipeline.

### `speechgraph` import errors

Install the package explicitly using the command shown in the setup section.
