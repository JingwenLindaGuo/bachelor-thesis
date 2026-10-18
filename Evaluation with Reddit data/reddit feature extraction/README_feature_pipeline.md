# Generic JSON-Based NLP Feature Extraction Pipeline

## Overview

This project implements a Python-based command-line pipeline for extracting a predefined set of NLP features from JSON records. The script reads either a list of JSON objects or a single JSON object, extracts features from a specified text field, and writes the successful records back to JSON with an additional `features` dictionary.

The pipeline currently orchestrates the following feature families:

- basic measures
- semantic coherence and maximum similarity
- lexical features
- local coherence overlap
- speech graph features
- ambiguous pronoun count

The pipeline follows a fail-fast policy at the record level: if any required feature extractor fails for a given record, that entire record is skipped.

---

## Main files

The main file required to run the tool is:

- `pipeline.py` — feature extraction pipeline script

The following local Python modules must also be available in the same directory, or otherwise importable from the Python environment:

- `lexical_features.py`
- `local_coh.py`
- `speechgraph_feature.py`
- `ambiguous_pronoun.py`
- `basic_meas.py`
- `coh_rep.py`

---

## Requirements

### Core runtime

- Python **3.10 or higher**
- Input JSON file containing either:
  - a list of records, or
  - a single record

### Python packages

Install the packages listed in `requirements_feature_pipeline.txt`.

> Note: some downstream feature modules may themselves depend on additional NLP libraries or models. The setup commands below include the extra resources explicitly referenced in `pipeline.py`.

---

## Installation

### 1. Create a virtual environment

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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

---

### 2. Install Python dependencies

```bash
pip install -r requirements_feature_pipeline.txt
```

---

### 3. Download required NLP resources

The current pipeline notes explicitly mention the following setup steps:

```bash
python -c "import benepar; benepar.download('benepar_en3')"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

The speech graph dependency can be installed with:

```bash
git+https://github.com/facuzeta/speechgraph/
```

If your environment does not support direct `pip install` from a Git URL through the requirements file, install it manually with:

```bash
pip install git+https://github.com/facuzeta/speechgraph/
```

---

## Input format

The script accepts either:

1. a **list of JSON records**, for example:

```json
[
  {
    "id": "001",
    "text": "This is the first example record."
  },
  {
    "id": "002",
    "text": "This is the second example record."
  }
]
```

2. a **single JSON record**, for example:

```json
{
  "id": "001",
  "text": "This is a single example record."
}
```

By default, the pipeline expects the text field to be named `text`, but this can be changed using `--text-field`.

---

## Basic usage

### Run on a JSON file with a standard `text` field

```bash
python pipeline.py --input normal.json --output normal_with_features.json
```

### Run on another JSON file and checkpoint every 5 records

```bash
python pipeline.py --input depression.json --output depression_with_features.json --save-every 5
```

### Run on a JSON file whose text field is not named `text`

```bash
python pipeline.py --input records.json --output records_with_features.json --text-field body
```

### Reduce console logging

```bash
python pipeline.py --input normal.json --output normal_with_features.json --quiet
```

---

## Command-line arguments

- `--input`  
  Path to the input JSON file.

- `--output`  
  Path to the output JSON file.

- `--text-field`  
  Name of the field containing the input text. Default: `text`

- `--save-every`  
  Save an intermediate checkpoint every N input records. Default: `5`

- `--quiet`  
  Reduce logging output.

---

## Output format

The output preserves the original record structure and appends a `features` dictionary to each successfully processed record.

Example output record:

```json
{
  "id": "001",
  "text": "This is an example record.",
  "features": {
    "word_count": 5,
    "sentence_count": 1
  }
}
```

Only successful records are included in the output. Records that fail any required feature extractor are skipped.

---

## Checkpointing behaviour

If `--save-every` is greater than 0, the script periodically writes a checkpoint to the output path during execution. The final output is also written again at the end of the run.

The implementation uses atomic file replacement to reduce the risk of partial output corruption.

---

## Example project structure

```text
project_folder/
  pipeline.py
  lexical_features.py
  local_coh.py
  speechgraph_feature.py
  ambiguous_pronoun.py
  basic_meas.py
  coh_rep.py
  requirements_feature_pipeline.txt
  build_and_run_feature_pipeline.ps1
  README_feature_pipeline.md
```

---

## Troubleshooting

### `ModuleNotFoundError`
Ensure that all required local feature modules are present in the same folder as `pipeline.py`, or are available on the Python path.

### Missing NLP resources
If the script fails because of missing `nltk` or `benepar` resources, run the download commands shown in the installation section.

### Records are skipped
This pipeline is designed to skip any record for which one or more required feature extractors fail. Review the console logs to inspect failure reasons.

### Large model initialization delay
The ambiguous pronoun feature uses a coreference model and may take time to initialize on first use.

---

## Reproducibility note

For reproducible execution, use the same Python version, the same local feature modules, the same model resources, and the same input JSON records.
