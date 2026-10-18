# Generic JSON Filtering Pipelines for Depressed and Control Instances

## Overview

This package contains two command-line Python scripts for filtering generic JSON text records:

- `processFiles_depression.py`: identifies candidate **depressed** instances using a rule-based diagnosis pre-filter and optional LLM verification.
- `processFiles_control.py`: identifies candidate **control** instances by excluding records or entities with mental-health-related signals.

Both scripts expect JSON input in one of two formats:

- a list of JSON objects
- a single JSON object

Both scripts preserve existing metadata fields and add task-specific output fields.

---

## Scripts

### 1. `processFiles_depression.py`

This script:

1. reads generic JSON records
2. extracts text from a specified text field
3. cleans the text for diagnosis detection
4. applies a regex-based diagnosis pre-filter
5. optionally applies LLM verification
6. writes retained depressed instances to JSON

Main command-line arguments:

- `--input`: input JSON path
- `--output`: output JSON path
- `--text-field`: name of the text field in each record
- `--min-words`: minimum cleaned word count
- `--no-llm`: disable LLM verification and use regex only
- `--model`: OpenAI model name for LLM verification

If LLM verification is enabled, the script requires the `OPENAI_API_KEY` environment variable.

### 2. `processFiles_control.py`

This script:

1. reads generic JSON records
2. extracts text from a specified text field
3. detects mental-health-related signals
4. excludes unsuitable records or entities
5. cleans retained text for downstream analysis
6. writes retained control instances to JSON

Main command-line arguments:

- `--input`: input JSON path
- `--output`: output JSON path
- `--text-field`: name of the text field in each record
- `--entity-field`: optional entity field for history-level exclusion
- `--min-words`: minimum cleaned word count
- `--no-entity-screening`: disable entity-level exclusion
- `--accepted-output`: optional path for accepted entity summary JSON
- `--rejected-output`: optional path for rejected entity summary JSON

---

## Requirements

- Python **3.10 or higher**
- packages listed in `requirements_processfiles.txt`

Optional:

- `OPENAI_API_KEY` if running `processFiles_depression.py` with LLM verification enabled

---

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_processfiles.txt
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements_processfiles.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_processfiles.txt
```

---

## Optional OpenAI API Setup

Only needed for `processFiles_depression.py` when LLM verification is enabled.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Windows Command Prompt

```cmd
set OPENAI_API_KEY=your_api_key_here
```

### macOS / Linux

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## Manual Usage

### Run the depressed-instance pipeline with LLM verification

```bash
python processFiles_depression.py --input input.json --output depressed_output.json --text-field text --min-words 500
```

### Run the depressed-instance pipeline with regex only

```bash
python processFiles_depression.py --input input.json --output depressed_output.json --text-field text --min-words 500 --no-llm
```

### Run the control-instance pipeline with entity-level screening

```bash
python processFiles_control.py --input input.json --output control_output.json --text-field text --entity-field author --min-words 500 --accepted-output accepted_entities.json --rejected-output rejected_entities.json
```

### Run the control-instance pipeline without entity-level screening

```bash
python processFiles_control.py --input input.json --output control_output.json --text-field text --min-words 500 --no-entity-screening
```

---

## Automated Build-and-Run Script

A PowerShell script named `build_and_run_processfiles.ps1` is included.

It can:

1. create a virtual environment if needed
2. activate the virtual environment
3. install required dependencies
4. run either the depressed-instance or control-instance pipeline

### Example: depressed-instance mode

```powershell
.\build_and_run_processfiles.ps1 -Mode depression -Input input.json -Output depressed_output.json -TextField text -MinWords 500
```

### Example: control-instance mode

```powershell
.\build_and_run_processfiles.ps1 -Mode control -Input input.json -Output control_output.json -TextField text -EntityField author -MinWords 500 -AcceptedOutput accepted_entities.json -RejectedOutput rejected_entities.json
```

---

## Output Structure

### Depressed-instance output

Each retained record preserves existing metadata and adds:

- `cleaned_text`
- `label` = `"depressed"`
- `evidence`
- `detection_method`

### Control-instance output

Each retained record preserves existing metadata and adds:

- `text` (cleaned text replaces the original text field content)
- `label` = `"control"`
- `filter_method` = `"lexical_exclusion"`

If requested, the control pipeline also writes:

- `accepted_entities.json`
- `rejected_entities.json`

---

## Notes

- The depressed pipeline uses cleaned text internally for diagnosis detection, but retains both the original text and the cleaned text in the output.
- The control pipeline replaces the output `text` field with cleaned text.
- Both scripts are designed for generic JSON records rather than Reddit-specific schemas.

---

## Troubleshooting

### `OPENAI_API_KEY is not set`

Set the environment variable before running the depressed pipeline with LLM verification, or add `--no-llm`.

### `Input JSON must be a list of objects`

Ensure the input file is either:

- a JSON list of objects, or
- a single JSON object

### `This script requires Python 3.10 or higher`

Install and run the scripts with Python 3.10+.
