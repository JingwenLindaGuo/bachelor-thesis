# Reddit Filtering Pipelines

## Acknowledgements

Parts of the Reddit dump streaming infrastructure used in this project were adapted from the *Arctic Shift* repository by Arthur Heitmann.

ArthurHeitmann. *arctic_shift*. GitHub repository. Available at: https://github.com/ArthurHeitmann/arctic_shift

In particular, this project reuses and adapts the general approach for processing compressed Reddit `.zst` files, while the project-specific filtering logic, text cleaning, diagnosis detection, and dataset construction logic were developed separately for this work.

## Data Source

All Reddit data used in this project were obtained from the public archive links provided through the Arctic Shift repository by Arthur Heitmann. In particular, the monthly Reddit dump files used in this project were downloaded via the links listed in `download_links.md`:

https://github.com/ArthurHeitmann/arctic_shift/blob/master/download_links.md

The present project uses these data for project-specific filtering, text preprocessing, and downstream NLP analysis.
---


## Overview

This project includes two related Python pipelines for processing Reddit submission dumps:

1. `diagnosis_pipeline.py`
   - identifies posts containing explicit self-disclosures of a current clinical diagnosis of depression;
   - optionally applies LLM verification;
   - optionally builds user-specific time windows and collects posts within those windows.

2. `diagnosis_pipeline_control.py`
   - constructs a control dataset by selecting candidate authors from long self-posts;
   - excludes authors who post in mental-health-related subreddits or whose text contains mental-health terminology or diagnosis signals;
   - saves the final cleaned control dataset and author-level acceptance or rejection metadata.

These two scripts serve different roles in the overall dataset construction workflow and therefore do **not** currently expose identical command-line interfaces. The diagnosis pipeline is CLI-driven, while the current control pipeline relies on script-level configuration near the top of the file. This README and the accompanying PowerShell helper script are designed to work with both. fileciteturn8file0turn8file1

---

## Required files

The main project files are:

- `diagnosis_pipeline.py`
- `diagnosis_pipeline_control.py`
- `fileStreams.py`
- `utils.py`
- `requirements.txt`
- `build_and_run.ps1`
- `README.md`

The helper modules `fileStreams.py` and `utils.py` must be available in the same folder as the pipeline scripts, or otherwise be importable from the active Python environment. Both pipeline scripts import `getFileJsonStream` and `FileProgressLog` from those modules. fileciteturn8file0turn8file1

---

## Requirements

- Python **3.10 or higher**
- Reddit submission files in `.zst` format
- Dependencies listed in `requirements.txt`

Optional:

- an OpenAI API key only if you plan to run `diagnosis_pipeline.py` with `--use-llm`

The diagnosis pipeline imports `openai` only when LLM verification is enabled, while the control pipeline does not use the OpenAI API. fileciteturn8file0turn8file1

---

## Installation

### 1. Create a virtual environment

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Optional OpenAI API setup

Only the diagnosis pipeline may require an API key, and only when `--use-llm` is enabled. The control pipeline does not require OpenAI credentials. fileciteturn8file0turn8file1

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

## Input data

Both pipelines expect Reddit submission data in `.zst` format.

Example folder structure:

```text
submissions/
  RS_2025-01.zst
  RS_2025-02.zst
  RS_2025-03.zst
```

---

## Pipeline A: diagnosis pipeline

### Script

```text
diagnosis_pipeline.py
```

### Interface style

This script uses an `argparse` command-line interface. It accepts parameters such as:

- `--input-path`
- `--output-dir`
- `--recursive`
- `--max-hits`
- `--use-llm`
- `--llm-model`
- `--collect-window-posts`
- `--days-back`
- `--min-words`
- `--group-posts-by-user` fileciteturn8file0

### Basic usage

```bash
python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs
```

### Recursive diagnosis collection

```bash
python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs --recursive
```

### With LLM verification

```bash
python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs --use-llm
```

### With window-based post collection

```bash
python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs --collect-window-posts
```

### Example on Windows PowerShell

```powershell
python diagnosis_pipeline.py --input-path "C:\Users\guoji\Downloads\reddit\submissions" --output-dir dep_outputs --recursive --collect-window-posts --min-words 500 --days-back 183
```

### Main outputs

Depending on the options used, this script may produce:

- `diagnosis_hits.json`
- `user_windows.json`
- `posts_in_windows.json`
- `posts_by_user.json` fileciteturn8file0

### Notes on text representation

This script cleans Reddit text before filtering and saves the cleaned version in the `text` field. In the window-collection stage, the minimum word-count threshold is still computed from the raw text before cleaning. fileciteturn8file0

---

## Pipeline B: control pipeline

### Script

```text
diagnosis_pipeline_control.py
```

### Interface style

The current version of this script is **configuration-driven rather than CLI-driven**. It uses hard-coded variables near the top of the file, including:

- `fileOrFolderPath`
- `recursive`
- `MIN_WORDS`
- `TARGET_AUTHORS`
- `OUTPUT_PATH`
- `MAX_CANDIDATE_AUTHORS` fileciteturn8file1

Before running the control pipeline, edit those values directly in the script if needed.

### Current workflow

The control pipeline performs three passes:

1. discover candidate authors from long self-posts;
2. inspect their full posting history for mental-health-related subreddits or terminology;
3. build the final cleaned control dataset from accepted authors. fileciteturn8file1

### Basic usage

```bash
python diagnosis_pipeline_control.py
```

### Example on Windows PowerShell

```powershell
python diagnosis_pipeline_control.py
```

### Main outputs

This script writes:

- `normal.json`
- `accepted_normal_authors.json`
- `rejected_normal_authors.json` fileciteturn8file1

### Notes on text representation

The final `normal.json` file contains cleaned text in the `text` field. During dataset construction, raw text is cleaned with `clean_social_media_text_for_structure(...)`, and the cleaned word count is stored in `word_count`. fileciteturn8file1

---

## Using the PowerShell helper script

The included `build_and_run.ps1` script supports both pipelines.

### Diagnosis mode

```powershell
.\build_and_run.ps1 -Mode diagnosis -InputPath "C:\Users\guoji\Downloads\reddit\submissions" -OutputDir dep_outputs -Recursive -CollectWindowPosts
```

### Diagnosis mode with LLM verification

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
.\build_and_run.ps1 -Mode diagnosis -InputPath "C:\Users\guoji\Downloads\reddit\submissions" -OutputDir dep_outputs -UseLLM
```

### Control mode

```powershell
.\build_and_run.ps1 -Mode control
```

In control mode, the helper script simply runs `diagnosis_pipeline_control.py` after environment setup. Because the current control pipeline is configuration-driven, make sure its internal configuration is correct before execution. fileciteturn8file1

--- 

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'fileStreams'`
Ensure that `fileStreams.py` is in the same folder as the pipeline scripts, or otherwise available in the Python path.

### Error: `OPENAI_API_KEY is not set`
Only relevant to `diagnosis_pipeline.py` when `--use-llm` is enabled. Either set the environment variable or run without `--use-llm`. fileciteturn8file0

### Error: `Input path does not exist`
Check that the diagnosis pipeline `--input-path` is correct, or ensure that the hard-coded `fileOrFolderPath` in the control pipeline points to a valid folder. fileciteturn8file0turn8file1

### Error: `No .zst files found`
Check that the target folder actually contains Reddit submission files ending in `.zst`. The diagnosis pipeline checks this via `iter_input_files(...)`, while the control pipeline expects the configured folder to contain `.zst` files discovered by `get_zst_files(...)`. fileciteturn8file0turn8file1
