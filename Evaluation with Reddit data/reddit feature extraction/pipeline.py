"""
Generic JSON-based NLP feature extraction pipeline.

This script reads JSON records, extracts a predefined set of NLP features from a
specified text field, and writes the successful records back to JSON with an
additional ``features`` dictionary.

Supported input formats
-----------------------
- a list of JSON objects
- a single JSON object

Feature families orchestrated by this script
-------------------------------------------
- basic measures
- semantic coherence and maximum similarity
- lexical features
- local coherence overlap
- speech graph features
- ambiguous pronoun count

Failure policy
--------------
If any feature extractor fails for a given record, that entire record is skipped.
Only successful records are written to the output.

Typical usage
-------------
    python pipeline_submission_ready.py \
        --input normal.json \
        --output normal_with_features.json

    python pipeline_submission_ready.py \
        --input depression.json \
        --output depression_with_features.json \
        --save-every 5
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# === import feature modules ===
from lexical_features import extract_lexical_features
from local_coh import compute_coherence_features
from speechgraph_feature import compute_speechgraph_with_random_norm
from ambiguous_pronoun import compute_ambiguous_pronoun_features_from_pred
from basic_meas import extract_basic_features
from coh_rep import extract_semantic_features



# ---------------------------------------------------------------------------
# Notes for environment setup
# ---------------------------------------------------------------------------
# Before running this pipeline, ensure the required external resources are
# available. Typical setup commands include:
#
# python -c "import benepar; benepar.download('benepar_en3')"
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
# pip install git+https://github.com/facuzeta/speechgraph/


# ---------------------------------------------------------------------------
# Lazy global resources
# ---------------------------------------------------------------------------
_coref_model = None


def get_coref_model():
    """
    Lazily initialize and return the coreference model.

    The model is loaded only when ambiguous pronoun features are requested,
    which avoids unnecessary initialization overhead during import.
    """
    global _coref_model
    if _coref_model is None:
        print("[Init] Loading coreference model...")
        from fastcoref import FCoref

        _coref_model = FCoref(device="cpu")
        print("[Init] Coreference model ready.")
    return _coref_model


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FeatureExtractionError(Exception):
    """
    Raised when any required feature extractor fails for a record.

    The upper-level processing logic treats this as a record-level failure and
    skips the entire record.
    """


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def get_record_name(record: Dict[str, Any], idx: int) -> str:
    """
    Return a readable identifier for a record for progress logging.

    Priority order:
    ``author`` -> ``id`` -> ``post_id`` -> ``name`` -> ``title`` -> fallback index
    """
    if not isinstance(record, dict):
        return f"record_{idx}"

    for key in ["author", "id", "post_id", "name", "title"]:
        if key in record and record[key]:
            return str(record[key])

    return f"record_{idx}"


def normalize_text(text: Any) -> str:
    """
    Convert the input into a normalized string.

    This function is intentionally conservative: it strips leading/trailing
    whitespace and collapses repeated whitespace into single spaces.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def rough_sentence_count(text: str) -> int:
    """
    Estimate the number of sentences in a text using simple punctuation-based
    segmentation.

    This heuristic is used only as a lightweight guard before calling feature
    extractors that require multi-sentence input.
    """
    if not text:
        return 0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return len(sentences)


def require_valid_feature_dict(result: Any, feature_name: str) -> Dict[str, Any]:
    """
    Validate that an extractor output is a dictionary.

    Parameters
    ----------
    result:
        Extractor output to validate.
    feature_name:
        Human-readable feature block name for error reporting.

    Returns
    -------
    dict
        The validated feature dictionary.

    Raises
    ------
    FeatureExtractionError
        If the extractor returns ``None`` or a non-dictionary value.
    """
    if result is None:
        raise FeatureExtractionError(f"{feature_name} returned None")

    if not isinstance(result, dict):
        raise FeatureExtractionError(
            f"{feature_name} returned invalid type: {type(result).__name__}"
        )

    return result


def atomic_save_json(data: Any, output_path: Union[str, Path]) -> None:
    """
    Save JSON atomically to reduce the risk of partial-file corruption.

    The output is written to a temporary file first and then moved into place.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def extract_ambiguous_pronoun_feature(text: str) -> Dict[str, Any]:
    """
    Extract the ambiguous pronoun feature from a text.

    Only ``ambiguous_pronoun_count`` is retained as the final output feature.
    Any failure in coreference prediction or downstream processing causes the
    entire record to be skipped by the caller.
    """
    text = normalize_text(text)

    if not text:
        raise FeatureExtractionError("ambiguous pronoun: empty text")

    coref_model = get_coref_model()
    predictions = coref_model.predict(texts=[text])

    if predictions is None or len(predictions) == 0:
        raise FeatureExtractionError("ambiguous pronoun: coreference prediction empty")

    prediction = predictions[0]
    coref_features = compute_ambiguous_pronoun_features_from_pred(prediction)

    if not isinstance(coref_features, dict):
        raise FeatureExtractionError("ambiguous pronoun: invalid coreference feature output")

    if "ambiguous_pronoun_count" not in coref_features:
        raise FeatureExtractionError("ambiguous pronoun: missing ambiguous_pronoun_count")

    return {"ambiguous_pronoun_count": coref_features["ambiguous_pronoun_count"]}


def extract_all_features(text: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract all supported feature groups from a text.

    Parameters
    ----------
    text:
        Input text.
    verbose:
        Whether to print step-by-step progress messages.

    Returns
    -------
    dict
        Combined feature dictionary.

    Raises
    ------
    FeatureExtractionError
        If any feature extractor fails.
    """
    text = normalize_text(text)

    if not text:
        raise FeatureExtractionError("empty text")

    word_count = len(text.split())
    sentence_count = rough_sentence_count(text)

    if word_count == 0:
        raise FeatureExtractionError("text has zero words")

    if sentence_count < 2:
        raise FeatureExtractionError(
            f"text has too few sentences for full feature extraction: sent_count={sentence_count}"
        )

    features: Dict[str, Any] = {}

    try:
        if verbose:
            print("    -> basic measures")
        result = extract_basic_features(text)
        features.update(require_valid_feature_dict(result, "basic measures"))
    except Exception as e:  # pragma: no cover - depends on external models/resources
        raise FeatureExtractionError(f"basic measures failed: {repr(e)}")

    try:
        if verbose:
            print("    -> semantic coherence / maximum similarity")
        result = extract_semantic_features(text)
        features.update(require_valid_feature_dict(result, "semantic coherence"))
    except Exception as e:  # pragma: no cover
        raise FeatureExtractionError(f"semantic coherence failed: {repr(e)}")

    try:
        if verbose:
            print("    -> lexical")
        result = extract_lexical_features(text)
        features.update(require_valid_feature_dict(result, "lexical"))
    except Exception as e:  # pragma: no cover
        raise FeatureExtractionError(f"lexical failed: {repr(e)}")

    try:
        if verbose:
            print("    -> local coherence overlap")
        result = compute_coherence_features(text)
        features.update(require_valid_feature_dict(result, "local coherence overlap"))
    except Exception as e:  # pragma: no cover
        raise FeatureExtractionError(f"local coherence overlap failed: {repr(e)}")

    try:
        if verbose:
            print("    -> speech graph")
        result = compute_speechgraph_with_random_norm(text, n_iter=20)
        features.update(require_valid_feature_dict(result, "speech graph"))
    except Exception as e:  # pragma: no cover
        raise FeatureExtractionError(f"speech graph failed: {repr(e)}")

    try:
        if verbose:
            print("    -> ambiguous pronoun")
        result = extract_ambiguous_pronoun_feature(text)
        features.update(require_valid_feature_dict(result, "ambiguous pronoun"))
    except Exception as e:  # pragma: no cover
        raise FeatureExtractionError(f"ambiguous pronoun failed: {repr(e)}")

    return features


# ---------------------------------------------------------------------------
# Record-level processing
# ---------------------------------------------------------------------------
def process_record(
    record: Dict[str, Any],
    text_field: str = "text",
    idx: int = 0,
    total: int = 1,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Process a single JSON record.

    Parameters
    ----------
    record:
        Input record.
    text_field:
        Name of the field containing the text to analyse.
    idx:
        Current 1-based record index for progress logging.
    total:
        Total number of records.
    verbose:
        Whether to print detailed progress information.

    Returns
    -------
    dict or None
        A deep-copied record with an added ``features`` dictionary on success,
        or ``None`` if the record is skipped.
    """
    if not isinstance(record, dict):
        print(f"[Skip] {idx}/{total} - record is not a dict")
        return None

    record_name = get_record_name(record, idx)
    text = record.get(text_field, "")

    if verbose:
        pct = (idx / total * 100) if total > 0 else 0
        print(f"\n[Progress] {idx}/{total} ({pct:.1f}%) - Processing: {record_name}")

    if not text or not str(text).strip():
        if verbose:
            print(f"    -> skip: no text found in field '{text_field}'")
        return None

    start = time.perf_counter()

    try:
        features = extract_all_features(text=text, verbose=verbose)

        output_record = copy.deepcopy(record)
        output_record["features"] = features

        end = time.perf_counter()
        if verbose:
            print(f"    -> done in {end - start:.2f}s")

        return output_record

    except Exception as e:
        end = time.perf_counter()
        print(f"    -> skipped in {end - start:.2f}s | reason: {repr(e)}")
        return None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_json(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    text_field: str = "text",
    output_path: Optional[Union[str, Path]] = None,
    save_every: int = 10,
    verbose: bool = True,
) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a JSON object or a list of JSON objects.

    Parameters
    ----------
    data:
        Input JSON content, either a list of records or a single record.
    text_field:
        Name of the input text field.
    output_path:
        Optional checkpoint output path.
    save_every:
        Save a checkpoint every N input records when ``output_path`` is given.
    verbose:
        Whether to print detailed logs.

    Returns
    -------
    list or dict or None
        Processed output containing only successful records.
    """
    if isinstance(data, list):
        total = len(data)
        if verbose:
            print("[Input] JSON format: list of records")
            print(f"[Input] Total records: {total}")
            print(f"[Input] Text field: {text_field}")

        results: List[Dict[str, Any]] = []
        skipped_count = 0

        for i, record in enumerate(data, start=1):
            processed = process_record(
                record=record,
                text_field=text_field,
                idx=i,
                total=total,
                verbose=verbose,
            )

            if processed is not None:
                results.append(processed)
            else:
                skipped_count += 1

            if output_path and save_every > 0 and i % save_every == 0:
                try:
                    if verbose:
                        print(
                            f"[Checkpoint] Saving partial output at record {i} "
                            f"-> {output_path} | kept={len(results)} skipped={skipped_count}"
                        )
                    atomic_save_json(results, output_path)
                except Exception as e:
                    print(f"[Checkpoint Save Error] {repr(e)}")
                    print("    -> continue running...")

        if verbose:
            print(f"\n[Summary] Total input records: {total}")
            print(f"[Summary] Successful records kept: {len(results)}")
            print(f"[Summary] Skipped records: {skipped_count}")

        return results

    if isinstance(data, dict):
        if verbose:
            print("[Input] JSON format: single dict")
            print(f"[Input] Text field: {text_field}")
        return process_record(
            record=data,
            text_field=text_field,
            idx=1,
            total=1,
            verbose=verbose,
        )

    raise ValueError("Invalid JSON format: input must be a dict or a list of dicts.")


# ---------------------------------------------------------------------------
# Input preview
# ---------------------------------------------------------------------------
def preview_json_structure(data: Any) -> None:
    """
    Print a lightweight summary of the input JSON structure.
    """
    print("\n[Preview] Input JSON structure:")

    if isinstance(data, list):
        print("  - Type: list")
        print(f"  - Length: {len(data)}")
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"  - First record keys: {list(data[0].keys())}")
    elif isinstance(data, dict):
        print("  - Type: dict")
        print(f"  - Keys: {list(data.keys())}")
    else:
        print(f"  - Unsupported type: {type(data)}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Extract NLP features from JSON records.")
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSON file.")
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the text field in each record (default: text).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a checkpoint every N input records (default: 5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser


def main() -> None:
    """
    Run the command-line interface.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[Start] Reading input JSON from: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    preview_json_structure(data)

    start_all = time.perf_counter()

    result = process_json(
        data=data,
        text_field=args.text_field,
        output_path=output_path,
        save_every=args.save_every,
        verbose=not args.quiet,
    )

    print(f"\n[Final Save] Writing full output to: {output_path}")
    atomic_save_json(result, output_path)

    end_all = time.perf_counter()
    print(f"[Done] Output saved to: {output_path}")
    print(f"[Done] Total time: {end_all - start_all:.2f}s")


if __name__ == "__main__":
    main()
