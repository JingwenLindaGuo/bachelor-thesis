# -*- coding: utf-8 -*- 
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


# =========================================================
# Notes
# =========================================================
# Before running:
# python -c "import benepar; benepar.download('benepar_en3')"
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# pip install git+https://github.com/facuzeta/speechgraph/
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"


# =========================================================
# Global lazy resources
# =========================================================
_coref_model = None


def get_coref_model():
    """
    Lazily load the coreference model only when needed.
    This avoids initializing a large model during import time.
    """
    global _coref_model
    if _coref_model is None:
        print("[Init] Loading coreference model...")
        from fastcoref import FCoref
        _coref_model = FCoref(device="cpu")
        print("[Init] Coreference model ready.")
    return _coref_model


# =========================================================
# Exceptions
# =========================================================
class FeatureExtractionError(Exception):
    """
    Raised when any feature extractor fails.
    The upper-level logic will then skip the entire record.
    """
    pass


# =========================================================
# Utility
# =========================================================
def get_record_name(record: Dict[str, Any], idx: int) -> str:
    """
    Return a readable name for each record to make progress logs clearer.

    Priority order:
    author -> id -> post_id -> name -> title -> idx
    """
    if not isinstance(record, dict):
        return f"record_{idx}"

    for key in ["author", "id", "post_id", "name", "title"]:
        if key in record and record[key]:
            return str(record[key])

    return f"record_{idx}"


def normalize_text(text: Any) -> str:
    """
    Convert the input into a safe string as much as possible.
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
    Roughly estimate the number of sentences in the text,
    to avoid sending obviously unsuitable text into some extractors.
    """
    if not text:
        return 0
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return len(sents)


def require_valid_feature_dict(result: Any, feature_name: str) -> Dict[str, Any]:
    """
    Ensure that an extractor returns a dictionary.
    Otherwise, treat it as a failure and skip the entire record.
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
    Save output atomically to avoid corrupting the output file
    if the program is interrupted during writing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


# =========================================================
# Feature extraction: ambiguous pronoun
# =========================================================
def extract_ambiguous_pronoun_feature(text: str) -> Dict[str, Any]:
    """
    Input: text
    Output: ambiguous pronoun feature

    Only ambiguous_pronoun_count is kept as the final feature.
    Exceptions are not suppressed here; if anything fails,
    the entire record will be skipped.
    """
    text = normalize_text(text)

    if not text:
        raise FeatureExtractionError("ambiguous pronoun: empty text")

    coref_model = get_coref_model()
    preds = coref_model.predict(texts=[text])

    if preds is None or len(preds) == 0:
        raise FeatureExtractionError("ambiguous pronoun: coref prediction empty")

    pred = preds[0]
    coref_features = compute_ambiguous_pronoun_features_from_pred(pred)

    if not isinstance(coref_features, dict):
        raise FeatureExtractionError("ambiguous pronoun: invalid coref feature output")

    if "ambiguous_pronoun_count" not in coref_features:
        raise FeatureExtractionError("ambiguous pronoun: missing ambiguous_pronoun_count")

    return {
        "ambiguous_pronoun_count": coref_features["ambiguous_pronoun_count"]
    }


# =========================================================
# Feature extraction: all
# =========================================================
def extract_all_features(text: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Core function: text -> all features

    Rule:
    If any extractor fails, raise an exception immediately,
    and the upper-level logic will skip the entire record.
    """
    text = normalize_text(text)

    if not text:
        raise FeatureExtractionError("empty text")

    word_count = len(text.split())
    sent_count = rough_sentence_count(text)

    if word_count == 0:
        raise FeatureExtractionError("text has zero words")

    # Semantic/coherence features usually require at least 2 sentences
    if sent_count < 2:
        raise FeatureExtractionError(
            f"text has too few sentences for full feature extraction: sent_count={sent_count}"
        )

    features: Dict[str, Any] = {}

    # === basic measures ===
    try:
        if verbose:
            print("    -> basic measures")
        result = extract_basic_features(text)
        features.update(require_valid_feature_dict(result, "basic measures"))
    except Exception as e:
        raise FeatureExtractionError(f"basic measures failed: {repr(e)}")

    # === semantic coherence + maximum similarity ===
    try:
        if verbose:
            print("    -> semantic coherence / maximum similarity")
        result = extract_semantic_features(text)
        features.update(require_valid_feature_dict(result, "semantic coherence"))
    except Exception as e:
        raise FeatureExtractionError(f"semantic coherence failed: {repr(e)}")

    # === lexical ===
    try:
        if verbose:
            print("    -> lexical")
        result = extract_lexical_features(text)
        features.update(require_valid_feature_dict(result, "lexical"))
    except Exception as e:
        raise FeatureExtractionError(f"lexical failed: {repr(e)}")

    # === coherence overlap ===
    try:
        if verbose:
            print("    -> local coherence overlap")
        result = compute_coherence_features(text)
        features.update(require_valid_feature_dict(result, "local coherence overlap"))
    except Exception as e:
        raise FeatureExtractionError(f"local coherence overlap failed: {repr(e)}")

    # === speech graph ===
    try:
        if verbose:
            print("    -> speech graph")
        result = compute_speechgraph_with_random_norm(text, n_iter=20)
        features.update(require_valid_feature_dict(result, "speech graph"))
    except Exception as e:
        raise FeatureExtractionError(f"speech graph failed: {repr(e)}")

    # === ambiguous pronoun ===
    try:
        if verbose:
            print("    -> ambiguous pronoun")
        result = extract_ambiguous_pronoun_feature(text)
        features.update(require_valid_feature_dict(result, "ambiguous pronoun"))
    except Exception as e:
        raise FeatureExtractionError(f"ambiguous pronoun failed: {repr(e)}")

    return features


# =========================================================
# Record processing
# =========================================================
def process_record(
    record: Dict[str, Any],
    text_field: str = "text",
    idx: int = 0,
    total: int = 1,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Process a single JSON record.

    Returns:
    - success: a new record with extracted features
    - failure: None, meaning the entire record is skipped
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

        # Do not modify the input record in place
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


# =========================================================
# Batch processing
# =========================================================
def process_json(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    text_field: str = "text",
    output_path: Optional[Union[str, Path]] = None,
    save_every: int = 10,
    verbose: bool = True,
) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Supported input formats:
    - list of dicts
    - single dict

    Parameters:
    - text_field: name of the input text field
    - output_path: path for intermediate checkpoint saving
    - save_every: save once every N input records
    - verbose: whether to print detailed progress logs

    Rules:
    - If any extractor fails for a record, skip the entire record
    - Only successful records are kept in the output
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
            data,
            text_field=text_field,
            idx=1,
            total=1,
            verbose=verbose,
        )

    raise ValueError("Invalid JSON format: input must be a dict or a list of dicts.")


# =========================================================
# Preview
# =========================================================
def preview_json_structure(data: Any) -> None:
    """
    Print the basic structure of the input JSON
    to help confirm the format.
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


# =========================================================
# CLI
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract NLP features from JSON records."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the text field in each record (default: text)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N input records (default: 5)",
    )
    return parser


def main() -> None:
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
        verbose=True,
    )

    print(f"\n[Final Save] Writing full output to: {output_path}")
    atomic_save_json(result, output_path)

    end_all = time.perf_counter()
    print(f"[Done] Output saved to: {output_path}")
    print(f"[Done] Total time: {end_all - start_all:.2f}s")


if __name__ == "__main__":
    main()