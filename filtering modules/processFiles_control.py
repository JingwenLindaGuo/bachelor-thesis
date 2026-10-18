import sys
import os
import json
import re
import html
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any

if sys.version_info < (3, 10):
    raise RuntimeError("This script requires Python 3.10 or higher")


# =========================================================
# Text cleaning
# =========================================================
def clean_text_for_structure(text: str) -> str:
    if not text:
        return ""

    base_contraction_map = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }

    def build_contraction_map_final(base_map: dict[str, str]) -> dict[str, str]:
        final_map: dict[str, str] = {}
        for k, v in base_map.items():
            key_original = k.lower()
            value_final = v.lower()
            final_map[key_original] = value_final
            final_map[key_original.replace("'", "")] = value_final
        return final_map

    contraction_map_final = build_contraction_map_final(base_contraction_map)

    def expand_contractions_final(s: str, contraction_map: dict[str, str]) -> str:
        pattern = re.compile(
            r"\b(?:"
            + "|".join(re.escape(k) for k in sorted(contraction_map, key=len, reverse=True))
            + r")\b",
            flags=re.IGNORECASE,
        )

        def replace(match: re.Match[str]) -> str:
            word = match.group(0)
            return contraction_map.get(word.lower(), word)

        return pattern.sub(replace, s)

    text = text.strip()
    if text.lower() in {"[deleted]", "[removed]"}:
        return ""

    text = html.unescape(text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )

    text = re.sub(r"\\x[0-9a-fA-F]{2}", " ", text)
    text = text.replace("�", " ")

    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1 URL", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    text = re.sub(r"@(\w+)", r"\1", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)

    text = re.sub(r"\n\s*\n+", " <PARA> ", text)
    text = re.sub(r"(?<![.!?])\n+", ". ", text)
    text = re.sub(r"(?<=[.!?])\n+", " ", text)
    text = text.replace("<PARA>", "\n\n")

    text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
    text = re.sub(r"[^\w\s\.\,\!\?\:\;\'\"\-\(\)\n/]", " ", text)

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" +([)\]])", r"\1", text)

    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"\b(\w+)\s+'\s*(\w+)\b", r"\1'\2", text)
    text = re.sub(r"\b(\w+)\s+t\b", r"\1t", text)
    text = re.sub(r"\b(\w+)\s+ve\b", r"\1ve", text)
    text = re.sub(r"\b(\w+)\s+ll\b", r"\1ll", text)
    text = re.sub(r"\b(\w+)\s+re\b", r"\1re", text)
    text = re.sub(r"\b(\w+)\s+d\b", r"\1d", text)
    text = re.sub(r"\b(\w+)\s+m\b", r"\1m", text)
    text = re.sub(r"\b(\w+)\s+s\b", r"\1s", text)
    text = re.sub(r"\bcann't\b", "can't", text)

    text = expand_contractions_final(text, contraction_map_final)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# Mental-health signal detection
# =========================================================
STRONG_DIAGNOSIS_PATTERNS = [
    r"\bi was diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi have been diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi've been diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi got diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi was officially diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi was formally diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi have been formally diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi have been clinically diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bmy (?:doctor|psychiatrist|therapist|psychologist) diagnosed me with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi received (?:an? )?diagnosis of (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi got (?:an? )?diagnosis of (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bmy diagnosis is (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bi was diagnosed as having (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
]

NEGATIVE_DIAGNOSIS_PATTERNS = [
    r"\bnot diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bnever diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bmisdiagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bruled? out (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bdo not have (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bdon't have (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
    r"\bmy (?:mom|mother|dad|father|sister|brother|friend|boyfriend|girlfriend|partner|wife|husband) diagnosed\b",
    r"\bmy (?:mom|mother|dad|father|sister|brother|friend|boyfriend|girlfriend|partner|wife|husband) was diagnosed with (?:clinical )?(?:depression|major depressive disorder|mdd)\b",
]

MENTAL_HEALTH_TERM_PATTERNS = [
    r"\bmental health\b",
    r"\bdepress(?:ion|ed|ive)?\b",
    r"\banxiet(?:y|ies)\b",
    r"\bpanic attacks?\b",
    r"\bocd\b",
    r"\badhd\b",
    r"\bbipolar\b",
    r"\bbpd\b",
    r"\bborderline personality\b",
    r"\bptsd\b",
    r"\bcptsd\b",
    r"\bschizo(?:phrenia|affective)?\b",
    r"\bpsychosis\b",
    r"\bsuicid(?:e|al)\b",
    r"\bself[- ]?harm\b",
    r"\beat(?:ing)? disorder\b",
    r"\banorexi(?:a|c)\b",
    r"\bbulimi(?:a|c)\b",
    r"\bautis(?:m|tic)\b",
]

MENTAL_HEALTH_TERM_REGEXES = [re.compile(p, re.I) for p in MENTAL_HEALTH_TERM_PATTERNS]


def strong_diagnosis_pattern(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()

    if any(re.search(p, lowered, re.I) for p in NEGATIVE_DIAGNOSIS_PATTERNS):
        return False

    if any(re.search(p, lowered, re.I) for p in STRONG_DIAGNOSIS_PATTERNS):
        return True

    return False


def contains_mental_health_signal(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()

    if strong_diagnosis_pattern(lowered):
        return True

    return any(pattern.search(lowered) for pattern in MENTAL_HEALTH_TERM_REGEXES)


# =========================================================
# JSON IO
# =========================================================
def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("Input JSON must be a list of objects")
        return data

    if isinstance(data, dict):
        return [data]

    raise ValueError("Input JSON must be either a list of objects or a single object")


def save_json(data: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# Generic control filtering
# =========================================================
def get_text_from_record(record: dict[str, Any], text_field: str = "text") -> str:
    return str(record.get(text_field, "") or "").strip()


def process_control_record(
    record: dict[str, Any],
    text_field: str = "text",
    min_words: int = 500,
) -> dict[str, Any] | None:
    raw_text = get_text_from_record(record, text_field=text_field)
    if not raw_text:
        return None

    if contains_mental_health_signal(raw_text):
        return None

    cleaned_text = clean_text_for_structure(raw_text)
    if not cleaned_text:
        return None

    word_count = len(cleaned_text.split())
    if word_count < min_words:
        return None

    output_record = dict(record)
    output_record["text"] = cleaned_text 
    output_record["label"] = "control"
    output_record["filter_method"] = "lexical_exclusion"

    return output_record


def group_records_by_entity(
    records: list[dict[str, Any]],
    entity_field: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        entity_value = record.get(entity_field)
        if entity_value is None:
            continue
        grouped[str(entity_value)].append(record)

    return grouped


def entity_has_mental_health_signal(
    records: list[dict[str, Any]],
    text_field: str = "text",
) -> bool:
    for record in records:
        text = get_text_from_record(record, text_field=text_field)
        if text and contains_mental_health_signal(text):
            return True
    return False


def filter_control_instances(
    records: list[dict[str, Any]],
    text_field: str = "text",
    entity_field: str | None = None,
    min_words: int = 500,
    exclude_entity_if_any_signal: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    accepted_entities: list[dict[str, Any]] = []
    rejected_entities: list[dict[str, Any]] = []

    # -----------------------------------------------------
    # Case 1: no entity-level screening
    # -----------------------------------------------------
    if entity_field is None or not exclude_entity_if_any_signal:
        for idx, record in enumerate(records, start=1):
            result = process_control_record(
                record=record,
                text_field=text_field,
                min_words=min_words,
            )
            if result is not None:
                retained.append(result)
                print(f"[{idx}] retained control instance")
        return retained, accepted_entities, rejected_entities

    # -----------------------------------------------------
    # Case 2: entity-level screening
    # -----------------------------------------------------
    grouped = group_records_by_entity(records, entity_field=entity_field)

    for entity_id, entity_records in grouped.items():
        if entity_has_mental_health_signal(entity_records, text_field=text_field):
            rejected_entities.append({
                "entity_id": entity_id,
                "reason": "mental health signal detected in entity history",
            })
            continue

        entity_retained = []
        for record in entity_records:
            result = process_control_record(
                record=record,
                text_field=text_field,
                min_words=min_words,
            )
            if result is not None:
                entity_retained.append(result)

        if not entity_retained:
            continue

        retained.extend(entity_retained)
        accepted_entities.append({
            "entity_id": entity_id,
            "num_records": len(entity_retained),
        })

        print(f"retained entity {entity_id} with {len(entity_retained)} control instances")

    return retained, accepted_entities, rejected_entities


# =========================================================
# Main
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter control instances from generic JSON text records.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--text-field", default="text", help="Name of the text field in each JSON record")
    parser.add_argument("--entity-field", default=None, help="Optional entity field for history-level exclusion")
    parser.add_argument("--min-words", type=int, default=500, help="Minimum cleaned word count")
    parser.add_argument(
        "--no-entity-screening",
        action="store_true",
        help="Disable entity-level exclusion even if entity-field is provided",
    )
    parser.add_argument("--accepted-output", default=None, help="Optional path to accepted entity summary JSON")
    parser.add_argument("--rejected-output", default=None, help="Optional path to rejected entity summary JSON")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    records = load_json_records(args.input)

    retained, accepted_entities, rejected_entities = filter_control_instances(
        records=records,
        text_field=args.text_field,
        entity_field=args.entity_field,
        min_words=args.min_words,
        exclude_entity_if_any_signal=not args.no_entity_screening,
    )

    save_json(retained, args.output)

    if args.accepted_output is not None:
        save_json(accepted_entities, args.accepted_output)

    if args.rejected_output is not None:
        save_json(rejected_entities, args.rejected_output)

    print(f"Saved {len(retained)} control instances to {args.output}")

    if args.accepted_output is not None:
        print(f"Saved accepted entity summary to {args.accepted_output}")

    if args.rejected_output is not None:
        print(f"Saved rejected entity summary to {args.rejected_output}")


if __name__ == "__main__":
    main()