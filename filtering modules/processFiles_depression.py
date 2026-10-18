import sys
import os
import json
import re
import html
import argparse
from pathlib import Path
from typing import Any

if sys.version_info < (3, 10):
    raise RuntimeError("This script requires Python 3.10 or higher")


# =========================================================
# Text cleaning
# =========================================================
def clean_text_for_diagnosis_detection(text: str) -> str:
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
    text = re.sub(r"([\(\[]) +", r"\1", text)
    text = re.sub(r" +([\)\]])", r"\1", text)

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
# Rule-based diagnosis pre-filter
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


def regex_prefilter_diagnosis(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()

    if any(re.search(p, lowered, re.I) for p in NEGATIVE_DIAGNOSIS_PATTERNS):
        return False

    if any(re.search(p, lowered, re.I) for p in STRONG_DIAGNOSIS_PATTERNS):
        return True

    return False


# =========================================================
# LLM verification
# =========================================================
def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise clinical text annotator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def detect_depression_llm(text: str, model: str = "gpt-4o-mini") -> tuple[str, str | None]:
    prompt = f"""
You are a clinical text annotator.

Task:
Determine whether the following text contains an explicit self-disclosure of a CURRENT clinical diagnosis of depression.

Strict Criteria (ALL must be satisfied):

1. Clinical diagnosis:
The author must explicitly indicate a formal diagnosis by a professional (e.g., "diagnosed", "doctor", "psychiatrist", "therapist").

2. Explicit statement:
The statement must be certain, not speculative.
- EXCLUDE: "I think I have depression", "I might have depression"

3. Current condition:
The author must clearly indicate they are currently experiencing depression.
- INCLUDE: "I was diagnosed with depression and still struggle"
- EXCLUDE: "I was diagnosed years ago" (without indication of current condition)

Important:
- "I have depression" alone is NOT sufficient unless combined with clinical evidence.
- Past diagnosis without current suffering -> NO

Output JSON only:
{{
  "label": "YES" or "NO",
  "evidence": null or "exact sentence"
}}

Text:
{text}
""".strip()

    response_text = call_llm(prompt, model=model)

    try:
        result = json.loads(response_text)
        label = str(result.get("label", "NO")).strip().upper()
        evidence = result.get("evidence", None)
        if evidence == "null":
            evidence = None
        if label not in {"YES", "NO"}:
            label = "NO"
        return label, evidence
    except json.JSONDecodeError:
        print("Warning: failed to parse LLM JSON response")
        print("Raw response:", response_text)
        return "NO", None


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
        # allow single-record JSON
        return [data]

    raise ValueError("Input JSON must be either a list of objects or a single object")


def save_json(data: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# Record processing
# =========================================================
def process_record(
    record: dict[str, Any],
    text_field: str = "text",
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    min_words: int = 0,
) -> dict[str, Any] | None:
    raw_text = str(record.get(text_field, "") or "").strip()
    if not raw_text:
        return None

    cleaned_text = clean_text_for_diagnosis_detection(raw_text)
    if not cleaned_text:
        return None

    cleaned_word_count = len(cleaned_text.split())
    if cleaned_word_count < min_words:
        return None

    if not regex_prefilter_diagnosis(cleaned_text):
        return None

    evidence: str | None = None
    detection_method = "regex"

    if use_llm:
        label, evidence = detect_depression_llm(cleaned_text, model=model)
        if label != "YES":
            return None
        detection_method = "regex+llm"

    output_record = dict(record)  # preserve original metadata/covariates
    output_record["text"] = raw_text
    output_record["cleaned_text"] = cleaned_text 
    output_record["label"] = "depressed"
    output_record["evidence"] = evidence
    output_record["detection_method"] = detection_method

    return output_record


def filter_depressed_instances(
    records: list[dict[str, Any]],
    text_field: str = "text",
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    min_words: int = 0,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        result = process_record(
            record=record,
            text_field=text_field,
            use_llm=use_llm,
            model=model,
            min_words=min_words,
        )
        if result is not None:
            retained.append(result)
            print(f"[{idx}] retained depressed instance")

    return retained


# =========================================================
# Main
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter depressed instances from generic JSON text records.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--text-field", default="text", help="Name of the text field in each JSON record")
    parser.add_argument("--min-words", type=int, default=500, help="Minimum cleaned word count required to retain a record")
    parser.add_argument("--no-llm", action="store_true", help="Use regex pre-filter only")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    records = load_json_records(args.input)
    retained = filter_depressed_instances(
        records=records,
        text_field=args.text_field,
        use_llm=not args.no_llm,
        model=args.model,
        min_words=args.min_words,
    )
    save_json(retained, args.output)

    print(f"Saved {len(retained)} depressed instances to {args.output}")


if __name__ == "__main__":
    main()
