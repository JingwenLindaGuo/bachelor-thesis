"""
Reddit diagnosis-post filtering pipeline.

This script scans Reddit submission .zst files and identifies self-posts that
contain an explicit self-disclosure of a current clinical diagnosis of depression.

Implemented stages:
1. Stream Reddit submission files from a single file or a folder
2. Keep self-posts only
3. Clean social media text for downstream analysis
4. Apply regex-based diagnosis detection
5. Optionally apply LLM verification
6. Save diagnosis hits
7. Optionally build user-level time windows and collect posts within those windows

Typical usage:
    python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs
    python diagnosis_pipeline.py --input-path /path/to/submissions --output-dir outputs --use-llm --collect-window-posts
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fileStreams import getFileJsonStream
from utils import FileProgressLog


# =========================
# Python version guard
# =========================
if sys.version_info < (3, 10):
    raise RuntimeError("This script requires Python 3.10 or higher.")


# =========================
# Regex patterns
# =========================
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


# =========================
# Argument parsing
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Filter Reddit submissions for explicit self-disclosures of a current clinical diagnosis of depression."
    )

    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a single .zst file or a folder containing .zst files.",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnosis_pipeline_outputs",
        help="Directory for JSON outputs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .zst files when input-path is a folder.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=None,
        help="Optional global cap on the number of diagnosis hits to collect.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM verification after regex pre-filtering.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Model name for LLM verification.",
    )
    parser.add_argument(
        "--collect-window-posts",
        action="store_true",
        help="After collecting diagnosis hits, build user windows and collect posts within those windows.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=183,
        help="Number of days before the earliest diagnosis post to include in each user's time window.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Minimum raw word count for posts collected in the time windows.",
    )
    parser.add_argument(
        "--group-posts-by-user",
        action="store_true",
        help="Additionally save collected window posts grouped by author.",
    )

    return parser


# =========================
# Utility functions
# =========================
def iter_input_files(input_path: Path, recursive: bool) -> list[Path]:
    """
    Resolve the input path to a sorted list of .zst files.

    Parameters
    ----------
    input_path : Path
        Path to a single file or a directory.
    recursive : bool
        Whether to recursively search subfolders when input_path is a directory.

    Returns
    -------
    list[Path]
        Sorted list of .zst files to process.
    """
    if input_path.is_file():
        if input_path.suffix != ".zst":
            raise ValueError(f"Expected a .zst file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if recursive:
        files = sorted(p for p in input_path.rglob("*.zst") if p.is_file())
    else:
        files = sorted(p for p in input_path.glob("*.zst") if p.is_file())

    if not files:
        raise FileNotFoundError(f"No .zst files found under: {input_path}")

    return files


def save_json(data: Any, output_path: Path) -> None:
    """
    Save data as UTF-8 JSON with indentation.

    Datetime objects are serialized to ISO-8601 strings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def default_serializer(obj: Any) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=default_serializer)


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output.

    This helper strips optional Markdown code fences before parsing.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return json.loads(cleaned)


def clean_social_media_text_for_structure(text: str) -> str:
    """
    Clean social media text while preserving sentence- and discourse-level
    structure as much as possible for downstream analysis.
    """
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
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
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

        def replace(match: re.Match) -> str:
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


def strong_diagnosis_pattern(text: str) -> bool:
    """
    Return True if the text matches a strong diagnosis disclosure pattern and
    does not match any negative exclusion pattern.
    """
    if not text:
        return False

    lowered = text.lower()

    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in NEGATIVE_DIAGNOSIS_PATTERNS):
        return False

    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in STRONG_DIAGNOSIS_PATTERNS):
        return True

    return False


def is_deleted_or_removed(text: str) -> bool:
    """Return True if the text is an obvious placeholder rather than real content."""
    return text.strip().lower() in {"[deleted]", "[removed]"}


def deduplicate_records(records: list[dict[str, Any]], id_key: str = "id") -> list[dict[str, Any]]:
    """Deduplicate records by a given ID field while preserving order."""
    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []

    for record in records:
        record_id = record.get(id_key)
        if record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(record)

    return deduped


# =========================
# Optional LLM verification
# =========================
def build_openai_client(use_llm: bool):
    """
    Lazily create and return an OpenAI client if LLM verification is enabled.

    Returns None when use_llm is False.
    """
    if not use_llm:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either set the environment variable or run without --use-llm."
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def call_llm(client, prompt: str, model: str) -> str:
    """Call the chat completion API and return the raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise clinical text annotator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def detect_depression_llm(client, text: str, model: str) -> tuple[str, Optional[str]]:
    """
    Use an LLM to verify whether the text contains an explicit self-disclosure
    of a current clinical diagnosis of depression.

    Returns
    -------
    tuple[str, Optional[str]]
        ("YES" or "NO", evidence sentence or None)
    """
    prompt = f"""
You are a clinical text annotator.

Task:
Determine whether the following text contains an explicit self-disclosure of a CURRENT clinical diagnosis of depression.

Strict Criteria (ALL must be satisfied):

1. Clinical diagnosis:
The author must explicitly indicate a formal diagnosis by a professional
(e.g., "diagnosed", "doctor", "psychiatrist", "therapist").

2. Explicit statement:
The statement must be certain, not speculative.
- EXCLUDE: "I think I have depression", "I might have depression"

3. Current condition:
The author must clearly indicate they are currently experiencing depression.
- INCLUDE: "I was diagnosed with depression and still struggle"
- EXCLUDE: "I was diagnosed years ago" (without indication of current condition)

Important:
- "I have depression" ALONE is NOT sufficient unless combined with clinical evidence.
- Past diagnosis WITHOUT current suffering -> NO

Output JSON only:
{{
  "label": "YES" or "NO",
  "evidence": null or "exact sentence"
}}

If there is no evidence, evidence must be null (without quotes).

Text:
{text}
"""

    raw_response = call_llm(client, prompt, model)

    try:
        result = parse_json_object(raw_response)
        label = str(result.get("label", "NO")).upper().strip()
        evidence = result.get("evidence", None)

        if label not in {"YES", "NO"}:
            label = "NO"

        if evidence == "null":
            evidence = None

        if evidence is not None:
            evidence = str(evidence)

        return label, evidence

    except json.JSONDecodeError:
        print("[Warning] Failed to parse JSON from LLM response. Defaulting to NO.")
        print(f"[Warning] Raw response: {raw_response}")
        return "NO", None


# =========================
# Pass 1: diagnosis hit collection
# =========================
def collect_diagnosis_hits_from_file(
    file_path: Path,
    client=None,
    llm_model: str = "gpt-4o-mini",
    max_hits: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Process one .zst file and collect candidate diagnosis posts.

    The pipeline first cleans the raw text, then uses the cleaned text for
    regex filtering and optional LLM verification. The saved 'text' field is
    the cleaned version rather than the original raw text.
    """
    print(f"[Pass 1] Processing file: {file_path}")
    hits: list[dict[str, Any]] = []

    with file_path.open("rb") as f:
        json_stream = getFileJsonStream(str(file_path), f)
        if json_stream is None:
            print(f"[Warning] Skipping unsupported file: {file_path}")
            return hits

        progress_log = FileProgressLog(str(file_path), f)

        for row in json_stream:
            progress_log.onRow()

            if max_hits is not None and len(hits) >= max_hits:
                break

            if not isinstance(row, dict):
                continue

            if not row.get("is_self"):
                continue

            raw_text = (row.get("selftext") or "").strip()
            if not raw_text or is_deleted_or_removed(raw_text):
                continue

            text = clean_social_media_text_for_structure(raw_text)
            if not text:
                continue

            if not strong_diagnosis_pattern(text):
                continue

            llm_label = None
            evidence = None

            if client is not None:
                llm_label, evidence = detect_depression_llm(client, text, llm_model)
                if llm_label != "YES":
                    continue

            hit = {
                "author": row.get("author"),
                "created_utc": row.get("created_utc"),
                "id": row.get("id"),
                "subreddit": row.get("subreddit"),
                "text": text,
                "llm_label": llm_label,
                "evidence": evidence,
            }
            hits.append(hit)

    return hits


# =========================
# Pass 2 helpers: time windows and post collection
# =========================
def build_user_windows(diagnosis_hits: list[dict[str, Any]], days_back: int = 183) -> dict[str, dict[str, Any]]:
    """
    Build a time window for each author based on their diagnosis-post timestamps.

    If a user has one diagnosis hit:
        start_time = diagnosis_time - days_back
        end_time = diagnosis_time

    If a user has multiple diagnosis hits:
        start_time = earliest_diagnosis_time - days_back
        end_time = latest_diagnosis_time
    """
    user_times: dict[str, list[datetime]] = defaultdict(list)

    for hit in diagnosis_hits:
        author = hit.get("author")
        ts = hit.get("created_utc")

        if not author or ts is None:
            continue

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        user_times[author].append(dt)

    user_windows: dict[str, dict[str, Any]] = {}

    for author, times in user_times.items():
        times = sorted(times)

        if len(times) == 1:
            end_time = times[0]
            start_time = end_time - timedelta(days=days_back)
        else:
            earliest = times[0]
            latest = times[-1]
            start_time = earliest - timedelta(days=days_back)
            end_time = latest

        user_windows[author] = {
            "start_time": start_time,
            "end_time": end_time,
            "num_diagnosis_posts": len(times),
            "diagnosis_times": times,
        }

    return user_windows


def collect_posts_in_windows_from_file(
    file_path: Path,
    user_windows: dict[str, dict[str, Any]],
    min_words: int = 500,
) -> list[dict[str, Any]]:
    """
    Process one .zst file and collect self-posts that fall within previously
    defined user time windows.

    Posts are filtered by raw word count, but the saved 'text' field is the
    cleaned version produced by clean_social_media_text_for_structure().
    """
    print(f"[Pass 2] Processing file: {file_path}")
    collected: list[dict[str, Any]] = []

    with file_path.open("rb") as f:
        json_stream = getFileJsonStream(str(file_path), f)
        if json_stream is None:
            print(f"[Warning] Skipping unsupported file: {file_path}")
            return collected

        progress_log = FileProgressLog(str(file_path), f)

        for row in json_stream:
            progress_log.onRow()

            if not isinstance(row, dict):
                continue

            if not row.get("is_self"):
                continue

            author = row.get("author")
            if author not in user_windows:
                continue

            ts = row.get("created_utc")
            if ts is None:
                continue

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            window = user_windows[author]

            if not (window["start_time"] <= dt <= window["end_time"]):
                continue

            raw_text = (row.get("selftext") or "").strip()
            if not raw_text or is_deleted_or_removed(raw_text):
                continue

            word_count = len(raw_text.split())
            if word_count < min_words:
                continue

            text = clean_social_media_text_for_structure(raw_text)
            if not text:
                continue

            collected.append(
                {
                    "author": author,
                    "id": row.get("id"),
                    "subreddit": row.get("subreddit"),
                    "created_utc": ts,
                    "datetime_utc": dt.isoformat(),
                    "word_count": word_count,
                    "text": text,
                }
            )

    return collected


def group_posts_by_user(posts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group collected posts by author and sort each author's posts by timestamp."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for post in posts:
        author = post.get("author")
        if author:
            grouped[author].append(post)

    for author in grouped:
        grouped[author].sort(key=lambda x: x.get("created_utc", 0))

    return dict(grouped)


# =========================
# Main pipeline
# =========================
def main() -> None:
    """Run the diagnosis filtering pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    files = iter_input_files(input_path, recursive=args.recursive)

    print("=" * 80)
    print("Diagnosis filtering pipeline started")
    print(f"Input path: {input_path}")
    print(f"Number of files: {len(files)}")
    print(f"Output directory: {output_dir}")
    print(f"LLM verification: {args.use_llm}")
    print(f"Collect window posts: {args.collect_window_posts}")
    print("=" * 80)

    client = build_openai_client(args.use_llm)

    # -------------------------
    # Pass 1: collect diagnosis hits
    # -------------------------
    all_hits: list[dict[str, Any]] = []

    for file_index, file_path in enumerate(files, start=1):
        print(f"\n[Pass 1] File {file_index}/{len(files)}")

        remaining = None
        if args.max_hits is not None:
            remaining = args.max_hits - len(all_hits)
            if remaining <= 0:
                break

        file_hits = collect_diagnosis_hits_from_file(
            file_path=file_path,
            client=client,
            llm_model=args.llm_model,
            max_hits=remaining,
        )
        all_hits.extend(file_hits)

    all_hits = deduplicate_records(all_hits, id_key="id")
    diagnosis_hits_path = output_dir / "diagnosis_hits.json"
    save_json(all_hits, diagnosis_hits_path)

    print(f"\n[Done] Total diagnosis hits: {len(all_hits)}")
    print(f"[Saved] {diagnosis_hits_path}")

    # -------------------------
    # Optional Pass 2
    # -------------------------
    if args.collect_window_posts:
        user_windows = build_user_windows(all_hits, days_back=args.days_back)
        user_windows_path = output_dir / "user_windows.json"
        save_json(user_windows, user_windows_path)

        print(f"\n[Done] Built user windows for {len(user_windows)} users")
        print(f"[Saved] {user_windows_path}")

        all_window_posts: list[dict[str, Any]] = []

        for file_index, file_path in enumerate(files, start=1):
            print(f"\n[Pass 2] File {file_index}/{len(files)}")
            file_posts = collect_posts_in_windows_from_file(
                file_path=file_path,
                user_windows=user_windows,
                min_words=args.min_words,
            )
            all_window_posts.extend(file_posts)

        all_window_posts = deduplicate_records(all_window_posts, id_key="id")
        window_posts_path = output_dir / "posts_in_windows.json"
        save_json(all_window_posts, window_posts_path)

        print(f"\n[Done] Total collected window posts: {len(all_window_posts)}")
        print(f"[Saved] {window_posts_path}")

        if args.group_posts_by_user:
            grouped = group_posts_by_user(all_window_posts)
            grouped_path = output_dir / "posts_by_user.json"
            save_json(grouped, grouped_path)
            print(f"[Saved] {grouped_path}")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()