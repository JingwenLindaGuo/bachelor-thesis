import sys
version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 10):
	raise RuntimeError("This script requires Python 3.10 or higher")
import os
from typing import Iterable

from fileStreams import getFileJsonStream
from utils import FileProgressLog
import json
import re
import html
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os

fileOrFolderPath = r"C:\Users\guoji\Downloads\reddit\submissions"
recursive = False
 

import os
import time

def format_seconds(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def print_progress(prefix, start_time, processed_bytes, total_bytes):
    elapsed = time.time() - start_time
    pct = (processed_bytes / total_bytes * 100) if total_bytes > 0 else 0
    speed = processed_bytes / elapsed if elapsed > 0 else 0  # bytes/sec
    remaining = total_bytes - processed_bytes
    eta = remaining / speed if speed > 0 else None

    print(
        f"{prefix} | "
        f"{pct:6.2f}% | "
        f"{processed_bytes/1024/1024/1024:8.2f}/{total_bytes/1024/1024/1024:8.2f} GB | "
        f"{speed/1024/1024:7.2f} MB/s | "
        f"ETA {format_seconds(eta)}"
    )

import re
import html


def clean_social_media_text_for_structure(text: str) -> str:
    if not text:
        return ""

    # ===== contraction resources =====
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

    def build_contraction_map_final(base_map):
        final_map = {}
        for k, v in base_map.items():
            key_original = k.lower()
            value_final = v.lower()

            # 原始形式
            final_map[key_original] = value_final

            # 去掉 apostrophe 的形式
            key_no_apostrophe = key_original.replace("'", "")
            final_map[key_no_apostrophe] = value_final

        return final_map

    contraction_map_final = build_contraction_map_final(base_contraction_map)

    def expand_contractions_final(s, contraction_map):
        pattern = re.compile(
            r"\b(?:"
            + "|".join(re.escape(k) for k in sorted(contraction_map, key=len, reverse=True))
            + r")\b",
            flags=re.IGNORECASE,
        )

        def replace(match):
            word = match.group(0)
            return contraction_map.get(word.lower(), word)

        return pattern.sub(replace, s)

    # 0) remove obvious placeholders
    text = text.strip()
    if text.lower() in {"[deleted]", "[removed]"}:
        return ""

    # 1) HTML unescape
    text = html.unescape(text)

    # 1.5) normalize curly quotes / apostrophes FIRST
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", "-")
            .replace("–", "-")
    )

    # 2) encoding removal / mojibake cleanup
    text = re.sub(r'\\x[0-9a-fA-F]{2}', ' ', text)
    text = text.replace('�', ' ')

    # 3) markdown links: [text](url) -> text URL
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1 URL', text)

    # 4) remove plain URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    # 5) remove markdown emphasis markers but keep text
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

    # 6) remove @ symbol only, keep username
    text = re.sub(r'@(\w+)', r'\1', text)

    # 7) remove hashtags completely
    text = re.sub(r'#\w+', ' ', text)

    # 8) remove emoji / other non-text symbols
    text = re.sub(r'[\U00010000-\U0010ffff]', ' ', text)

    # 9) preserve paragraph structure
    text = re.sub(r'\n\s*\n+', ' <PARA> ', text)
    text = re.sub(r'(?<![.!?])\n+', '. ', text)
    text = re.sub(r'(?<=[.!?])\n+', ' ', text)
    text = text.replace('<PARA>', '\n\n')

    # 10) normalize repeated punctuation
    text = re.sub(r'([!?.,;:])\1{1,}', r'\1', text)

    # 11) keep useful punctuation for syntax/discourse
    text = re.sub(r"[^\w\s\.\,\!\?\:\;\'\"\-\(\)\n/]", ' ', text)

    # 12) normalize spaces around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([(\[]) +', r'\1', text)
    text = re.sub(r' +([)\]])', r'\1', text)

    # 13) lowercase
    text = text.lower()

    # 14) clean extra spaces but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ===== FINAL STEP: fix + expand contractions =====

    # A. fix split apostrophe: i 'm -> i'm, didn 't -> didn't
    text = re.sub(r"\b(\w+)\s+'\s*(\w+)\b", r"\1'\2", text)

    # B. fix very common split no-apostrophe patterns before expansion
    # can t -> cant, youll ve -> youllve, i m -> im
    text = re.sub(r"\b(\w+)\s+t\b", r"\1t", text)
    text = re.sub(r"\b(\w+)\s+ve\b", r"\1ve", text)
    text = re.sub(r"\b(\w+)\s+ll\b", r"\1ll", text)
    text = re.sub(r"\b(\w+)\s+re\b", r"\1re", text)
    text = re.sub(r"\b(\w+)\s+d\b", r"\1d", text)
    text = re.sub(r"\b(\w+)\s+m\b", r"\1m", text)
    text = re.sub(r"\b(\w+)\s+s\b", r"\1s", text)

    # C. specific typo fix before final expansion
    text = re.sub(r"\bcann't\b", "can't", text)

    # D. fully expand all supported contractions
    text = expand_contractions_final(text, contraction_map_final)

    # E. final whitespace cleanup
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


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

def strong_diagnosis_pattern(text: str) -> bool:
    if not text:
        return False

    text = text.lower()

    if any(re.search(p, text, re.I) for p in NEGATIVE_DIAGNOSIS_PATTERNS):
        return False

    if any(re.search(p, text, re.I) for p in STRONG_DIAGNOSIS_PATTERNS):
        return True

    return False


 

MIN_WORDS = 500
TARGET_AUTHORS = 1600
OUTPUT_PATH = "normal.json"
MAX_CANDIDATE_AUTHORS = 10000

# 这里定义的是“只要作者在这些 subreddit 发过 post，就直接剔除”
MENTAL_HEALTH_SUBREDDITS = {
    "depression",
    "mentalhealth",
    "suicidewatch",
    "anxiety",
    "bipolar",
    "bpd",
    "ptsd",
    "cptsd",
    "ocd",
    "adhd",
    "addiction",
    "selfharm",
    "stopselfharm",
    "eatingdisorders",
    "edanonymous",
    "autism",
    "schizophrenia",
    "psychosis",
    "raisedbynarcissists",
    "talktherapy",
    "therapy",
}

# subreddit 名字里只要出现这些 keyword，也判为 mental-health related
MENTAL_HEALTH_SUBREDDIT_KEYWORDS = [
    "depress",
    "mentalhealth",
    "suicide",
    "anxiety",
    "bipolar",
    "ptsd",
    "cptsd",
    "ocd",
    "adhd",
    "selfharm",
    "eatingdisorder",
    "psychosis",
    "schizo",
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


def get_zst_files(path: str):
    if recursive:
        files = []
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if f.endswith(".zst"):
                    files.append(os.path.join(root, f))
        return sorted(files)
    else:
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".zst")
        )


def get_post_body(row) -> str:
    return (row.get("selftext") or "").strip()


def get_post_analysis_text(row) -> str:
    """
    用于检查 mental-health signal。
    这里把 title + selftext 一起看，避免 signal 只出现在 title 里。
    """
    title = (row.get("title") or "").strip()
    body = (row.get("selftext") or "").strip()

    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def is_valid_author(author: str) -> bool:
    if not author:
        return False
    author = author.strip().lower()
    return author not in {"[deleted]", "automoderator"}


def is_mental_health_subreddit(subreddit: str) -> bool:
    if not subreddit:
        return False
    sub = subreddit.strip().lower()

    if sub in MENTAL_HEALTH_SUBREDDITS:
        return True

    return any(keyword in sub for keyword in MENTAL_HEALTH_SUBREDDIT_KEYWORDS)


def contains_mental_health_signal(text: str) -> bool:
    """
    只要出现明显 mental-health terminology，或者强 diagnosis pattern，就剔除。
    """
    if not text:
        return False

    text = text.lower()

    if strong_diagnosis_pattern(text):
        return True

    for pattern in MENTAL_HEALTH_TERM_REGEXES:
        if pattern.search(text):
            return True

    return False


def discover_candidate_authors(files, min_words=500, max_candidate_authors=10000):
    """
    PASS 1:
    尽快找到一批 candidate authors。
    一旦唯一 candidate author 数量达到 max_candidate_authors，就立刻停止，
    不再继续扫描剩余文件。
    
    注意：
    这里建议 files 传入时已经按“新 -> 旧”排序，
    这样 early stop 得到的 author 会更偏向近期。
    """
    latest_candidate_post = {}

    print("\n=== PASS 1: Discover candidate authors from 2025 posts ===")

    # 为了让 early stop 更偏向“最近”
    files = sorted(files, reverse=True)

    stop_early = False

    for i, file in enumerate(files):
        print(f"[PASS1] Processing file {i+1:3} {file}")

        with open(file, "rb") as f:
            jsonStream = getFileJsonStream(file, f)
            if jsonStream is None:
                print(f"Skipping unknown file {file}")
                continue

            progressLog = FileProgressLog(file, f)

            for row in jsonStream:
                progressLog.onRow()

                if not row.get("is_self"):
                    continue

                author = row.get("author")
                if not is_valid_author(author):
                    continue

                body = get_post_body(row)
                if not body:
                    continue

                raw_word_count = len(body.split())
                if raw_word_count < min_words:
                    continue

                ts = row.get("created_utc")
                if ts is None:
                    continue

                prev = latest_candidate_post.get(author)

                # 只要是这个 author 第一次被发现，就加入 candidate pool
                if prev is None:
                    latest_candidate_post[author] = {
                        "created_utc": ts,
                        "id": row.get("id"),
                        "subreddit": row.get("subreddit"),
                    }

                    if len(latest_candidate_post) >= max_candidate_authors:
                        print(f"Reached {max_candidate_authors} candidate authors during PASS 1")
                        stop_early = True
                        break

        if stop_early:
            break

    candidate_authors = list(latest_candidate_post.keys())

    print(f"Total candidate authors kept: {len(candidate_authors)}")
    return candidate_authors, latest_candidate_post


def inspect_candidate_authors(files, candidate_authors, min_words=500):
    """
    PASS 2:
    对 candidate authors 扫全年历史：
    - 只要发过 mental-health subreddit -> 剔除
    - 只要文本里出现 mental-health terminology / diagnosis -> 剔除
    - 否则收集其所有 500+ posts（text 先 clean）
    """
    candidate_set = set(candidate_authors)

    author_info = {
        author: {
            "disqualified": False,
            "reason": None,
            "posts": []
        }
        for author in candidate_authors
    }

    print("\n=== PASS 2: Inspect full 2025 posting history of candidate authors ===")

    for i, file in enumerate(files):
        print(f"[PASS2] Processing file {i+1:3} {file}")

        with open(file, "rb") as f:
            jsonStream = getFileJsonStream(file, f)
            if jsonStream is None:
                print(f"Skipping unknown file {file}")
                continue

            progressLog = FileProgressLog(file, f)

            for row in jsonStream:
                progressLog.onRow()

                if not row.get("is_self"):
                    continue

                author = row.get("author")
                if author not in candidate_set:
                    continue

                info = author_info[author]
                if info["disqualified"]:
                    continue

                subreddit = (row.get("subreddit") or "").strip()

                # 条件1：在 mental-health subreddit 发过 post
                if is_mental_health_subreddit(subreddit):
                    info["disqualified"] = True
                    info["reason"] = f"mental-health subreddit: {subreddit}"
                    info["posts"].clear()
                    continue

                analysis_text = get_post_analysis_text(row)

                # 条件2：文本里出现 mental-health terminology / diagnosis
                if contains_mental_health_signal(analysis_text):
                    info["disqualified"] = True
                    info["reason"] = "mental-health terminology or diagnosis signal"
                    info["posts"].clear()
                    continue

                body = get_post_body(row)
                if not body:
                    continue

                raw_word_count = len(body.split())
                if raw_word_count < min_words:
                    continue

                

                info["posts"].append({
                    "author": author,
                    "created_utc": row.get("created_utc"),
                    "id": row.get("id"),
                    "subreddit": row.get("subreddit"),
                    "raw_text": body,
                })

    # 每个 author 的 posts 按时间从新到旧排
    for author in author_info:
        author_info[author]["posts"].sort(
            key=lambda x: (x["created_utc"] if x["created_utc"] is not None else -1),
            reverse=True
        )

    return author_info


def build_normal_dataset(candidate_authors, author_info, target_authors=1600):
    """
    PASS 3:
    按 candidate_authors 的顺序，接受合格 author。
    只有当 author 被接受后，才对其 posts 做 cleaning 并加入 normal_posts。
    当 unique accepted authors 达到 target_authors 时停止。
    """
    normal_posts = []
    accepted_authors = []
    rejected_authors = []

    print("\n=== PASS 3: Build normal.json ===")

    for author in candidate_authors:
        info = author_info[author]

        if info["disqualified"]:
            rejected_authors.append({
                "author": author,
                "reason": info["reason"],
            })
            continue

        posts = info["posts"]
        if not posts:
            continue

        author_clean_posts = []

        # 只对最终接受的 author 做 cleaning
        for post in posts:
            raw_text = post.get("raw_text", "")
            if not raw_text:
                continue

            cleaned_text = clean_social_media_text_for_structure(raw_text)
            if not cleaned_text:
                continue

            cleaned_word_count = len(cleaned_text.split())
            if cleaned_word_count == 0:
                continue

            author_clean_posts.append({
                "author": post["author"],
                "created_utc": post["created_utc"],
                "id": post["id"],
                "subreddit": post["subreddit"],
                "text": cleaned_text,
                "word_count": cleaned_word_count,
            })

        # 如果这个 author 的 post 清洗后一个都不剩，就不接受
        if not author_clean_posts:
            continue

        accepted_authors.append({
            "author": author,
            "num_posts": len(author_clean_posts),
        })

        normal_posts.extend(author_clean_posts)

        if len(accepted_authors) >= target_authors:
            print(f"Reached target: {target_authors} unique authors")
            return normal_posts, accepted_authors, rejected_authors

    return normal_posts, accepted_authors, rejected_authors


def save_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def processFolder(path: str):
    files = get_zst_files(path)

    candidate_authors, candidate_meta = discover_candidate_authors(
    files=files,
    min_words=MIN_WORDS,
    max_candidate_authors=MAX_CANDIDATE_AUTHORS
)

    author_info = inspect_candidate_authors(
        files=files,
        candidate_authors=candidate_authors,
        min_words=MIN_WORDS
    )

    normal_posts, accepted_authors, rejected_authors = build_normal_dataset(
    candidate_authors=candidate_authors,
    author_info=author_info,
    target_authors=TARGET_AUTHORS
)

    save_json(normal_posts, OUTPUT_PATH)
    save_json(accepted_authors, "accepted_normal_authors.json")
    save_json(rejected_authors, "rejected_normal_authors.json")

    print("\nSaved:")
    print(f"- {OUTPUT_PATH} ({len(normal_posts)} posts from {len(accepted_authors)} accepted authors)")
    print(f"- accepted_normal_authors.json ({len(accepted_authors)} authors)")
    print(f"- rejected_normal_authors.json ({len(rejected_authors)} authors)")


def main():
    if os.path.isdir(fileOrFolderPath):
        processFolder(fileOrFolderPath)
    else:
        raise ValueError("This pipeline expects a folder containing 2025 .zst files")

    print("Done :>")


if __name__ == "__main__":
    main()