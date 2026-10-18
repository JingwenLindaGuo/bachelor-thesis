# -*- coding: utf-8 -*-

import json
import random
from typing import Any, Dict, List, Union

import numpy as np
import speechgraph


def compute_speechgraph_basic(text: str) -> Dict[str, Any]:
    """
    Compute raw speech graph features using speechgraph.naiveGraph().
    Returns at least LCC and LSC if successful.
    """
    if not text or not text.strip():
        return {
            "LCC": None,
            "LSC": None,
            "number_of_edges": None,
            "degree_average": None,
            "degree_std": None,
        }

    try:
        g = speechgraph.naiveGraph()
        res = g.analyzeText(text)

        return {
            "LCC": res.get("LCC"),
            "LSC": res.get("LSC"),
            "number_of_edges": res.get("number_of_edges"),
            "degree_average": res.get("degree_average"),
            "degree_std": res.get("degree_std"),
        }

    except Exception as e:
        print(f"[Warning] speechgraph failed: {e}")
        return {
            "LCC": None,
            "LSC": None,
            "number_of_edges": None,
            "degree_average": None,
            "degree_std": None,
        }


def randomize_text_words(text: str) -> str:
    """
    Shuffle word order for random-normalised graph features.
    """
    words = text.split()
    if len(words) < 2:
        return text

    words = words.copy()
    random.shuffle(words)
    return " ".join(words)


def compute_speechgraph_with_random_norm(text: str, n_iter: int = 20) -> Dict[str, Any]:
    """
    Compute:
    - LCC
    - LSC
    - LCCr
    - LSCr

    LCCr = LCC_real / mean(LCC_random)
    LSCr = LSC_real / mean(LSC_random)
    """
    raw = compute_speechgraph_basic(text)

    LCC_real = raw["LCC"]
    LSC_real = raw["LSC"]

    if LCC_real is None or LSC_real is None:
        raw["LCCr"] = None
        raw["LSCr"] = None
        return raw

    words = text.split()
    if len(words) < 5:
        raw["LCCr"] = None
        raw["LSCr"] = None
        return raw

    rand_lcc = []
    rand_lsc = []

    for _ in range(n_iter):
        rand_text = randomize_text_words(text)
        rand_res = compute_speechgraph_basic(rand_text)

        if rand_res["LCC"] is not None:
            rand_lcc.append(rand_res["LCC"])
        if rand_res["LSC"] is not None:
            rand_lsc.append(rand_res["LSC"])

    if len(rand_lcc) == 0 or np.mean(rand_lcc) == 0:
        raw["LCCr"] = None
    else:
        raw["LCCr"] = LCC_real / float(np.mean(rand_lcc))

    if len(rand_lsc) == 0 or np.mean(rand_lsc) == 0:
        raw["LSCr"] = None
    else:
        raw["LSCr"] = LSC_real / float(np.mean(rand_lsc))

    return raw


def append_speechgraph_features(record: Dict[str, Any], n_iter: int = 20) -> Dict[str, Any]:
    text = record.get("text", "")
    features = compute_speechgraph_with_random_norm(text, n_iter=n_iter)

    record.update(features)
    return record


def process_json_data(data: Union[List[Dict[str, Any]], Dict[str, Any]], n_iter: int = 20):
    if isinstance(data, list):
        return [
            append_speechgraph_features(item, n_iter=n_iter) if isinstance(item, dict) else item
            for item in data
        ]

    if isinstance(data, dict):
        return append_speechgraph_features(data, n_iter=n_iter)

    raise ValueError("Input JSON must be a list of objects or a single object.")


def main():
    input_json_path = r"c:\Users\sem91\Documents\GitHub\NLP_psychosis\examples\input.json"
    output_json_path = r"c:\Users\sem91\Documents\GitHub\NLP_psychosis\examples\output_with_speechgraph.json"

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = process_json_data(data, n_iter=20)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Done. Output saved to: {output_json_path}")


if __name__ == "__main__":
    main()