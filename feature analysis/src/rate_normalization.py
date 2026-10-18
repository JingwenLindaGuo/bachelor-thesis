import json
from pathlib import Path

INPUT_FILES = [
    "depression_with_features.json",
    "normal_with_features.json",
]

COUNT_FEATURES_TO_NORMALIZE = [
    "definite_articles",
    "indefinite_articles",
    "pronouns",
    "first_person_singular",
    "first_person_plural",
    "second_person",
    "third_person",
    "ambiguous_pronoun_count",
]


def get_denominator_word_count(item: dict) -> float | None:
    features = item.get("features", {})
    if isinstance(features, dict):
        wc = features.get("word_count")
        if isinstance(wc, (int, float)) and wc > 0:
            return float(wc)

    wc_outer = item.get("word_count")
    if isinstance(wc_outer, (int, float)) and wc_outer > 0:
        return float(wc_outer)

    return None


def replace_with_per_100_word_features(data: list[dict]) -> list[dict]:
    for item in data:
        features = item.get("features", {})
        if not isinstance(features, dict):
            continue

        denom = get_denominator_word_count(item)
        if denom is None or denom <= 0:
            continue
 
        new_features = {}
        keys_to_delete = []

        for feat in COUNT_FEATURES_TO_NORMALIZE:
            value = features.get(feat)
            if isinstance(value, (int, float)):
                new_features[f"{feat}_per_100_words"] = (float(value) / denom) * 100.0
                keys_to_delete.append(feat)
 
        features.update(new_features)
 
        for feat in keys_to_delete:
            del features[feat]

    return data


def main():
    for input_file in INPUT_FILES:
        input_path = Path(input_file)
        output_path = input_path.with_name(input_path.stem + "_with_rates.json")

        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{input_file} 's JSON's top file must be a list。")

        data = replace_with_per_100_word_features(data)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()