import re
from typing import Dict


def tokenize(text: str):
    """
    Simple tokenizer: lowercase + keep words only
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def extract_lexical_features(text: str) -> Dict[str, int]:
    tokens = tokenize(text)
 
    definite_articles = {"the"}
    indefinite_articles = {"a", "an"}

    first_person_singular = {"i", "me", "my", "mine", "myself"}
    first_person_plural = {"we", "us", "our", "ours", "ourselves"}
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    third_person = {
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "it", "its", "itself"
    }

    pronouns_all = (
        first_person_singular
        | first_person_plural
        | second_person
        | third_person
    )
 
    features = {
        "definite_articles": 0,
        "indefinite_articles": 0,
        "pronouns": 0,
        "first_person_singular": 0,
        "first_person_plural": 0,
        "second_person": 0,
        "third_person": 0,
    }
 
    for token in tokens:
        if token in definite_articles:
            features["definite_articles"] += 1

        if token in indefinite_articles:
            features["indefinite_articles"] += 1

        if token in pronouns_all:
            features["pronouns"] += 1

        if token in first_person_singular:
            features["first_person_singular"] += 1

        if token in first_person_plural:
            features["first_person_plural"] += 1

        if token in second_person:
            features["second_person"] += 1

        if token in third_person:
            features["third_person"] += 1

    return features