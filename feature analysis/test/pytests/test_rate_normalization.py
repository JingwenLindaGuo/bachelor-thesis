import pytest

from src.rate_normalization import (
    get_denominator_word_count,
    replace_with_per_100_word_features,
)


def test_get_denominator_word_count_prefers_nested_feature_word_count():
    item = {
        "word_count": 200,
        "features": {"word_count": 100}
    }
    assert get_denominator_word_count(item) == 100.0


def test_get_denominator_word_count_falls_back_to_outer_word_count():
    item = {
        "word_count": 250,
        "features": {}
    }
    assert get_denominator_word_count(item) == 250.0


def test_get_denominator_word_count_returns_none_for_invalid_values():
    item = {
        "word_count": 0,
        "features": {"word_count": -3}
    }
    assert get_denominator_word_count(item) is None


def test_replace_with_per_100_word_features_adds_normalized_and_deletes_raw_counts():
    data = [
        {
            "id": "1",
            "features": {
                "word_count": 100,
                "pronouns": 5,
                "definite_articles": 2,
                "semantic_coherence": 0.4,
            }
        }
    ]

    out = replace_with_per_100_word_features(data)
    feats = out[0]["features"]

    assert feats["pronouns_per_100_words"] == 5.0
    assert feats["definite_articles_per_100_words"] == 2.0
    assert "pronouns" not in feats
    assert "definite_articles" not in feats
    assert feats["semantic_coherence"] == 0.4


def test_replace_with_per_100_word_features_skips_when_features_not_dict():
    data = [{"id": "1", "features": None, "word_count": 100}]
    out = replace_with_per_100_word_features(data)
    assert out == data


def test_replace_with_per_100_word_features_skips_when_denominator_missing():
    data = [
        {
            "id": "1",
            "features": {
                "pronouns": 8
            }
        }
    ]
    out = replace_with_per_100_word_features(data)
    assert "pronouns_per_100_words" not in out[0]["features"]
    assert out[0]["features"]["pronouns"] == 8