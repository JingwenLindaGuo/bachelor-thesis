import json
import sys
from pathlib import Path

import pytest
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_extraction_pipeline as fep


def test_normalize_text_handles_none_and_whitespace():
    assert fep.normalize_text(None) == ""
    assert fep.normalize_text("  hello   world \n") == "hello world"
    assert fep.normalize_text(123) == "123"


def test_rough_sentence_count_basic():
    text = "This is one sentence. This is another!\nAnd a third?"
    assert fep.rough_sentence_count(text) == 3


def test_require_valid_feature_dict_accepts_dict():
    result = {"a": 1}
    assert fep.require_valid_feature_dict(result, "dummy") == result


def test_require_valid_feature_dict_rejects_none():
    with pytest.raises(fep.FeatureExtractionError):
        fep.require_valid_feature_dict(None, "dummy")


def test_require_valid_feature_dict_rejects_non_dict():
    with pytest.raises(fep.FeatureExtractionError):
        fep.require_valid_feature_dict(["not", "a", "dict"], "dummy")


def test_atomic_save_json_writes_file(tmp_path):
    output_path = tmp_path / "out.json"
    data = {"x": 1, "y": [1, 2, 3]}

    fep.atomic_save_json(data, output_path)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == data


def test_extract_ambiguous_pronoun_feature_success(monkeypatch):
    class DummyPred:
        pass

    class DummyCorefModel:
        def predict(self, texts):
            assert texts == ["He said he was late."]
            return [DummyPred()]

    monkeypatch.setattr(fep, "get_coref_model", lambda: DummyCorefModel())
    monkeypatch.setattr(
        fep,
        "compute_ambiguous_pronoun_features_from_pred",
        lambda pred: {"ambiguous_pronoun_count": 2},
    )

    result = fep.extract_ambiguous_pronoun_feature("He said he was late.")
    assert result == {"ambiguous_pronoun_count": 2}


def test_extract_ambiguous_pronoun_feature_empty_text():
    with pytest.raises(fep.FeatureExtractionError):
        fep.extract_ambiguous_pronoun_feature("")


def test_extract_all_features_success(monkeypatch):
    monkeypatch.setattr(
        fep, "extract_basic_features",
        lambda text: {
            "word_count": 10,
            "sentence_count": 2,
            "mean_sentence_length": 5.0,
        }
    )
    monkeypatch.setattr(
        fep, "extract_semantic_features",
        lambda text: {
            "semantic_coherence": 0.5,
            "maximum_similarity": 0.8,
        }
    )
    monkeypatch.setattr(
        fep, "extract_lexical_features",
        lambda text: {
            "definite_articles": 1,
            "indefinite_articles": 2,
            "pronouns": 3,
            "first_person_singular": 1,
            "first_person_plural": 0,
            "second_person": 1,
            "third_person": 1,
        }
    )
    monkeypatch.setattr(
        fep, "compute_coherence_features",
        lambda text: {
            "noun_overlap": 0.1,
            "pronoun_overlap": 0.2,
            "content_overlap": 0.3,
        }
    )
    monkeypatch.setattr(
        fep, "compute_speechgraph_with_random_norm",
        lambda text, n_iter=20: {
            "LCC": 10,
            "LSC": 9,
            "number_of_edges": 11,
            "degree_average": 2.0,
            "degree_std": 1.0,
            "LCCr": 1.1,
            "LSCr": 0.9,
        }
    )
    monkeypatch.setattr(
        fep, "extract_ambiguous_pronoun_feature",
        lambda text: {"ambiguous_pronoun_count": 0}
    )

    text = "This is sentence one. This is sentence two."
    result = fep.extract_all_features(text)

    assert result["word_count"] == 10
    assert result["semantic_coherence"] == 0.5
    assert result["definite_articles"] == 1
    assert result["noun_overlap"] == 0.1
    assert result["LCC"] == 10
    assert result["ambiguous_pronoun_count"] == 0


def test_extract_all_features_rejects_too_few_sentences():
    with pytest.raises(fep.FeatureExtractionError):
        fep.extract_all_features("Only one sentence")


def test_extract_all_features_raises_when_one_submodule_fails(monkeypatch):
    monkeypatch.setattr(
        fep, "extract_basic_features",
        lambda text: {
            "word_count": 10,
            "sentence_count": 2,
            "mean_sentence_length": 5.0,
        }
    )
    monkeypatch.setattr(
        fep, "extract_semantic_features",
        lambda text: (_ for _ in ()).throw(RuntimeError("semantic broken"))
    )

    with pytest.raises(fep.FeatureExtractionError):
        fep.extract_all_features("This is sentence one. This is sentence two.")


def test_process_record_success(monkeypatch):
    original = {
        "id": "001",
        "text": "This is sentence one. This is sentence two.",
        "meta": {"group": "test"},
    }

    monkeypatch.setattr(
        fep,
        "extract_all_features",
        lambda text, verbose=False: {"word_count": 10, "sentence_count": 2},
    )

    result = fep.process_record(original, verbose=False)

    assert result is not None
    assert result["id"] == "001"
    assert result["features"]["word_count"] == 10

    # 确保没有原地修改输入
    assert "features" not in original


def test_process_record_returns_none_when_text_missing():
    record = {"id": "001", "text": ""}
    result = fep.process_record(record, verbose=False)
    assert result is None


def test_process_record_returns_none_when_extraction_fails(monkeypatch):
    record = {"id": "001", "text": "This is sentence one. This is sentence two."}

    def broken_extract(text, verbose=False):
        raise fep.FeatureExtractionError("boom")

    monkeypatch.setattr(fep, "extract_all_features", broken_extract)

    result = fep.process_record(record, verbose=False)
    assert result is None


def test_process_json_list_keeps_only_successful_records(monkeypatch):
    data = [
        {"id": "001", "text": "A. B."},
        {"id": "002", "text": "C. D."},
        {"id": "003", "text": "E. F."},
    ]

    def fake_process_record(record, text_field="text", idx=0, total=1, verbose=True):
        if record["id"] == "002":
            return None
        return {**record, "features": {"ok": True}}

    monkeypatch.setattr(fep, "process_record", fake_process_record)

    result = fep.process_json(data, verbose=False)

    assert isinstance(result, list)
    assert len(result) == 2
    assert [r["id"] for r in result] == ["001", "003"]


def test_process_json_single_dict(monkeypatch):
    data = {"id": "001", "text": "A. B."}

    monkeypatch.setattr(
        fep,
        "process_record",
        lambda record, text_field="text", idx=0, total=1, verbose=True: {
            **record, "features": {"ok": True}
        },
    )

    result = fep.process_json(data, verbose=False)
    assert result["id"] == "001"
    assert result["features"]["ok"] is True


def test_process_json_invalid_input():
    with pytest.raises(ValueError):
        fep.process_json("not a dict or list", verbose=False)


def test_process_json_checkpoint_save(tmp_path, monkeypatch):
    data = [
        {"id": "001", "text": "A. B."},
        {"id": "002", "text": "C. D."},
    ]
    output_path = tmp_path / "checkpoint.json"

    monkeypatch.setattr(
        fep,
        "process_record",
        lambda record, text_field="text", idx=0, total=1, verbose=True: {
            **record, "features": {"ok": True}
        },
    )

    fep.process_json(
        data,
        output_path=output_path,
        save_every=1,
        verbose=False,
    )

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert saved[0]["features"]["ok"] is True


def test_build_arg_parser_parses_required_args():
    parser = fep.build_arg_parser()
    args = parser.parse_args([
        "--input", "in.json",
        "--output", "out.json",
    ])

    assert args.input == "in.json"
    assert args.output == "out.json"
    assert args.text_field == "text"
    assert args.save_every == 5