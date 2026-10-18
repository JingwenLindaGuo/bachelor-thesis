import json
from pathlib import Path

import pytest

import processFiles_depression as dep


def test_clean_text_for_diagnosis_detection_basic():
    text = "I've been diagnosed with depression!!! See https://example.com :)"
    cleaned = dep.clean_text_for_diagnosis_detection(text)
    assert "i have been diagnosed with depression" in cleaned
    assert "url" in cleaned
    assert "!!!" not in cleaned


def test_regex_prefilter_positive():
    text = "I was formally diagnosed with depression last year and still struggle."
    assert dep.regex_prefilter_diagnosis(text) is True


def test_regex_prefilter_negative_third_person():
    text = "My sister was diagnosed with depression."
    assert dep.regex_prefilter_diagnosis(text) is False


def test_regex_prefilter_negative_negation():
    text = "I was never diagnosed with depression."
    assert dep.regex_prefilter_diagnosis(text) is False


def test_process_record_regex_only_success():
    record = {
        "id": "001",
        "text": "I was diagnosed with depression by my therapist and I still struggle."
    }
    result = dep.process_record(record, use_llm=False, min_words=0)
    assert result is not None
    assert result["label"] == "depressed"
    assert result["detection_method"] == "regex"
    assert "cleaned_text" in result


def test_process_record_respects_min_words():
    record = {
        "id": "002",
        "text": "I was diagnosed with depression."
    }
    result = dep.process_record(record, use_llm=False, min_words=500)
    assert result is None


def test_process_record_llm_yes(monkeypatch):
    def fake_detect(_text, model="gpt-4o-mini"):
        return "YES", "I was diagnosed with depression by my therapist."

    monkeypatch.setattr(dep, "detect_depression_llm", fake_detect)

    record = {
        "id": "003",
        "text": "I was diagnosed with depression by my therapist and I still struggle."
    }
    result = dep.process_record(record, use_llm=True, min_words=0)
    assert result is not None
    assert result["label"] == "depressed"
    assert result["detection_method"] == "regex+llm"
    assert result["evidence"] == "I was diagnosed with depression by my therapist."


def test_process_record_llm_no(monkeypatch):
    def fake_detect(_text, model="gpt-4o-mini"):
        return "NO", None

    monkeypatch.setattr(dep, "detect_depression_llm", fake_detect)

    record = {
        "id": "004",
        "text": "I was diagnosed with depression years ago."
    }
    result = dep.process_record(record, use_llm=True, min_words=0)
    assert result is None


def test_filter_depressed_instances_keeps_only_positive(monkeypatch):
    def fake_detect(text, model="gpt-4o-mini"):
        if "still struggle" in text:
            return "YES", "evidence"
        return "NO", None

    monkeypatch.setattr(dep, "detect_depression_llm", fake_detect)

    records = [
        {"id": "a", "text": "I was diagnosed with depression and still struggle."},
        {"id": "b", "text": "I was diagnosed with depression years ago."},
        {"id": "c", "text": "My sister was diagnosed with depression."},
    ]

    results = dep.filter_depressed_instances(records, use_llm=True, min_words=0)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_load_json_records_list(tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"text": "abc"}]), encoding="utf-8")
    data = dep.load_json_records(p)
    assert isinstance(data, list)
    assert len(data) == 1


def test_load_json_records_single_object(tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"text": "abc"}), encoding="utf-8")
    data = dep.load_json_records(p)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["text"] == "abc"