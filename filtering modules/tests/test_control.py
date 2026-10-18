import json
from pathlib import Path

import pytest

import processFiles_control as ctrl


def make_long_neutral_text(n=520):
    return " ".join(["neutral"] * n)


def test_clean_text_for_structure_basic():
    text = "I'm happy!!! Visit https://example.com :)"
    cleaned = ctrl.clean_text_for_structure(text)
    assert "i am happy" in cleaned
    assert "url" in cleaned
    assert "!!!" not in cleaned


def test_contains_mental_health_signal_positive_diagnosis():
    text = "I was diagnosed with depression by my therapist."
    assert ctrl.contains_mental_health_signal(text) is True


def test_contains_mental_health_signal_positive_term():
    text = "I have severe anxiety and panic attacks."
    assert ctrl.contains_mental_health_signal(text) is True


def test_contains_mental_health_signal_negative():
    text = "I moved to a new city and enjoy reading on weekends."
    assert ctrl.contains_mental_health_signal(text) is False


def test_process_control_record_success():
    record = {
        "id": "001",
        "text": make_long_neutral_text(520)
    }
    result = ctrl.process_control_record(record, min_words=500)
    assert result is not None
    assert result["label"] == "control"
    assert result["filter_method"] == "lexical_exclusion"


def test_process_control_record_rejects_signal():
    record = {
        "id": "002",
        "text": "I was diagnosed with depression by my therapist."
    }
    result = ctrl.process_control_record(record, min_words=0)
    assert result is None


def test_process_control_record_rejects_short_text():
    record = {
        "id": "003",
        "text": "I enjoy reading."
    }
    result = ctrl.process_control_record(record, min_words=500)
    assert result is None


def test_group_records_by_entity():
    records = [
        {"entity_id": "u1", "text": "a"},
        {"entity_id": "u1", "text": "b"},
        {"entity_id": "u2", "text": "c"},
    ]
    grouped = ctrl.group_records_by_entity(records, "entity_id")
    assert set(grouped.keys()) == {"u1", "u2"}
    assert len(grouped["u1"]) == 2


def test_entity_has_mental_health_signal():
    records = [
        {"entity_id": "u1", "text": make_long_neutral_text(20)},
        {"entity_id": "u1", "text": "I was diagnosed with depression."},
    ]
    assert ctrl.entity_has_mental_health_signal(records) is True


def test_filter_control_instances_record_level():
    records = [
        {"id": "a", "text": make_long_neutral_text(520)},
        {"id": "b", "text": "I was diagnosed with depression."},
    ]
    retained, accepted, rejected = ctrl.filter_control_instances(
        records,
        min_words=500,
        entity_field=None,
    )
    assert len(retained) == 1
    assert retained[0]["id"] == "a"
    assert accepted == []
    assert rejected == []


def test_filter_control_instances_entity_level():
    records = [
        {"id": "a1", "entity_id": "u1", "text": make_long_neutral_text(520)},
        {"id": "a2", "entity_id": "u1", "text": make_long_neutral_text(530)},
        {"id": "b1", "entity_id": "u2", "text": make_long_neutral_text(520)},
        {"id": "b2", "entity_id": "u2", "text": "I was diagnosed with depression."},
    ]

    retained, accepted, rejected = ctrl.filter_control_instances(
        records,
        entity_field="entity_id",
        min_words=500,
        exclude_entity_if_any_signal=True,
    )

    retained_ids = {r["id"] for r in retained}
    assert retained_ids == {"a1", "a2"}
    assert accepted == [{"entity_id": "u1", "num_records": 2}]
    assert rejected == [{"entity_id": "u2", "reason": "mental health signal detected in entity history"}]


def test_load_json_records_single_object(tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"text": "abc"}), encoding="utf-8")
    data = ctrl.load_json_records(p)
    assert isinstance(data, list)
    assert len(data) == 1