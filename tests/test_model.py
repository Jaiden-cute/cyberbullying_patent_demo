from dataclasses import replace
import json

import pytest

from src.config import Settings
from src.model_runtime import ROOT, HeadPayload, MaskHead
from src.pipeline import run_pipeline
from src.runtime import load_runtime
from src.training_data import TRAINING_SOURCE, read_examples, split_statistics, validate_splits

DATA = ROOT / "data"


def test_synthetic_csv_has_disjoint_scenarios_and_all_labels():
    train, test = (read_examples(DATA / f"synthetic_masking_{split}.csv") for split in ("train", "test"))
    validate_splits(train, test)
    assert len(train) == 64 and len(test) == 32
    assert split_statistics(train)["negative_rows"] == 16
    assert split_statistics(test)["multi_label_rows"] == 10
    copied = replace(train[0], sample_id="leaked", scenario_group="different-id")
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(train, [*test, copied])


def test_trained_request_reports_missing_model_and_strict_status(settings, messages, tmp_path):
    raw = settings.model_dump()
    raw["engineering_demo_default"]["models"]["embedding_path"] = str(tmp_path / "missing")
    runtime = load_runtime(Settings.model_validate(raw), requested="trained", strict=True, use_qwen=False)
    run = run_pipeline(messages, settings, runtime=runtime)
    row = run["events"][0]
    assert run["run_mode"] == "UNAVAILABLE"
    assert row["candidates"][0].S2.A is None
    assert row["candidates"][0].S2.feature_sources["S"] == "sentence_bert_unavailable"
    assert row["S6"].U is None


@pytest.fixture(scope="module")
def real_runtime():
    from src.config import load_settings

    settings = load_settings(ROOT / "config.yaml")
    model_settings = settings.engineering_demo_default.models
    pytest.importorskip("sentence_transformers")
    if not (ROOT / model_settings.embedding_path / "model.safetensors").is_file():
        pytest.skip("provision the pinned Sentence-BERT artifact first")
    runtime = load_runtime(settings, requested="trained", use_qwen=False)
    assert runtime.encoder is not None, runtime.encoder_status
    assert runtime.head is not None, runtime.head_status
    return runtime


@pytest.mark.model
def test_saved_classifier_reproduces_real_holdout_probabilities(real_runtime):
    encoder, head = real_runtime.encoder, real_runtime.head
    cases = read_examples(DATA / "synthetic_masking_test.csv")
    training_report = json.loads((ROOT / "reports/mask_head_training.json").read_text(encoding="utf-8"))
    saved = {row["sample_id"]: row["p"] for row in training_report["prediction_cases"]}
    assert head.provenance["training_data_source"] == TRAINING_SOURCE
    assert training_report["final_training_bce"] < training_report["initial_training_bce"]
    assert head.payload.embedding_dimension == len(encoder.embed(cases[0].reply))
    for example in (cases[0], cases[3], cases[8], cases[-1]):
        p = head.predict_from_embeddings(encoder.embed(example.event_summary), encoder.embed(example.reply))
        assert len(p) == 8 and all(0 <= value <= 1 for value in p)
        assert p == pytest.approx(saved[example.sample_id], abs=1e-6)
    assert saved[cases[0].sample_id] != saved[cases[3].sample_id]


@pytest.mark.model
def test_classifier_rejects_tampered_weights(real_runtime, tmp_path):
    payload = real_runtime.head.payload.model_dump(mode="json")
    payload["W"][0][0] += 0.1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        MaskHead.load(path, real_runtime.encoder)


@pytest.mark.model
def test_actual_model_in_hybrid_pipeline(real_runtime, settings, messages, features):
    report = run_pipeline(messages, settings, features=features, runtime=real_runtime)
    assert report["run_mode"] == "HYBRID"
    reply = report["events"][0]["candidates"][0]
    assert reply.S2.feature_sources["S"] == "sentence_bert"
    assert reply.S2.feature_sources["D"] == "heuristic_fallback"
    assert reply.S3_masking.non_model_score is None and len(reply.S3_masking.p) == 8
    assert reply.S3_masking.training_data_source == TRAINING_SOURCE
    assert report["events"][0]["S4"]["pattern_source"] == reply.S3_masking.model_version
    assert report["events"][-1]["S5"].P_status == "available"
    assert report["events"][-1]["S6"].support_source == "safe_template"


@pytest.mark.model
def test_offline_association_calibration_learns_from_fixture_without_editing_configuration(real_runtime, tmp_path):
    from scripts.calibrate_association import calibrate

    labeled = tmp_path / "association_fixture.csv"
    labeled.write_text("L,Q,S,T,D,G,label\n1,0,0.8,0.9,0,0,1\n0,0,-0.5,0.1,0,0,0\n", encoding="utf-8")
    config_path = ROOT / "config.yaml"
    original = config_path.read_bytes()
    result = calibrate(labeled, config_path, epochs=5, learning_rate=0.05)
    assert result["sample_count"] == 2
    assert result["association_weights"]["b1"] > 1.5
    assert config_path.read_bytes() == original
