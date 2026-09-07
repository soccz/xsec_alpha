import json
import pytest

from scripts import rebuild_calibration, retrain_pipeline
from utils import model_release


@pytest.fixture
def release_files(tmp_path):
    targets = [tmp_path / "models" / "xsec_6h.pkl", tmp_path / "models" / "xsec_12h.pkl",
               tmp_path / "output" / "calibration_sigma.json", tmp_path / "output" / "holdout_report.json"]
    staged = {}
    for i, target in enumerate(targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"old-{i}")
        source = tmp_path / f"staged-{i}"
        source.write_text(f"new-{i}")
        staged[target] = source
    return tmp_path, staged


def test_release_publishes_complete_set_and_archives_previous(release_files):
    root, staged = release_files
    model_release.publish_release(staged, "test", root)
    for i, target in enumerate(staged):
        assert target.read_text() == f"new-{i}"
    archived = root / "models" / "archive" / "release_test"
    assert len(list(archived.iterdir())) == 4
    assert not (root / "output" / "model_release_pending.json").exists()


def test_failure_mid_publication_restores_entire_previous_set(release_files, monkeypatch):
    root, staged = release_files
    original = model_release._replace_from
    calls = 0
    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replacement failure")
        original(source, target)
    monkeypatch.setattr(model_release, "_replace_from", fail_once)
    with pytest.raises(OSError, match="injected"):
        model_release.publish_release(staged, "failed", root)
    for i, target in enumerate(staged):
        assert target.read_text() == f"old-{i}"
    assert not (root / "output" / "model_release_pending.json").exists()


def test_reader_recovers_interrupted_publication(release_files):
    root, staged = release_files
    target = next(iter(staged))
    backup = root / "old-model"
    backup.write_bytes(target.read_bytes())
    target.write_text("partially-published")
    journal = root / "output" / "model_release_pending.json"
    model_release._write_journal(journal, {"state": "prepared", "entries": [
        {"target": str(target.relative_to(root)), "backup": "old-model", "existed": True},
    ]})
    with model_release.model_release_guard(root):
        assert target.read_text() == "old-0"
        # Nested readers in the same thread must not deadlock.
        with model_release.model_release_guard(root):
            assert not journal.exists()


def test_committed_journal_is_not_rolled_back(release_files):
    root, staged = release_files
    journal = root / "output" / "model_release_pending.json"
    model_release._write_journal(journal, {"state": "committed", "entries": []})
    with model_release.model_release_guard(root):
        assert not journal.exists()


def test_committed_release_survives_journal_cleanup_failure(release_files, monkeypatch):
    from pathlib import Path
    root, staged = release_files
    journal = root / "output" / "model_release_pending.json"
    original = Path.unlink
    def fail_cleanup(path, *args, **kwargs):
        if path == journal:
            raise OSError("injected cleanup failure")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_cleanup)
        model_release.publish_release(staged, "cleanup-failure", root)
    assert json.loads(journal.read_text())["state"] == "committed"
    with model_release.model_release_guard(root):
        assert not journal.exists()
        for i, target in enumerate(staged):
            assert target.read_text() == f"new-{i}"


def configure_stage(root, staged, monkeypatch):
    monkeypatch.setattr(retrain_pipeline, "ROOT", root)
    monkeypatch.setattr(retrain_pipeline, "MODELS_DIR", root / "models")
    monkeypatch.setattr(retrain_pipeline, "log", lambda message: None)
    return {6: next(iter(staged.values()))}


def test_calibration_failure_never_changes_production(release_files, monkeypatch):
    root, staged = release_files
    candidates = configure_stage(root, staged, monkeypatch)
    def fail(**kwargs):
        raise RuntimeError("calibration failed")
    monkeypatch.setattr(rebuild_calibration, "build_calibration_document", fail)
    with pytest.raises(RuntimeError, match="calibration failed"):
        retrain_pipeline._stage_and_publish(candidates, "failed-calibration")
    for i, target in enumerate(staged):
        assert target.read_text() == f"old-{i}"


def calibration_for_stage(**kwargs):
    bucket = {"sigma_low": 2, "sigma_high": None, "n": 100, "hit_rate": 0.57,
              "mean_signed_return_pct": 0.7, "std_return_pct": 3.0}
    return {"generated_at": "2026-09-07T00:00Z", "short_6h": [bucket], "long_12h": [bucket]}


def test_staged_model_set_publishes_matching_report(release_files, monkeypatch):
    root, staged = release_files
    candidates = configure_stage(root, staged, monkeypatch)
    monkeypatch.setattr(rebuild_calibration, "build_calibration_document", calibration_for_stage)
    monkeypatch.setattr(retrain_pipeline, "measure_holdout_stats",
                        lambda path, horizon: {"ic": 0.12, "tstat": 3.0, "n_periods": 20})
    retrain_pipeline._stage_and_publish(candidates, "complete")
    assert (root / "models" / "xsec_6h.pkl").read_text() == "new-0"
    assert (root / "models" / "xsec_12h.pkl").read_text() == "old-1"
    report = json.loads((root / "output" / "holdout_report.json").read_text())
    assert report["short_h6"]["hit_2sigma"] == 0.57
    assert report["short_h6"]["sigma_asof"] == "2026-09-07T00:00Z"


def test_report_failure_never_changes_production(release_files, monkeypatch):
    root, staged = release_files
    candidates = configure_stage(root, staged, monkeypatch)
    monkeypatch.setattr(rebuild_calibration, "build_calibration_document", calibration_for_stage)
    monkeypatch.setattr(retrain_pipeline, "measure_holdout_stats",
                        lambda path, horizon: {"ic": float("nan"), "n_periods": 0})
    with pytest.raises(ValueError, match="finite IC"):
        retrain_pipeline._stage_and_publish(candidates, "bad-report")
    for i, target in enumerate(staged):
        assert target.read_text() == f"old-{i}"
