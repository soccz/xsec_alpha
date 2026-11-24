import json
from datetime import datetime, timezone

from scripts import check_dashboard_delivery as delivery


def test_receipts_and_price_audit_are_private_optional_exports(tmp_path, monkeypatch):
    from utils import dashboard_export as export

    output = tmp_path / "output"
    monkeypatch.setattr(export, "OUTPUT_DIR", output)
    for key, relative in (
        ("basket_coverage", "basket_coverage/summary.json"),
        ("dashboard_delivery", "dashboard_publication/delivery.json"),
    ):
        assert export.build_summary_payload()[key] is None
        dest = output / relative
        dest.parent.mkdir(parents=True)
        dest.write_text('{"checked_at":"fixture","internal_probe":"private-marker"}')
        assert export.build_summary_payload()[key]["internal_probe"] == "private-marker"
    public = json.dumps(export.build_public_summary_payload())
    assert (
        "private-marker" not in public
        and "basket_coverage" not in public
        and "dashboard_delivery" not in public
    )


def test_failure_recovery_and_supersession(tmp_path):
    target = tmp_path / "payloads" / delivery.PREFIX
    (target / "dashboard/data").mkdir(parents=True)
    bodies = {delivery.FILES[1]: b"encrypted"}
    bodies[delivery.FILES[0]] = json.dumps(
        {
            "asof": "version1",
            "operating_receipt": {"summary_sha256": delivery.digest(b"encrypted")},
        }
    ).encode()
    for name, body in bodies.items():
        (target / name).write_bytes(body)
    status = tmp_path / "status.json"
    status.write_text('{"status":"published"}')
    assert (
        delivery.verify(tmp_path, lambda *args: b"old")["status"]
        == "served_other_generation"
    )
    good = delivery.verify(tmp_path, lambda name, _: bodies[name])
    assert good["status"] == "verified"

    def changing(name, _):
        status.write_text('{"status":"published","checked_at":"newer"}')
        return bodies[name]

    assert delivery.verify(tmp_path, changing)["status"] == "local_generation_changed"

    def timeout(*_):
        raise TimeoutError()

    unavailable = delivery.verify(tmp_path, timeout)
    assert unavailable["status"] == "unverified"
    assert unavailable["last_verified"] == good["last_verified"]
    status.write_text('{"status":"failed"}')
    assert delivery.verify(tmp_path, timeout)["status"] == "no_confirmed_push"


def test_file_confirmation_round_and_freshness_are_independent(tmp_path):
    folder = tmp_path / "dashboard_publication"
    folder.mkdir()
    now = datetime(2026, 11, 24, 5, 45, tzinfo=timezone.utc)
    report = {
        "run_id": "A",
        "data_asof": "2026-11-24T05:00:00Z",
        "telegram": {"state": "sent", "acknowledged_at": "proof"},
    }
    latest = tmp_path / "latest_operator_report.json"
    latest.write_text(json.dumps(report))
    witness = folder / "report_witness.json"
    witness.write_text(
        json.dumps(
            {"report": delivery.report_identity(report), "summary_sha256": "abc"}
        )
    )
    result = delivery.round_check(folder, "abc", now)
    assert (
        result["match"] == "same"
        and result["freshness"] == "current"
        and result["same_ack"]
        and result["same_picks"]
    )
    latest.write_text(json.dumps(dict(report, ideas=[{"market": "KRW-DIFFERENT"}])))
    assert not delivery.round_check(folder, "abc", now)["same_picks"]
    assert delivery.round_check(folder, "wrong", now)["match"] == "unknown"
    latest.write_text(json.dumps(dict(report, run_id="B")))
    assert delivery.round_check(folder, "abc", now)["match"] == "different"
    latest.write_text(json.dumps(dict(report, data_asof="2026-11-23T23:00:00Z")))
    assert delivery.round_check(folder, "abc", now)["freshness"] == "stale"
    assert (
        delivery.round_check(folder, "abc", now.replace(minute=44))["freshness"]
        == "current"
    )
    latest.write_text(json.dumps(dict(report, data_asof=None)))
    assert delivery.round_check(folder, "abc", now)["freshness"] == "unknown"
