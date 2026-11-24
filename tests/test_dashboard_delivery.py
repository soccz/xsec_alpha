import json

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
