from datetime import datetime, timedelta, timezone

from scripts import notify_unit_failure as nu

NOW = datetime(2026, 10, 5, 3, 0, tzinfo=timezone.utc)


class Sender:
    def __init__(self, ok=True):
        self.ok, self.messages = ok, []

    def __call__(self, text):
        self.messages.append(text)
        return self.ok


def test_one_notice_per_unit_per_quiet_window(tmp_path):
    state, send = tmp_path / "state.json", Sender()
    assert nu.notify("xsec-backup.service", NOW, state, send)
    assert nu.notify("xsec-backup.service", NOW + timedelta(hours=5), state, send)
    assert nu.notify("xsec-deadlines.service", NOW + timedelta(hours=5), state, send)
    assert nu.notify("xsec-backup.service", NOW + timedelta(hours=7), state, send)
    assert [m.split("\n")[1] for m in send.messages] == [
        "xsec-backup.service", "xsec-deadlines.service", "xsec-backup.service"]


def test_failed_delivery_is_not_recorded_so_next_failure_retries(tmp_path):
    state = tmp_path / "state.json"
    assert not nu.notify("xsec-alpha.service", NOW, state, Sender(ok=False))
    send = Sender()
    assert nu.notify("xsec-alpha.service", NOW + timedelta(minutes=1), state, send)
    assert len(send.messages) == 1


def test_unit_name_is_sanitized_and_no_paths_leave_the_server(tmp_path):
    send = Sender()
    nu.notify("bad<b>unit</b>/../../etc", NOW, tmp_path / "s.json", send)
    assert "<b>unit" not in send.messages[0] and "/" not in send.messages[0].split("\n")[1]
