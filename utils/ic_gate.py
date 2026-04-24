"""IC-based execution gate (automates CLAUDE.md §7).

Reads the last-N IC readings per side from output/ic_history*.json and returns
a gate decision that fetch_and_rank honors. Persists the decision to
output/gate_state.json so health_snapshot can display it.

States (most → least severe):
  LIQUIDATE  3+ consecutive IC<0.03  → emit no picks; require human ack
  FREEZE     2  consecutive IC<0.03  → force watch-only on that side
  WARN       any recent IC<0.05      → proceed but surface it
  OK         last-3 all >= 0.05

Conservative failure mode: missing history returns WARN, not OK.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

GateStatus = Literal["OK", "WARN", "FREEZE", "LIQUIDATE"]
ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILES = {
    "short": ROOT / "output" / "ic_history.json",
    "long":  ROOT / "output" / "ic_history_long.json",
}
STATE_FILE = ROOT / "output" / "gate_state.json"

FREEZE_THRESHOLD = 0.03
WARN_THRESHOLD   = 0.05
LOOKBACK = 3


@dataclass
class SideGate:
    side: str
    status: GateStatus
    reason: str
    last_ic: float | None
    ic_tail: list[float]
    watch_only: bool         # True → force side to watch-only
    block: bool              # True → emit no picks at all

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def _load_ics(path: Path) -> list[float]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [float(r["ic"]) for r in data if "ic" in r and r["ic"] is not None]
    except Exception:
        return []


def _decide(ics: list[float]) -> tuple[GateStatus, str]:
    if not ics:
        return "WARN", "no IC history — treating as degraded"
    trailing_low = 0
    for v in reversed(ics):
        if v < FREEZE_THRESHOLD:
            trailing_low += 1
        else:
            break
    if trailing_low >= 3:
        return "LIQUIDATE", f"{trailing_low} consecutive IC<{FREEZE_THRESHOLD} — §7 liquidate"
    if trailing_low >= 2:
        return "FREEZE", f"{trailing_low} consecutive IC<{FREEZE_THRESHOLD} — §7 freeze"
    last = ics[-LOOKBACK:]
    if any(v < FREEZE_THRESHOLD for v in last):
        return "WARN", f"last-{len(last)} contains IC<{FREEZE_THRESHOLD}"
    if any(v < WARN_THRESHOLD for v in last):
        return "WARN", f"last-{len(last)} contains IC<{WARN_THRESHOLD}"
    return "OK", f"last-{len(last)} all >= {WARN_THRESHOLD}"


def evaluate_side(side: str) -> SideGate:
    path = HISTORY_FILES[side]
    ics = _load_ics(path)
    status, reason = _decide(ics)
    return SideGate(
        side=side,
        status=status,
        reason=reason,
        last_ic=ics[-1] if ics else None,
        ic_tail=ics[-LOOKBACK:] if ics else [],
        watch_only=(status in ("FREEZE", "LIQUIDATE")),
        block=(status == "LIQUIDATE"),
    )


def evaluate_all(persist: bool = True) -> dict[str, SideGate]:
    gates = {side: evaluate_side(side) for side in HISTORY_FILES}
    if persist:
        STATE_FILE.parent.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "gates": {k: v.as_dict() for k, v in gates.items()},
            },
            indent=2,
            default=str,
        ))
    return gates


def render(gates: dict[str, SideGate]) -> str:
    lines = ["IC gate state:"]
    for side, g in gates.items():
        tail = ", ".join(f"{v:+.4f}" for v in g.ic_tail) if g.ic_tail else "—"
        flags = []
        if g.block: flags.append("BLOCK")
        if g.watch_only and not g.block: flags.append("WATCH")
        flag_str = f" [{' '.join(flags)}]" if flags else ""
        lines.append(f"  {side:6s} {g.status:9s}{flag_str}  tail=[{tail}]  {g.reason}")
    return "\n".join(lines)


def main():
    gates = evaluate_all(persist=True)
    print(render(gates))
    # Exit 1 if anything worse than WARN
    worst = max(
        (g.status for g in gates.values()),
        key=lambda s: ["OK", "WARN", "FREEZE", "LIQUIDATE"].index(s),
    )
    import sys
    sys.exit(0 if worst in ("OK", "WARN") else 1)


if __name__ == "__main__":
    main()
