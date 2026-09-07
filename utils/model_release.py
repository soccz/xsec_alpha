"""Publish model/report sets with a recoverable journal and reader exclusion."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

from utils.run_lock import run_lock
from utils.logger import logger

ROOT = Path(__file__).resolve().parent.parent
_owned = threading.local()


def artifact_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _replace_from(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".release.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_journal(path: Path, document: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(document, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _recover(root: Path) -> None:
    journal = root / "output" / "model_release_pending.json"
    if not journal.exists():
        return
    document = json.loads(journal.read_text())
    if document["state"] == "prepared":
        for entry in document["entries"]:
            target = (root / entry["target"]).resolve()
            backup = (root / entry["backup"]).resolve()
            target.relative_to(root)
            backup.relative_to(root)
            if entry["existed"]:
                _replace_from(backup, target)
            else:
                target.unlink(missing_ok=True)
    elif document["state"] != "committed":
        raise ValueError("Unknown model release journal state")
    journal.unlink()


@contextmanager
def model_release_guard(root: Path = ROOT):
    """Readers recover interrupted publication before using production files."""
    root = Path(root).resolve()
    owned = getattr(_owned, "roots", set())
    if root in owned:
        yield
        return
    with run_lock("model_release", lock_dir=str(root / "logs" / "locks"),
                  timeout_sec=600, exit_code=os.EX_TEMPFAIL):
        _owned.roots = owned | {root}
        try:
            _recover(root)
            yield
        finally:
            _owned.roots = owned


def publish_release(staged: dict[Path, Path], version: str, root: Path = ROOT) -> None:
    """Archive old targets, publish the complete staged set, roll back on failure."""
    root = Path(root).resolve()
    with model_release_guard(root):
        archive = root / "models" / "archive" / f"release_{version}"
        archive.mkdir(parents=True, exist_ok=False)
        entries = []
        for index, (target, source) in enumerate(staged.items()):
            target = Path(target).resolve()
            target.relative_to(root)
            if not Path(source).is_file():
                raise FileNotFoundError(source)
            backup = archive / f"{index}_{target.name}"
            existed = target.exists()
            if existed:
                shutil.copy2(target, backup)
            entries.append({"target": str(target.relative_to(root)),
                            "backup": str(backup.relative_to(root)), "existed": existed})
        journal = root / "output" / "model_release_pending.json"
        journal.parent.mkdir(parents=True, exist_ok=True)
        document = {"state": "prepared", "version": version, "entries": entries}
        _write_journal(journal, document)
        try:
            for target, source in staged.items():
                _replace_from(Path(source), Path(target))
            _write_journal(journal, {**document, "state": "committed"})
        except BaseException:
            _recover(root)
            raise
        try:
            journal.unlink()
        except OSError:
            logger.warning("Release committed; journal cleanup deferred to next reader")
