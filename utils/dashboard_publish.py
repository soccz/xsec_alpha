"""Publish only explicit xsec files onto the current remote tree, never a user's checkout."""
import os
from pathlib import Path
import subprocess
import tempfile

from utils.logger import logger
from utils.run_lock import run_lock

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "https://github.com/soccz/soccz.github.io.git"
PREFIX = "projects/xsec-alpha"
PAYLOAD_PATHS = tuple(f"{PREFIX}/dashboard/data/{name}.json" for name in ("summary", "history", "accuracy")) + (
    f"{PREFIX}/public_summary.json",
)
ALLOWED_PATHS = {*PAYLOAD_PATHS, f"{PREFIX}/index.html", f"{PREFIX}/dashboard/index.html"}


def publish_files(source, *, paths=PAYLOAD_PATHS, cache=None, remote=REMOTE):
    """CAS-style fast-forward push; a remote race retries from its new parent."""
    source = Path(source)
    cache = Path(cache) if cache is not None else ROOT / "output/dashboard_publication/git"
    if not paths or any(path not in ALLOWED_PATHS for path in paths):
        raise ValueError("Only explicitly allowed xsec files may be published")
    payloads = {path: (source / path).read_bytes() for path in paths}
    cache.mkdir(parents=True, exist_ok=True)

    def git(*args, env=None, input=None, check=True):
        return subprocess.run(["git", "-C", str(cache), *args], input=input,
                              capture_output=True, check=check, timeout=60, env=env)

    if not (cache / "HEAD").exists():
        git("init", "--bare", "--initial-branch=main")
        git("remote", "add", "origin", remote)
        git("config", "user.name", "xsec dashboard publisher")
        git("config", "user.email", "soccz@users.noreply.github.com")
    if git("remote", "get-url", "origin").stdout.decode().strip() != remote:
        raise ValueError("Publication cache has an unexpected remote")
    for _ in range(2):
        # A depth-one fetch may hide ancestry even for a normal remote advance.
        # Refresh only this disposable tracking ref; the remote push stays non-force.
        git("fetch", "--depth=1", "--filter=blob:none", "origin", "+refs/heads/main:refs/remotes/origin/main")
        parent = git("rev-parse", "refs/remotes/origin/main").stdout.decode().strip()
        with tempfile.TemporaryDirectory(dir=cache, prefix="index-") as folder:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(folder) / "index")}
            git("read-tree", parent, env=env)
            for path, payload in payloads.items():
                blob = git("hash-object", "-w", "--stdin", input=payload).stdout.decode().strip()
                git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
            tree = git("write-tree", env=env).stdout.decode().strip()
            if tree == git("rev-parse", f"{parent}^{{tree}}").stdout.decode().strip():
                return True
            commit = git("commit-tree", tree, "-p", parent, "-m", "xsec-alpha: publish latest verified state").stdout.decode().strip()
        # No --force: a concurrently advanced remote cannot be overwritten.
        if git("push", "origin", f"{commit}:refs/heads/main", check=False).returncode == 0:
            return True
    return False


def publish_dashboard(push=True):
    from utils.dashboard_export import PIN_DEFAULT, export_to
    from utils.experiment_supervisor import _write
    from utils.prospective import _utc

    folder = ROOT / "output/dashboard_publication"
    try:
        with run_lock("dashboard_publish", lock_dir=str(ROOT / "logs/locks"), timeout_sec=90, exit_code=75):
            source = folder / "payloads"
            target = source / PREFIX
            export_to(target / "dashboard/data", PIN_DEFAULT, public_target=target / "public_summary.json")
            success = publish_files(source) if push else True
            if push:
                _write(folder / "status.json", {"checked_at": _utc().isoformat(),
                                                "status": "published" if success else "failed"})
            return success
    except (Exception, SystemExit) as exc:
        logger.warning("Dashboard publication failed: %s", type(exc).__name__)
        _write(folder / "status.json", {"checked_at": _utc().isoformat(), "status": "failed",
                                        "error_type": type(exc).__name__})
        return False
