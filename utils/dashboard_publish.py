"""Publish only explicit xsec files onto the current remote tree, never a user's checkout."""
import os
from pathlib import Path
import shutil
import subprocess

from utils.logger import logger
from utils.run_lock import run_lock

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "https://github.com/soccz/soccz.github.io.git"
PREFIX = "projects/xsec-alpha"
PAYLOAD_PATHS = tuple(f"{PREFIX}/dashboard/data/{name}.json" for name in ("summary", "history", "accuracy")) + (
    f"{PREFIX}/public_summary.json",
)
ALLOWED_PATHS = {*PAYLOAD_PATHS, f"{PREFIX}/index.html", f"{PREFIX}/dashboard/index.html"}
# The cache is disposable. Auto-gc detached from short-lived git calls used to die
# mid-run (tmp_pack garbage); too many packs mean it should simply be rebuilt.
PACK_LIMIT = 20
NO_AUTO_GC = ("-c", "gc.auto=0", "-c", "maintenance.auto=false")


def _cache_needs_rebuild(cache):
    pack_dir = Path(cache) / "objects" / "pack"
    if not pack_dir.is_dir():
        return False
    names = os.listdir(pack_dir)
    return any(name.startswith("tmp_pack") for name in names) or sum(name.endswith(".pack") for name in names) > PACK_LIMIT


def _tree_entries(git, tree):
    entries = {}
    for record in git("ls-tree", "-z", tree).stdout.split(b"\0"):
        if record:
            meta, name = record.split(b"\t", 1)
            mode, kind, oid = meta.split(b" ")
            entries[name] = (mode, kind, oid)
    return entries


def _write_tree(git, entries):
    """Serialize a canonical tree object without looking up any referenced object."""
    def order(item):
        name, (_mode, kind, _oid) = item
        return name + (b"/" if kind == b"tree" else b"")

    raw = b"".join((b"40000" if kind == b"tree" else mode) + b" " + name + b"\0" + bytes.fromhex(oid.decode())
                   for name, (mode, kind, oid) in sorted(entries.items(), key=order))
    return git("hash-object", "-t", "tree", "-w", "--stdin", "--literally", input=raw).stdout.strip()


def _rewrite_tree(git, tree, changes):
    """Return the id of `tree` with `changes` ({b"a/b/file": blob_id}) applied as 100644 blobs."""
    entries = _tree_entries(git, tree) if tree else {}
    nested = {}
    for path, blob in changes.items():
        head, sep, rest = path.partition(b"/")
        if sep:
            nested.setdefault(head, {})[rest] = blob
        else:
            entries[head] = (b"100644", b"blob", blob)
    for name, sub in nested.items():
        current = entries.get(name)
        base = current[2] if current and current[1] == b"tree" else None
        entries[name] = (b"040000", b"tree", _rewrite_tree(git, base, sub))
    return _write_tree(git, entries)


def publish_files(source, *, paths=PAYLOAD_PATHS, cache=None, remote=REMOTE):
    """CAS-style fast-forward push; a remote race retries from its new parent."""
    source = Path(source)
    cache = Path(cache) if cache is not None else ROOT / "output/dashboard_publication/git"
    if not paths or any(path not in ALLOWED_PATHS for path in paths):
        raise ValueError("Only explicitly allowed xsec files may be published")
    payloads = {path: (source / path).read_bytes() for path in paths}
    if _cache_needs_rebuild(cache):
        logger.info("Rebuilding dashboard publication cache (pack garbage or too many packs)")
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    def git(*args, env=None, input=None, check=True):
        return subprocess.run(["git", *NO_AUTO_GC, "-C", str(cache), *args], input=input,
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
        blobs = {path.encode(): git("hash-object", "-w", "--stdin", input=payload).stdout.strip()
                 for path, payload in payloads.items()}
        root = git("rev-parse", f"{parent}^{{tree}}").stdout.strip()
        # Rebuild only the trees on the published paths. read-tree/write-tree (and mktree)
        # make git 2.34 lazily fetch every blob of the shared site tree into this blob:none
        # cache (~0.7 GB/day of growth); unchanged entries are kept by id only.
        tree = _rewrite_tree(git, root, blobs)
        if tree == root:
            return True
        commit = git("commit-tree", tree.decode(), "-p", parent, "-m", "xsec-alpha: publish latest verified state").stdout.decode().strip()
        # No --force: a concurrently advanced remote cannot be overwritten.
        # --no-thin: a thin pack would lazily fetch the old payload blobs as delta bases.
        if git("push", "--no-thin", "origin", f"{commit}:refs/heads/main", check=False).returncode == 0:
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
