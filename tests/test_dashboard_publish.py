import subprocess

import pytest

from utils.dashboard_publish import PAYLOAD_PATHS, publish_files


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def site(tmp_path):
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    repo = tmp_path / "site"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    data = repo / "projects/xsec-alpha/dashboard/data"
    data.mkdir(parents=True)
    for name in ("summary", "history", "accuracy"):
        (data / f"{name}.json").write_text('{"encrypted": "old"}')
    (data.parent.parent / "public_summary.json").write_text('{}')
    (repo / "unrelated.txt").write_text("original")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "initial")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "main")
    (data / "summary.json").write_text('{"encrypted": "new"}')
    return repo, remote


def test_dashboard_push_never_commits_unrelated_worktree_edits(site, tmp_path):
    repo, remote = site
    (repo / "unrelated.txt").write_text("user edits")
    before = git(repo, "rev-parse", "HEAD")
    assert publish_files(repo, cache=tmp_path / "publisher", remote=str(remote))
    assert git(remote, "show", "main:unrelated.txt") == "original"
    assert (repo / "unrelated.txt").read_text() == "user edits"
    assert git(repo, "rev-parse", "HEAD") == before
    assert "unrelated.txt" in git(repo, "diff", "--name-only")
    assert git(remote, "show", f"main:{PAYLOAD_PATHS[0]}") == '{"encrypted": "new"}'


def test_dashboard_push_leaves_unrelated_staged_files_untouched(site, tmp_path):
    repo, remote = site
    before = git(remote, "rev-parse", "main")
    (repo / "unrelated.txt").write_text("staged user edits")
    git(repo, "add", "unrelated.txt")
    assert publish_files(repo, cache=tmp_path / "publisher", remote=str(remote))
    assert git(repo, "diff", "--cached", "--name-only") == "unrelated.txt"
    assert git(remote, "rev-parse", "main^") == before
    assert git(remote, "show", "main:unrelated.txt") == "original"


def test_diverged_checkout_and_untracked_files_do_not_block_or_leak(site, tmp_path):
    repo, remote = site
    git(repo, "add", "projects/xsec-alpha/dashboard/data/summary.json")
    git(repo, "commit", "-m", "previous pending publish")
    (repo / "private.txt").write_text("must remain local")
    peer = tmp_path / "peer"
    git(tmp_path, "clone", str(remote), str(peer))
    git(peer, "config", "user.name", "Peer")
    git(peer, "config", "user.email", "peer@example.invalid")
    (peer / "unrelated.txt").write_text("remote update")
    git(peer, "add", "unrelated.txt")
    git(peer, "commit", "-m", "unrelated remote update")
    git(peer, "push", "origin", "main")
    head = git(repo, "rev-parse", "HEAD")
    assert publish_files(repo, cache=tmp_path / "publisher", remote=str(remote))
    published = git(remote, "rev-parse", "main")
    assert git(remote, "show", "main:unrelated.txt") == "remote update"
    assert "private.txt" not in git(remote, "ls-tree", "-r", "--name-only", "main")
    assert git(repo, "rev-parse", "HEAD") == head
    assert publish_files(repo, cache=tmp_path / "publisher", remote=str(remote))
    assert git(remote, "rev-parse", "main") == published


def test_publisher_refuses_unscoped_paths(site, tmp_path):
    repo, remote = site
    with pytest.raises(ValueError, match="explicitly allowed"):
        publish_files(repo, paths=["unrelated.txt"], cache=tmp_path / "publisher", remote=str(remote))
