import subprocess

import pytest

from scripts.fetch_and_rank import _push_dashboard_to_github


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


def test_dashboard_push_never_commits_unrelated_worktree_edits(site):
    repo, remote = site
    (repo / "unrelated.txt").write_text("user edits")
    assert _push_dashboard_to_github(repo)
    assert git(remote, "show", "main:unrelated.txt") == "original"
    assert (repo / "unrelated.txt").read_text() == "user edits"
    assert git(repo, "diff", "--name-only") == "unrelated.txt"
    assert git(repo, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")


def test_dashboard_push_refuses_unrelated_staged_files(site):
    repo, remote = site
    before = git(remote, "rev-parse", "main")
    (repo / "unrelated.txt").write_text("staged user edits")
    git(repo, "add", "unrelated.txt")
    assert not _push_dashboard_to_github(repo)
    assert git(repo, "diff", "--cached", "--name-only") == "unrelated.txt"
    assert git(remote, "rev-parse", "main") == before


def test_no_new_payload_still_pushes_prior_unpublished_commit(site):
    repo, remote = site
    git(repo, "add", "projects/xsec-alpha/dashboard/data/summary.json")
    git(repo, "commit", "-m", "previous pending publish")
    assert _push_dashboard_to_github(repo)
    assert git(repo, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
