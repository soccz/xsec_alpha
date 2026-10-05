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


def test_tracking_cache_can_follow_replaced_remote_without_force_pushing_or_losing_work(site, tmp_path):
    repo, remote = site
    cache = tmp_path / "publisher"
    assert publish_files(repo, cache=cache, remote=str(remote))
    git(cache, "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main")
    (repo / "unrelated.txt").write_text("new authoritative remote work")
    git(repo, "add", "unrelated.txt")
    git(repo, "commit", "-m", "independent remote branch")
    replacement = git(repo, "rev-parse", "HEAD")
    git(remote, "fetch", str(repo), "+refs/heads/main:refs/heads/main")
    assert publish_files(repo, cache=cache, remote=str(remote))
    assert git(remote, "rev-parse", "main^") == replacement
    assert git(remote, "show", "main:unrelated.txt") == "new authoritative remote work"
    assert git(repo, "rev-parse", "HEAD") == replacement


def test_depth_one_fetch_can_follow_a_normal_remote_child(site, tmp_path):
    repo, remote = site
    cache = tmp_path / "publisher"
    # file:// exercises actual shallow fetches; plain local paths ignore --depth.
    assert publish_files(repo, cache=cache, remote=remote.as_uri())
    head = git(remote, "rev-parse", "main")
    assert publish_files(repo, cache=cache, remote=remote.as_uri())
    assert git(remote, "rev-parse", "main") == head


def test_partial_cache_publishes_without_fetching_unrelated_blobs(site, tmp_path):
    repo, remote = site
    (repo / "big.bin").write_bytes(bytes(range(256)) * 800)
    git(repo, "add", "big.bin")
    git(repo, "commit", "-m", "large unrelated file from another project")
    git(repo, "push", "origin", "main")
    git(remote, "config", "uploadpack.allowFilter", "true")
    git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")
    big = git(remote, "rev-parse", "main:big.bin")
    cache = tmp_path / "publisher"
    assert publish_files(repo, cache=cache, remote=remote.as_uri())
    assert git(remote, "show", f"main:{PAYLOAD_PATHS[0]}") == '{"encrypted": "new"}'
    assert git(remote, "rev-parse", "main:big.bin") == big
    missing = git(cache, "rev-list", "--objects", "--missing=print", "refs/remotes/origin/main")
    assert f"?{big}" in missing.split()


def test_cache_with_pack_garbage_is_rebuilt(site, tmp_path):
    repo, remote = site
    cache = tmp_path / "publisher"
    assert publish_files(repo, cache=cache, remote=remote.as_uri())
    garbage = cache / "objects/pack/tmp_pack_dead"
    garbage.write_bytes(b"partial")
    (repo / "projects/xsec-alpha/dashboard/data/summary.json").write_text('{"encrypted": "newer"}')
    assert publish_files(repo, cache=cache, remote=remote.as_uri())
    assert not garbage.exists()
    assert git(remote, "show", f"main:{PAYLOAD_PATHS[0]}") == '{"encrypted": "newer"}'
