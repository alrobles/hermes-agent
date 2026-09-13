"""Regression coverage for release-tag selection in install/update E2E."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PICK_RELEASE_TAGS = REPO_ROOT / "scripts" / "sandbox" / "pick-release-tags.sh"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")


def test_pick_release_tags_falls_back_to_current_branch_when_repo_has_no_releases(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    completed = subprocess.run(
        ["bash", str(PICK_RELEASE_TAGS), "--repo", str(repo)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["refs/heads/main"]
    assert "falling back to refs/heads/main" in completed.stderr


def test_pick_release_tags_still_rejects_shallow_repos_without_visible_tags(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--depth", "1", f"file://{source}", str(clone)],
        capture_output=True,
        text=True,
        check=True,
    )

    completed = subprocess.run(
        ["bash", str(PICK_RELEASE_TAGS), "--repo", str(clone)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "A shallow clone has no tags" in completed.stderr
    assert "fetch-depth: 0" in completed.stderr
