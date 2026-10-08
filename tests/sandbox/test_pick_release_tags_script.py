from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICK_RELEASE_TAGS = REPO_ROOT / "scripts" / "sandbox" / "pick-release-tags.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_picker(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PICK_RELEASE_TAGS), "--repo", str(repo), "--count", "5"],
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")


def test_tagless_full_repo_falls_back_to_current_branch(tmp_path: Path) -> None:
    repo = tmp_path / "full"
    repo.mkdir()
    _init_repo(repo)

    result = _run_picker(repo)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["refs/heads/main"]


def test_tagless_shallow_repo_still_errors(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    shallow = tmp_path / "shallow"

    _git(tmp_path, "init", "--bare", str(origin))
    work.mkdir()
    _init_repo(work)
    (work / "README.md").write_text("hello again\n", encoding="utf-8")
    _git(work, "commit", "-am", "second")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "origin", "main")
    _git(tmp_path, "clone", "--depth", "1", "--branch", "main", f"file://{origin}", str(shallow))
    assert _git_output(shallow, "rev-parse", "--is-shallow-repository") == "true"

    result = _run_picker(shallow)

    assert result.returncode == 1
    assert "A shallow clone has no tags" in result.stderr
