"""Git worktrees for parallel safety (step 8).

The moment the system spawns more than one agent, file edits start colliding. A
git worktree gives each agent its own working directory on its own branch,
sharing the same repo history — so a maker's edits in worktree A literally can't
touch a verifier reading worktree B. Each parallel structural experiment runs in
its own checkout; the orchestrator collects results and merges the best one.

Thin, safe wrappers over `git worktree`. Everything shells out to git and raises
a clear error if git or the repo isn't available.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


def _git(*args: str, cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # git not installed
        raise WorktreeError("git is not installed") from exc
    if result.returncode != 0:
        raise WorktreeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def repo_root(cwd: str | Path | None = None) -> Path:
    return Path(_git("rev-parse", "--show-toplevel", cwd=cwd))


@dataclass
class Worktree:
    path: Path
    branch: str


def create(branch: str, *, base: str = "HEAD", root: str | Path | None = None,
           parent: str | Path = ".worktrees") -> Worktree:
    """Create a worktree on a new branch under `parent` (default .worktrees/)."""
    repo = repo_root(root)
    dest = repo / parent / branch.replace("/", "-")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-b", branch, str(dest), base, cwd=repo)
    return Worktree(path=dest, branch=branch)


def list_worktrees(root: str | Path | None = None) -> list[Worktree]:
    out = _git("worktree", "list", "--porcelain", cwd=root)
    trees: list[Worktree] = []
    path: str | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
        elif line.startswith("branch "):
            branch = line[len("branch "):].replace("refs/heads/", "")
            if path:
                trees.append(Worktree(path=Path(path), branch=branch))
                path = None
    return trees


def remove(worktree: str | Path | Worktree, *, force: bool = True,
           root: str | Path | None = None) -> None:
    path = worktree.path if isinstance(worktree, Worktree) else Path(worktree)
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    _git(*args, cwd=root)


@contextmanager
def isolated(branch: str, *, base: str = "HEAD", root: str | Path | None = None,
             cleanup: bool = True):
    """Context manager: run agent work in a throwaway worktree, then clean up."""
    wt = create(branch, base=base, root=root)
    try:
        yield wt
    finally:
        if cleanup:
            try:
                remove(wt, root=root)
                _git("branch", "-D", branch, cwd=root)
            except WorktreeError:
                pass
