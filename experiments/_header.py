"""Shared runtime header for experiment scripts."""

import hashlib
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_command(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def print_header(nonce: str, seed: int, script_path: str) -> dict:
    """Print and return the runtime metadata for an experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = (
        f"cuda:{'_'.join(torch.cuda.get_device_name(0).split())}"
        if torch.cuda.is_available()
        else "cpu"
    )
    repo_root = Path(__file__).resolve().parents[1]
    warnings = []
    try:
        git_commit = _git_command(repo_root, "rev-parse", "HEAD").stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        git_commit = "KHONG_XAC_DINH"
        warnings.append(f"git rev-parse HEAD failed: {error}")

    try:
        git_dirty = "true" if _git_command(repo_root, "status", "--porcelain").stdout.strip() else "false"
    except (OSError, subprocess.CalledProcessError) as error:
        git_dirty = "KHONG_XAC_DINH"
        warnings.append(f"git status --porcelain failed: {error}")

    values = {
        "nonce": str(nonce),
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "seed": str(seed),
        "script_sha256": _sha256_file(script_path),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    print("[HEADER] " + " ".join(f"{key}={value}" for key, value in values.items()))
    for warning in warnings:
        print(f"[HEADER-WARN] {warning}")
    return values
