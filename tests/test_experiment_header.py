import hashlib
import io
import random
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments._header import print_header


REPO_ROOT = Path(__file__).resolve().parents[1]


def _header_output(tmp_path, nonce="10920", seed=17):
    script = tmp_path / "sample.py"
    script.write_text("print('sample')\n", encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = print_header(nonce, seed, str(script))
    return result, buffer.getvalue().splitlines(), script


def test_header_is_one_line_and_contains_all_fields(tmp_path):
    result, lines, _ = _header_output(tmp_path)

    expected_fields = {
        "nonce",
        "timestamp",
        "python",
        "torch",
        "transformers",
        "device",
        "seed",
        "script_sha256",
        "git_commit",
        "git_dirty",
    }
    print(f"expected={expected_fields} actual={set(result)}")
    assert len(lines) == 1
    assert lines[0].startswith("[HEADER]")
    assert expected_fields <= set(result)
    assert all(f"{field}=" in lines[0] for field in expected_fields)


def test_header_tokens_parse_to_returned_dict(tmp_path):
    result, lines, _ = _header_output(tmp_path)
    tokens = lines[0].split()[1:]
    actual = dict(token.split("=", 1) for token in tokens if "=" in token)
    expected = result
    print(f"expected={expected} actual={actual}")
    assert all(token.count("=") == 1 for token in tokens)
    assert actual == expected


def test_script_hash_matches_file(tmp_path):
    result, _, script = _header_output(tmp_path)
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    actual = result["script_sha256"]
    print(f"expected={expected} actual={actual}")
    assert actual == expected


def test_git_commit_matches_independent_command(tmp_path):
    result, _, _ = _header_output(tmp_path)
    expected = subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    actual = result["git_commit"]
    print(f"expected={expected} actual={actual}")
    assert actual == expected


def test_git_failure_prints_warning_and_unknown_commit(monkeypatch, tmp_path):
    import experiments._header as header_module

    real_run = header_module.subprocess.run

    def failing_git(command, *args, **kwargs):
        if command and command[0] == "git":
            raise OSError("simulated git unavailable")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(header_module.subprocess, "run", failing_git)
    result, lines, _ = _header_output(tmp_path)

    expected = "KHONG_XAC_DINH"
    actual = result["git_commit"]
    print(f"expected={expected} actual={actual}")
    assert actual == expected
    assert any(line.startswith("[HEADER-WARN]") for line in lines)


def test_device_matches_runtime_and_seed_is_applied(tmp_path):
    result, _, _ = _header_output(tmp_path, seed=10920)
    expected_device = (
        f"cuda:{'_'.join(torch.cuda.get_device_name(0).split())}"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"expected={expected_device} actual={result['device']}")
    assert result["device"] == expected_device

    expected_random = random.Random(10920).random()
    expected_numpy = np.random.RandomState(10920).random_sample()
    random_actual = random.random()
    numpy_actual = np.random.random_sample()
    print(f"expected={expected_random} actual={random_actual}")
    print(f"expected={expected_numpy} actual={numpy_actual}")
    assert random_actual == expected_random
    assert numpy_actual == expected_numpy
