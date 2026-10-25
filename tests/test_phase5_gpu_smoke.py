import contextlib
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "phase5_gpu_smoke.py"


def test_smoke_subprocess_has_header_checks_and_result(tmp_path):
    env = os.environ.copy()
    env["PHASE5_SMOKE"] = "1"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--nonce", "12003"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    expected_code = 0
    actual_code = result.returncode
    print(f"expected={expected_code} actual={actual_code}")
    assert actual_code == expected_code
    assert lines[0].startswith("[HEADER]")
    assert "[MODE] mode=smoke" in lines
    check_lines = [line for line in lines if line.startswith("[CHECK]")]
    expected_checks = 4
    actual_checks = len(check_lines)
    print(f"expected={expected_checks} actual={actual_checks}")
    assert actual_checks == expected_checks
    assert all("ok=True" in line for line in check_lines)
    assert "[RESULT] all_ok=true" in lines


def test_run_checks_returns_three_passing_checks():
    module = importlib.import_module("experiments.phase5_gpu_smoke")
    model, tokenizer = module.build_model("smoke")
    input_ids = module.make_prompt("smoke", tokenizer, model)
    with contextlib.redirect_stdout(io.StringIO()):
        checks = module.run_checks(model, input_ids)
    expected = 4
    actual = len(checks)
    print(f"expected={expected} actual={actual}")
    assert actual == expected
    assert all(check["ok"] for check in checks)


def test_gpu_mode_without_cuda_fails_clearly(monkeypatch):
    module = importlib.import_module("experiments.phase5_gpu_smoke")
    monkeypatch.delenv("PHASE5_SMOKE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = module.main(["--nonce", "12003"])
    expected_ok = False
    actual_ok = code == 0
    print(f"expected={expected_ok} actual={actual_ok}")
    assert code != 0
    assert "[FAIL] reason=cuda_not_available" in output.getvalue()
