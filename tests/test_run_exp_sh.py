import hashlib
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bash_path(path):
    value = str(path)
    if len(value) >= 2 and value[1] == ":":
        return f"/{value[0].lower()}{value[2:].replace(chr(92), '/')}"
    return value


def _bash_executable():
    candidates = [
        os.environ.get("BASH"),
        r"C:\Program Files\Git\bin\bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "bash"


def _run_runner(tmp_path, script_text, nonce="65882", tag="runner_test", expected_code=None):
    script = tmp_path / "sample.py"
    script.write_text(script_text, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [_bash_executable(), _bash_path(ROOT / "experiments" / "run_exp.sh"), _bash_path(script), nonce, tag, "python"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if expected_code is not None:
        print(f"expected={expected_code} actual={result.returncode}")
        assert result.returncode == expected_code
    return result, tmp_path / "logs" / f"{tag}.log"


def test_runner_preserves_python_exit_code_and_utf8(tmp_path):
    result, log_path = _run_runner(
        tmp_path,
        "print('Đã chạy')\nimport sys\nsys.exit(3)\n",
        tag="nonzero",
        expected_code=3,
    )
    raw = log_path.read_bytes()
    text = raw.decode("utf-8")
    expected = True
    actual = "Đã chạy" in text
    print(f"expected={expected} actual={actual}")
    assert raw[:3] != b"\xef\xbb\xbf"
    assert "\r" not in text
    assert actual == expected
    assert text.splitlines()[-1] == "[EXIT] code=3"
    assert "[HASH] " in result.stdout


def test_runner_zero_exit_and_numbered_output(tmp_path):
    _, log_path = _run_runner(
        tmp_path,
        "print('line 1')\nprint('line 2')\n",
        tag="zero",
        expected_code=0,
    )
    numbered = (log_path.parent / "zero.numbered.txt").read_text(encoding="utf-8")
    expected = "\n".join(f"{index:4d}| {line}" for index, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1))
    print(f"expected={expected!r} actual={numbered.rstrip()!r}")
    assert numbered.rstrip("\n") == expected


def test_runner_hash_matches_log(tmp_path):
    result, log_path = _run_runner(tmp_path, "print('hash')\n", tag="hash", expected_code=0)
    expected = hashlib.sha256(log_path.read_bytes()).hexdigest()
    actual = next(line.split(maxsplit=1)[1] for line in result.stdout.splitlines() if line.startswith("[HASH]"))
    print(f"expected={expected} actual={actual}")
    assert actual == expected


def test_runner_missing_arguments_fails_without_log(tmp_path):
    result = subprocess.run(
        [_bash_executable(), _bash_path(ROOT / "experiments" / "run_exp.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(f"expected_nonzero={result.returncode != 0} actual={result.returncode}")
    assert result.returncode != 0
    assert not (tmp_path / "logs").exists()


def test_runner_script_has_lf_only():
    raw = (ROOT / "experiments" / "run_exp.sh").read_bytes()
    actual = b"\r" in raw
    print(f"expected_crlf=False actual={actual}")
    assert not actual
