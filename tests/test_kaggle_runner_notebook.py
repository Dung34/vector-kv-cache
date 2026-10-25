import ast
import builtins
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_runner.ipynb"


def _read_notebook():
    with NOTEBOOK.open(encoding="utf8") as handle:
        return nbformat.read(handle, as_version=4)


def test_notebook_is_valid_and_thin():
    notebook = _read_notebook()
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    print(f"expected_max_cells=6 actual={len(cells)}")
    assert len(cells) <= 6
    for cell in cells:
        lines = cell.source.splitlines()
        print(f"expected_max_lines=15 actual={len(lines)}")
        assert len(lines) <= 15
        assert "import src" not in cell.source
        assert "from src" not in cell.source


def test_notebook_declares_parameters_and_validates_inputs():
    notebook = _read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    names = {"REPO_URL", "COMMIT", "SCRIPT", "NONCE", "TAG", "TRANSFORMERS_VERSION"}
    actual_names = {name for name in names if f"{name} =" in source or f"{name}=" in source}
    print(f"expected={names} actual={actual_names}")
    assert actual_names == names
    assert "assert re.fullmatch" in source
    assert "assert NONCE" in source
    assert "checkout" in source
    assert "rev-parse" in source and "HEAD" in source
    assert "experiments/run_exp.sh" in source


def test_notebook_cell_zero_uses_direct_parameter_assignments():
    notebook = _read_notebook()
    cell_zero = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    names = {"REPO_URL", "COMMIT", "SCRIPT", "NONCE", "TAG", "TRANSFORMERS_VERSION"}
    actual = {line.split("=", 1)[0].strip() for line in cell_zero.splitlines() if "=" in line and not line.lstrip().startswith("assert")}
    expected = names
    print(f"expected={expected} actual={actual & names}")
    assert expected <= actual
    assert "os.environ" not in cell_zero
    assert "Hãy" in cell_zero


def test_notebook_cells_define_names_before_use():
    notebook = _read_notebook()
    available = set(dir(builtins))
    missing = []
    for index, cell in enumerate(cell.source for cell in notebook.cells if cell.cell_type == "code"):
        tree = ast.parse(cell)
        defined = set()
        loaded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (loaded if isinstance(node.ctx, ast.Load) else defined).add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
        missing_names = sorted(loaded - available - defined)
        if missing_names:
            missing.append({"cell": index, "names": missing_names})
        available.update(defined)
    expected = []
    print(f"expected={expected} actual={missing}")
    assert missing == expected


def test_notebook_packages_logs_results_and_manifest():
    notebook = _read_notebook()
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    expected_fragments = ["zipfile", "logs", "results", "MANIFEST.txt", "sha256", "COMMIT", "NONCE", "nvidia-smi"]
    actual = {fragment for fragment in expected_fragments if fragment in source}
    print(f"expected={set(expected_fragments)} actual={actual}")
    assert actual == set(expected_fragments)
