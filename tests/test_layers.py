"""The layers depend on each other only in the allowed directions; only data opens files."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "lungscope"
NOT_ALLOWED = {  # layer -> modules it must not import
    "data": ("lungscope.analysis", "lungscope.views", "lungscope.app", "trame", "trame_pyvista"),
    "analysis": (
        "lungscope.data",
        "lungscope.views",
        "lungscope.app",
        "pyvista",
        "vtk",
        "vtkmodules",
        "trame",
        "trame_pyvista",
        "matplotlib",
        "plotly",
    ),
    "views": ("lungscope.data", "lungscope.app"),
}
FILE_READERS = {"open", "read", "get_reader", "read_text", "read_bytes", "open_dataset", "open_zarr", "load"}


def modules(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).glob("*.py"))


def imported(path: Path):
    """Absolute names of everything a module imports, relative imports resolved.

    ``from x import y`` yields both x and x.y, since y may be a module: ``from lungscope import
    data`` imports the data layer.
    """
    package = ["lungscope"] if path.parent == PACKAGE else ["lungscope", path.parent.name]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, node.module] if node.module else base)
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names)


def file_access(path: Path) -> list[str]:
    """Calls in a module that read files."""
    calls = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in FILE_READERS:
                calls.append(name)
    return calls


def test_every_layer_exists():
    for layer in NOT_ALLOWED:
        assert (PACKAGE / layer / "__init__.py").is_file(), layer
    assert (PACKAGE / "app.py").is_file()


@pytest.mark.parametrize("layer", sorted(NOT_ALLOWED))
def test_layer_imports(layer):
    for path in modules(layer):
        for name in imported(path):
            bad = [m for m in NOT_ALLOWED[layer] if name == m or name.startswith(m + ".")]
            assert not bad, f"{layer}/{path.name} imports {name}"


@pytest.mark.parametrize("layer", ["analysis", "views"])
def test_no_file_access(layer):
    for path in modules(layer):
        assert not file_access(path), f"{layer}/{path.name} calls {file_access(path)}"


def test_app_reads_files_only_through_data():
    assert not file_access(PACKAGE / "app.py")


def test_imports_are_resolved_to_layers(tmp_path):
    module = tmp_path / "views" / "probe.py"
    module.parent.mkdir()
    module.write_text("from lungscope import data\nfrom .. import app\nfrom lungscope.analysis import run_spread\n")
    names = set(imported(module))
    assert {"lungscope.data", "lungscope.app", "lungscope.analysis.run_spread"} <= names
