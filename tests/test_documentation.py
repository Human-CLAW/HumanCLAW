"""Prevent release-source documentation coverage from silently regressing."""

import ast
import re

from humanclaw_bench.paths import repository_root


RELEASE_DOCUMENT_PAIRS = (
    ("README.md", "README_zh.md"),
    ("docs/ARCHITECTURE.md", "docs/ARCHITECTURE_zh.md"),
    ("docs/ASSETS.md", "docs/ASSETS_zh.md"),
    ("docs/METRICS.md", "docs/METRICS_zh.md"),
    ("docs/MODELS.md", "docs/MODELS_zh.md"),
    ("docs/VIDEOS.md", "docs/VIDEOS_zh.md"),
    ("docs/SPAWN_REPAIRS.md", "docs/SPAWN_REPAIRS_zh.md"),
    ("patches/habitat-sim/README.md", "patches/habitat-sim/README_zh.md"),
    ("resources/benchmark/README.md", "resources/benchmark/README_zh.md"),
    ("resources/hssd/README.md", "resources/hssd/README_zh.md"),
    ("weights/paper_fullval_v1/README.md", "weights/paper_fullval_v1/README_zh.md"),
    (
        "src/humanclaw_bench/motion/training/README.md",
        "src/humanclaw_bench/motion/training/README_zh.md",
    ),
)


def test_every_release_module_class_and_function_has_a_docstring() -> None:
    """Require documentation on nested helpers as well as public functions."""

    source = repository_root() / "src" / "humanclaw_bench"
    missing: list[str] = []
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{relative}: module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{relative}:{node.lineno}: {node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_release_docstrings_do_not_use_placeholder_language() -> None:
    """Reject the generic filler phrases that previously obscured code intent."""

    source = repository_root() / "src" / "humanclaw_bench"
    forbidden = (
        "used by this module",
        "surrounding benchmark operation",
        "state and validate its inputs",
        "Runtime support for",
    )
    offenders: list[str] = []
    for path in sorted(source.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(phrase in text for phrase in forbidden):
            offenders.append(str(path.relative_to(source)))
    assert not offenders, f"Placeholder documentation remains in: {offenders}"


def test_release_markdown_has_linked_chinese_counterparts() -> None:
    """Keep every release document paired and mutually linked."""

    root = repository_root()
    for english_name, chinese_name in RELEASE_DOCUMENT_PAIRS:
        english_path = root / english_name
        chinese_path = root / chinese_name
        assert english_path.is_file(), english_name
        assert chinese_path.is_file(), chinese_name
        assert f'href="{chinese_path.name}"' in english_path.read_text(encoding="utf-8")
        assert f'href="{english_path.name}"' in chinese_path.read_text(encoding="utf-8")


def test_release_markdown_calls_the_terminal_action_stop() -> None:
    """Use the public action name Stop while allowing the legacy filename stand.pt."""

    root = repository_root()
    for pair in RELEASE_DOCUMENT_PAIRS:
        for relative_name in pair:
            text = (root / relative_name).read_text(encoding="utf-8")
            text = text.replace("skills/stand.pt", "").replace("stand.pt", "")
            assert re.search(r"\bstand\b", text, re.IGNORECASE) is None, relative_name
