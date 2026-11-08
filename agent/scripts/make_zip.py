"""Build the platform archive: a FLAT zip with only root-level files.

The platform copies only root *.py files and requirements.txt into the task folder and runs
`python3 solution.py task.md` there; sub-folders are dropped. So the automl package is flattened:
  automl/task.py           -> automl_task.py
  automl/research/llm.py   -> automl_research_llm.py
and every `from automl.x.y import ...` is rewritten to `from automl_x_y import ...`.

Usage: python scripts/make_zip.py [output.zip]   (run from anywhere)
"""
import re
import sys
import zipfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
IMPORT = re.compile(r"\bfrom automl\.([A-Za-z_][\w.]*) import\b")


def flat_name(module: str) -> str:
    """'research.llm' -> 'automl_research_llm'."""
    return "automl_" + module.replace(".", "_")


def flatten_source(text: str) -> str:
    out = IMPORT.sub(lambda m: f"from {flat_name(m.group(1))} import", text)
    leftover = re.findall(r"^\s*(?:from|import)\s+automl\b.*$", out, flags=re.MULTILINE)
    if leftover:
        raise SystemExit(f"unflattened imports left: {leftover}")
    return out


def build(out: Path) -> list[str]:
    files: dict[str, str] = {"solution.py": flatten_source((AGENT_DIR / "solution.py").read_text(encoding="utf-8"))}
    for path in sorted((AGENT_DIR / "automl").rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        module = ".".join(path.relative_to(AGENT_DIR / "automl").with_suffix("").parts)
        files[flat_name(module) + ".py"] = flatten_source(path.read_text(encoding="utf-8"))
    modules = {name[:-3] for name in files if name.endswith(".py")}
    for name, text in files.items():  # e.g. `from automl.research import rd` would point nowhere
        for target in re.findall(r"\bfrom (automl_\w+) import\b", text):
            if target not in modules:
                raise SystemExit(f"{name}: import of {target} has no flat module; "
                                 "import functions from the module itself: from automl.pkg.module import f")
    files["requirements.txt"] = (AGENT_DIR / "requirements.txt").read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return list(files)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else AGENT_DIR / "dist" / "solution.zip"
    names = build(out)
    print(f"{out} ({out.stat().st_size // 1024} KB): {names}")


if __name__ == "__main__":
    main()
