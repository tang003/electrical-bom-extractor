"""Create a reproducible source-and-fixture handoff archive.

The archive contains the current working files, including uncommitted changes.
It deliberately excludes Git history, project output, caches, and local secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = (
    ".agents", "backend", "bom_extractor", "dataset", "frontend",
    "scripts", "skills", "tests", "tools", "参考文件",
)
INCLUDE_FILES = ("README.md", "requirements.txt", "requirements-dev.txt", "start_workbench.bat")
EXCLUDED_DIRS = {
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv", "env",
    "bin", "obj", "runs",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log")


def included_files() -> list[Path]:
    files = [ROOT / name for name in INCLUDE_FILES]
    for directory in INCLUDE_DIRS:
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    result = []
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.name == ".env" or path.name.startswith(".env.") or path.name.endswith(".local.json"):
            continue
        result.append(path)
    return sorted(result, key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / f"electrical-bom-extractor-agent-handoff-{date.today():%Y%m%d}.zip")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = included_files()
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    manifest = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            manifest.append(f"{sha256(data)}  {relative}")
        handoff = (ROOT / "HANDOFF.md").read_bytes()
        archive.writestr("HANDOFF.md", handoff)
        manifest.append(f"{sha256(handoff)}  HANDOFF.md")
        archive.writestr("WORKTREE_STATUS.txt", status.encode("utf-8"))
        manifest.append(f"{sha256(status.encode('utf-8'))}  WORKTREE_STATUS.txt")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"Archive: {destination}")
    print(f"Files: {len(files) + 3}")
    print(f"Bytes: {destination.stat().st_size}")
    print(f"SHA256: {sha256(destination.read_bytes())}")


if __name__ == "__main__":
    main()
