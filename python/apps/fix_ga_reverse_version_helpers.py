# python/apps/fix_ga_reverse_version_helpers.py

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "python" / "src" / "keshimasu_py" / "ga_reverse.py"


HELPERS = r'''
# ---------------------------------------------------------------------
# Program version helpers
# ---------------------------------------------------------------------


def _read_text_if_exists(path):
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:
        return None


def _read_json_if_exists(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _program_versions_root():
    return PROJECT_ROOT / "data" / "program_versions"


def _latest_program_version(kind):
    latest_path = _program_versions_root() / kind / "LATEST.txt"
    return _read_text_if_exists(latest_path)


def _program_version_manifest(kind, snapshot_name):
    if not snapshot_name:
        return None

    manifest_path = (
        _program_versions_root()
        / kind
        / snapshot_name
        / "manifest.json"
    )

    manifest = _read_json_if_exists(manifest_path)

    if not manifest:
        return None

    return {
        "kind": manifest.get("kind"),
        "generation": manifest.get("generation"),
        "name": manifest.get("name"),
        "snapshotName": manifest.get("snapshotName"),
        "createdAt": manifest.get("createdAt"),
        "note": manifest.get("note"),
        "git": manifest.get("git"),
        "fileCount": len(manifest.get("files", [])),
        "missingFileCount": len(manifest.get("missingFiles", [])),
    }


def collect_current_program_versions():
    generator_version = _latest_program_version("generator")
    solver_version = _latest_program_version("solver")

    # 現状、policy_model.onnx などのモデル成果物は solver snapshot に含める運用。
    model_version = solver_version

    return {
        "generator": generator_version,
        "solver": solver_version,
        "model": model_version,
        "manifests": {
            "generator": _program_version_manifest(
                "generator",
                generator_version,
            ),
            "solver": _program_version_manifest(
                "solver",
                solver_version,
            ),
            "model": _program_version_manifest(
                "solver",
                model_version,
            ),
        },
        "source": {
            "generatorLatest": str(
                _program_versions_root()
                / "generator"
                / "LATEST.txt"
            ),
            "solverLatest": str(
                _program_versions_root()
                / "solver"
                / "LATEST.txt"
            ),
        },
    }
'''


def insert_helpers(text: str) -> str:
    if "def collect_current_program_versions()" in text:
        print("collect_current_program_versions already exists.")
        return text

    # class の直前に入れるのが一番安全
    marker = "class KeshimasuGAPuzzleGenerator:"

    pos = text.find(marker)

    if pos < 0:
        # fallback: main CLI marker の前
        marker = "# ---------------------------------------------------------------------\n# CLI"
        pos = text.find(marker)

    if pos < 0:
        raise RuntimeError(
            "Could not find insertion point. "
            "Neither class KeshimasuGAPuzzleGenerator nor CLI marker was found."
        )

    return (
        text[:pos].rstrip()
        + "\n\n\n"
        + HELPERS.strip()
        + "\n\n\n"
        + text[pos:]
    )


def ensure_versions_field(text: str) -> str:
    if '"versions": collect_current_program_versions(),' in text:
        print("versions field already exists.")
        return text

    target = '            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),'

    if target not in text:
        raise RuntimeError("Could not find createdAt line in save() JSON block.")

    replacement = (
        '            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),\n'
        '            "versions": collect_current_program_versions(),'
    )

    return text.replace(target, replacement, 1)


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")

    text = insert_helpers(text)
    text = ensure_versions_field(text)

    TARGET.write_text(text, encoding="utf-8")

    print("Fixed version helpers in:")
    print(TARGET)


if __name__ == "__main__":
    main()
