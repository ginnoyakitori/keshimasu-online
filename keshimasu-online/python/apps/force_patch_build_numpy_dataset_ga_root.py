# python/apps/force_patch_build_numpy_dataset_ga_root.py

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "python" / "apps" / "build_numpy_dataset.py"


HELPERS = r'''
# ---------------------------------------------------------------------
# GA reverse include helpers
# ---------------------------------------------------------------------


DEFAULT_GA_ROOT = PROJECT_ROOT / "data" / "ga_reverse"
DEFAULT_COMBINED_REPLAY_ROOT = PROJECT_ROOT / "data" / "_tmp_combined_replays_for_numpy"


def _ga_list_json_files(root):
    if not root.exists():
        return []

    out = []

    for child in root.iterdir():
        if child.is_dir():
            out.extend(_ga_list_json_files(child))
        elif child.is_file() and child.suffix.lower() == ".json":
            name = child.name.lower()

            if name.endswith("manifest.json"):
                continue

            if name in ["index.json", "metadata.json"]:
                continue

            if "benchmark" in name:
                continue

            out.append(child)

    return sorted(out)


def _ga_target_f_from_json(path):
    try:
        data = read_json(path)
    except Exception:
        return None

    candidates = [
        data.get("targetWildcards"),
        data.get("targetF"),
        data.get("actualWildcards"),
    ]

    config = data.get("config")

    if isinstance(config, dict):
        candidates.append(config.get("targetWildcards"))
        candidates.append(config.get("targetF"))

    for value in candidates:
        if value is None:
            continue

        try:
            return int(value)
        except Exception:
            pass

    return None


def _safe_copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _prepare_combined_replay_root(replay_root, ga_root, include_ga, ga_limit):
    """
    selfplay replay と ga_reverse JSON を一時ディレクトリにまとめる。

    既存 build_dataset() は replay-root/f0〜f3 の epoch-*.json を読むので、
    GA JSON も targetWildcards に応じて fN/epoch-ga-*.json としてコピーする。
    """
    if not include_ga:
        return replay_root

    combined_root = DEFAULT_COMBINED_REPLAY_ROOT

    if combined_root.exists():
        import shutil
        shutil.rmtree(combined_root)

    combined_root.mkdir(parents=True, exist_ok=True)

    # selfplay をコピー
    copied_selfplay = 0

    for f in ["f0", "f1", "f2", "f3"]:
        src_dir = replay_root / f

        if not src_dir.exists():
            continue

        for src in _ga_list_json_files(src_dir):
            rel = src.relative_to(replay_root)
            dst = combined_root / rel
            _safe_copy_file(src, dst)
            copied_selfplay += 1

    # GA JSON をコピー
    copied_ga = 0
    ga_files = _ga_list_json_files(ga_root)

    if ga_limit and int(ga_limit) > 0:
        ga_files = ga_files[: int(ga_limit)]

    for src in ga_files:
        target_f = _ga_target_f_from_json(src)

        if target_f is None:
            continue

        dst_dir = combined_root / f"f{target_f}"

        # build_dataset 側の list_replay_files が epoch-*.json だけを見るため epoch- を付ける
        safe_stem = src.stem.replace(" ", "-")
        dst = dst_dir / f"epoch-ga-{safe_stem}.json"

        _safe_copy_file(src, dst)
        copied_ga += 1

    print("Combined replay root:", combined_root)
    print("  copied selfplay:", copied_selfplay)
    print("  copied ga      :", copied_ga)

    return combined_root
'''


def insert_helpers(text):
    if "def _prepare_combined_replay_root(" in text:
        return text

    marker = "# ---------------------------------------------------------------------\n# CLI"

    pos = text.find(marker)

    if pos < 0:
        raise RuntimeError("Could not find CLI marker.")

    return text[:pos].rstrip() + "\n\n" + HELPERS.strip() + "\n\n\n" + text[pos:]


def patch_parse_args(text):
    if "--ga-root" in text:
        return text

    marker = '''    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
'''

    if marker not in text:
        raise RuntimeError("Could not find --output-dir argument block.")

    insert = '''    parser.add_argument(
        "--ga-root",
        default=str(PROJECT_ROOT / "data" / "ga_reverse"),
        help="GA reverse puzzle JSON root. Included by default.",
    )

    parser.add_argument(
        "--no-ga",
        action="store_true",
        help="Do not include data/ga_reverse JSON files.",
    )

    parser.add_argument(
        "--ga-limit",
        type=int,
        default=0,
        help="Max GA reverse JSON files to include. 0 means no limit.",
    )

'''

    return text.replace(marker, marker + insert, 1)


def patch_main_call(text):
    if "effective_replay_root = _prepare_combined_replay_root" in text:
        return text

    old = '''    summary = build_dataset(
        replay_root=Path(args.replay_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        league=args.league,
        limit=int(args.limit),
        words_file=words_file,
        max_paths_per_word=int(args.max_paths_per_word),
    )
'''

    new = '''    effective_replay_root = _prepare_combined_replay_root(
        replay_root=Path(args.replay_root).resolve(),
        ga_root=Path(args.ga_root).resolve(),
        include_ga=not bool(args.no_ga),
        ga_limit=int(args.ga_limit),
    )

    summary = build_dataset(
        replay_root=effective_replay_root,
        output_dir=Path(args.output_dir).resolve(),
        league=args.league,
        limit=int(args.limit),
        words_file=words_file,
        max_paths_per_word=int(args.max_paths_per_word),
    )
'''

    if old not in text:
        raise RuntimeError("Could not find build_dataset() call in main().")

    return text.replace(old, new, 1)


def main():
    text = TARGET.read_text(encoding="utf-8")

    text = insert_helpers(text)
    text = patch_parse_args(text)
    text = patch_main_call(text)

    TARGET.write_text(text, encoding="utf-8")

    print("Patched:", TARGET)


if __name__ == "__main__":
    main()