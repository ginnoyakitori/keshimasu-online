# python/apps/patch_build_numpy_dataset_include_ga.py

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "python" / "apps" / "build_numpy_dataset.py"


GA_HELPERS = r'''
def list_ga_reverse_files(
    ga_root: Path,
    limit: int,
) -> List[Path]:
    """
    data/ga_reverse 以下の GA生成パズルJSONを列挙する。

    manifest系 JSON は除外し、実際のパズルJSON候補だけを返す。
    ただし最終的な妥当性は process_replay_file 側で検証する。
    """
    if not ga_root.exists():
        return []

    files = list_json_files(ga_root)

    out: List[Path] = []

    for file in files:
        name = file.name.lower()

        if name.endswith("manifest.json"):
            continue

        if name == "index.json":
            continue

        if "benchmark" in name:
            continue

        if "metadata" in name:
            continue

        out.append(file)

    out = sorted(out)

    if limit > 0:
        out = out[:limit]

    return out
'''


COLLECT_WORDS_HELPER = r'''
def replay_all_words_for_vocab(replay: Dict[str, Any]) -> List[str]:
    """
    語彙構築用に replay / GA JSON から単語を広めに集める。

    対象:
      - replay["words"]
      - puzzle/config words
      - solutionMoves / solvedMoves 内の word
      - genes 内の word
      - policyTrap topMove / correctFirstMove
    """
    words: List[str] = []

    words.extend(replay_words(replay))

    for move in replay_solution_moves(replay):
        word = move.get("word")

        if word:
            words.append(str(word))

    genes = replay.get("genes")

    if isinstance(genes, list):
        for gene in genes:
            if isinstance(gene, dict) and gene.get("word"):
                words.append(str(gene["word"]))

    policy_trap = replay.get("policyTrap")

    if isinstance(policy_trap, dict):
        for key in ["topMove", "correctFirstMove"]:
            value = policy_trap.get(key)

            if isinstance(value, dict) and value.get("word"):
                words.append(str(value["word"]))

    return [
        str(word)
        for word in words
        if 2 <= len(list(str(word))) <= MAX_WORD_LENGTH
    ]
'''


def insert_after_function(text: str, function_name: str, insert_text: str) -> str:
    if insert_text.strip().splitlines()[0] in text:
        return text

    pattern = re.compile(
        rf"def {re.escape(function_name)}\(.*?(?=\ndef |\n# ---------------------------------------------------------------------|\Z)",
        re.DOTALL,
    )

    match = pattern.search(text)

    if not match:
        raise RuntimeError(f"Could not find function: {function_name}")

    end = match.end()

    return text[:end].rstrip() + "\n\n\n" + insert_text.strip() + "\n\n" + text[end:].lstrip()


def patch_defaults(text: str) -> str:
    if "DEFAULT_GA_ROOT" in text:
        return text

    marker = 'DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"\n'

    if marker not in text:
        raise RuntimeError("Could not find DEFAULT_OUTPUT_DIR marker.")

    replacement = (
        marker
        + 'DEFAULT_GA_ROOT = PROJECT_ROOT / "data" / "ga_reverse"\n'
    )

    return text.replace(marker, replacement, 1)


def patch_collect_words(text: str) -> str:
    text = insert_after_function(
        text,
        "replay_words",
        COLLECT_WORDS_HELPER,
    )

    old = "            words.extend(replay_words(replay))"
    new = "            words.extend(replay_all_words_for_vocab(replay))"

    if old in text:
        text = text.replace(old, new)

    return text


def patch_ga_file_listing(text: str) -> str:
    text = insert_after_function(
        text,
        "list_replay_files",
        GA_HELPERS,
    )

    return text


def patch_build_dataset_signature(text: str) -> str:
    old = '''def build_dataset(
    *,
    replay_root: Path,
    output_dir: Path,
    league: str,
    limit: int,
    words_file: Optional[Path],
    max_paths_per_word: int,
) -> Dict[str, Any]:
'''

    new = '''def build_dataset(
    *,
    replay_root: Path,
    output_dir: Path,
    league: str,
    limit: int,
    words_file: Optional[Path],
    max_paths_per_word: int,
    ga_root: Path,
    include_ga: bool,
    ga_limit: int,
) -> Dict[str, Any]:
'''

    if old in text:
        return text.replace(old, new, 1)

    if "include_ga: bool" in text:
        return text

    raise RuntimeError("Could not patch build_dataset signature.")


def patch_build_dataset_file_block(text: str) -> str:
    old = '''    files = list_replay_files(
        replay_root,
        league,
        limit,
    )

    if not files:
        raise FileNotFoundError(
            f"No replay files found: root={replay_root}, league={league}"
        )
'''

    new = '''    selfplay_files = list_replay_files(
        replay_root,
        league,
        limit,
    )

    ga_files: List[Path] = []

    if include_ga:
        ga_files = list_ga_reverse_files(
            ga_root,
            ga_limit,
        )

    files = [
        *selfplay_files,
        *ga_files,
    ]

    if not files:
        raise FileNotFoundError(
            f"No input JSON files found: "
            f"selfplay_root={replay_root}, league={league}, "
            f"ga_root={ga_root}, include_ga={include_ga}"
        )
'''

    if old in text:
        return text.replace(old, new, 1)

    if "selfplay_files = list_replay_files" in text:
        return text

    raise RuntimeError("Could not patch build_dataset file listing block.")


def patch_prints_and_summary(text: str) -> str:
    old = '''    print("Files      :", len(files))
    print("Words      :", len(words_text))
'''

    new = '''    print("Selfplay files:", len(selfplay_files))
    print("GA files      :", len(ga_files))
    print("Files total   :", len(files))
    print("Words         :", len(words_text))
'''

    if old in text:
        text = text.replace(old, new, 1)

    old_summary = '''        "inputReplayFiles": len(files),
        "skippedFiles": len(skipped),
'''

    new_summary = '''        "inputReplayFiles": len(files),
        "inputSelfplayFiles": len(selfplay_files),
        "inputGaFiles": len(ga_files),
        "includeGaReverse": bool(include_ga),
        "gaRoot": str(ga_root),
        "skippedFiles": len(skipped),
'''

    if old_summary in text:
        text = text.replace(old_summary, new_summary, 1)

    return text


def patch_parse_args(text: str) -> str:
    if "--ga-root" not in text:
        marker = '''    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
'''

        insert = '''    parser.add_argument(
        "--ga-root",
        default=str(DEFAULT_GA_ROOT),
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
        help="Max GA reverse JSON files to load. 0 means no limit.",
    )

'''

        if marker not in text:
            raise RuntimeErrorut-dir parser block.")

        text = text.replace(marker, marker + insert, 1)

    return text


def patch_main_call(text: str) -> str:
    old = '''    summary = build_dataset(
        replay_root=Path(args.replay_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        league=args.league,
        limit=int(args.limit),
        words_file=words_file,
        max_paths_per_word=int(args.max_paths_per_word),
    )
'''

    new = '''    summary = build_dataset(
        replay_root=Path(args.replay_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        league=args.league,
        limit=int(args.limit),
        words_file=words_file,
        max_paths_per_word=int(args.max_paths_per_word),
        ga_root=Path(args.ga_root).resolve(),
        include_ga=not bool(args.no_ga),
        ga_limit=int(args.ga_limit),
    )
'''

    if old in text:
        return text.replace(old, new, 1)

    if "include_ga=not bool(args.no_ga)" in text:
        return text

    raise RuntimeError("Could not patch build_dataset() call in main().")


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")

    text = patch_defaults(text)
    text = patch_ga_file_listing(text)
    text = patch_collect_words(text)
    text = patch_build_dataset_signature(text)
    text = patch_build_dataset_file_block(text)
    text = patch_prints_and_summary(text)
    text = patch_parse_args(text)
    text = patch_main_call(text)

    TARGET.write_text(text, encoding="utf-8")

    print("Patched build_numpy_dataset.py to include GA reverse JSON:")
    print(TARGET)


if __name__ == "__main__":
    main()