# python/apps/patch_ga_reverse_word_modes_force.py

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GA_REVERSE = ROOT / "python" / "src" / "keshimasu_py" / "ga_reverse.py"
WORD_MODES = ROOT / "python" / "src" / "keshimasu_py" / "word_modes.py"
SNAPSHOT = ROOT / "python" / "apps" / "snapshot_program_version.py"


WORD_MODES_CODE = r'''
# python/src/keshimasu_py/word_modes.py

from __future__ import annotations

import json
from pathlib import Path


WORD_MODE_FILES = {
    "country": [
        "data/words/countries.json",
        "data/country_words.json",
        "country_words.json",
        "country_words_tmp.json",
    ],
    "capital": [
        "data/words/capitals.json",
    ],
    "pokemon": [
        "data/words/pokemon.json",
    ],
}


def _read_json_words(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [
            str(v)
            for v in data
            if str(v).strip()
        ]

    if isinstance(data, dict):
        for key in ["words", "data", "countries", "capitals", "pokemon"]:
            value = data.get(key)

            if isinstance(value, list):
                return [
                    str(v)
                    for v in value
                    if str(v).strip()
                ]

    raise ValueError(f"Unsupported word file format: {path}")


def resolve_words_file(mode, project_root):
    mode = str(mode or "country").lower()

    if mode == "custom":
        return None

    candidates = WORD_MODE_FILES.get(mode)

    if not candidates:
        valid = ", ".join(sorted(WORD_MODE_FILES.keys()))
        raise ValueError(f"Unknown word mode: {mode}. Valid modes: {valid}, custom")

    tried = []

    for rel in candidates:
        path = Path(project_root) / rel
        tried.append(str(path))

        if path.exists():
            return path

    raise FileNotFoundError(
        "No word file found for mode="
        + mode
        + ". Tried:\n  - "
        + "\n  - ".join(tried)
    )


def load_words_by_mode(mode, project_root):
    path = resolve_words_file(mode, project_root)

    if path is None:
        raise ValueError("mode=custom requires --words-file")

    words = _read_json_words(path)

    words = [
        str(w)
        for w in words
        if 2 <= len(list(str(w))) <= 5
    ]

    words = sorted(set(words))

    if not words:
        raise ValueError(f"No playable words in file: {path}")

    return words
'''


def ensure_word_modes_file() -> None:
    WORD_MODES.write_text(
        WORD_MODES_CODE.strip() + "\n",
        encoding="utf-8",
    )
    print("Wrote:", WORD_MODES)


def patch_ga_reverse() -> None:
    text = GA_REVERSE.read_text(encoding="utf-8")

    # import追加
    import_line = "from .word_modes import load_words_by_mode, resolve_words_file\n"

    if import_line not in text:
        marker = "from .core_solver import solve_numba\n"

        if marker not in text:
            raise RuntimeError("Could not find import marker: from .core_solver import solve_numba")

        text = text.replace(
            marker,
            marker + import_line,
            1,
        )
        print("Added word_modes import.")

    # parse_args に --word-mode 追加
    if "--word-mode" not in text:
        marker = '    parser.add_argument("--words-file", default="")\n'

        if marker not in text:
            raise RuntimeError('Could not find parser.add_argument("--words-file", default="")')

        insert = '''    parser.add_argument(
        "--word-mode",
        choices=["country", "capital", "pokemon", "custom"],
        default="country",
        help="Word list mode. Use custom together with --words-file.",
    )
'''

        text = text.replace(
            marker,
            marker + insert,
            1,
        )
        print("Added --word-mode argument.")

    # main() の words_file / words 読み込みブロックを置換
    old_exact = '''    if args.words_file:
        words_file = Path(args.words_file).resolve()
    else:
        words_file = _create_dummy_words_file()

    words = load_words_file(words_file)
'''

    new_block = '''    if args.words_file:
        words_file = Path(args.words_file).resolve()
        words = load_words_file(words_file)
        word_mode = args.word_mode if hasattr(args, "word_mode") else "custom"
    else:
        word_mode = args.word_mode if hasattr(args, "word_mode") else "country"
        words_file = resolve_words_file(word_mode, PROJECT_ROOT)

        if words_file is None:
            raise ValueError("word-mode=custom requires --words-file")

        words = load_words_by_mode(word_mode, PROJECT_ROOT)
'''

    if old_exact in text:
        text = text.replace(old_exact, new_block, 1)
        print("Replaced exact word loading block.")
    elif "word_mode = args.word_mode" not in text:
        pattern = re.compile(
            r'''    if args\.words_file:\n'''
            r'''        words_file = Path\(args\.words_file\)\.resolve\(\)\n'''
            r'''    else:\n'''
            r'''        words_file = _create_dummy_words_file\(\)\n\n'''
            r'''    words = load_words_file\(words_file\)\n''',
            re.MULTILINE,
        )

        text, count = pattern.subn(new_block, text, count=1)

        if count:
            print("Replaced regex word loading block.")
        else:
            raise RuntimeError("Could not find word loading block in main().")

    # printにWord mode追加
    marker = '    print("Words file:", words_file)\n'

    if marker in text and '    print("Word mode:", word_mode)\n' not in text:
        text = text.replace(
            marker,
            marker + '    print("Word mode:", word_mode)\n',
            1,
        )
        print("Added Word mode print.")

    # generatorへ属性を持たせる
    run_marker = "    best = generator.run()\n"

    if run_marker in text and "generator.word_mode = word_mode" not in text:
        text = text.replace(
            run_marker,
            '    generator.word_mode = word_mode\n'
            '    generator.words_file_path = str(words_file)\n\n'
            + run_marker,
            1,
        )
        print("Added generator.word_mode / words_file_path.")

    # save JSON の config に wordMode / wordsFile を追加
    seed_marker = '                "seed": self.seed,\n'

    if seed_marker in text and '"wordMode": getattr(self, "word_mode", None),' not in text:
        text = text.replace(
            seed_marker,
            seed_marker
            + '                "wordMode": getattr(self, "word_mode", None),\n'
            + '                "wordsFile": getattr(self, "words_file_path", None),\n',
            1,
        )
        print("Added wordMode / wordsFile to config.")

    GA_REVERSE.write_text(text, encoding="utf-8")
    print("Patched:", GA_REVERSE)


def patch_snapshot_program_version() -> None:
    if not SNAPSHOT.exists():
        return

    text = SNAPSHOT.read_text(encoding="utf-8")

    target = '    "python/src/keshimasu_py/ga_reverse.py",\n'

    if target in text and '"python/src/keshimasu_py/word_modes.py",' not in text:
        text = text.replace(
            target,
            target + '    "python/src/keshimasu_py/word_modes.py",\n',
            1,
        )
        SNAPSHOT.write_text(text, encoding="utf-8")
        print("Added word_modes.py to GENERATOR_FILES in snapshot script.")


def main() -> None:
    if not GA_REVERSE.exists():
        raise FileNotFoundError(GA_REVERSE)

    ensure_word_modes_file()
    patch_ga_reverse()
    patch_snapshot_program_version()


if __name__ == "__main__":
    main()