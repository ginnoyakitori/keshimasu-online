# python/apps/force_patch_ga_reverse_word_mode.py

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GA_REVERSE = ROOT / "python" / "src" / "keshimasu_py" / "ga_reverse.py"
WORD_MODES = ROOT / "python" / "src" / "keshimasu_py" / "word_modes.py"


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
        raise ValueError(
            f"Unknown word mode: {mode}. Valid modes: {valid}, custom"
        )

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


NEW_PARSE_ARGS = r'''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GA reverse generator using keshimasu_py.solve_numba"
    )

    parser.add_argument("--words-file", default="")
    parser.add_argument(
        "--word-mode",
        choices=["country", "capital", "pokemon", "custom"],
        default="country",
        help="Word list mode. Use custom together with --words-file.",
    )

    parser.add_argument("--target-f", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)

    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--mutation", type=float, default=0.25)

    parser.add_argument("--max-nodes", type=int, default=300000)
    parser.add_argument("--max-depth", type=int, default=130)
    parser.add_argument("--max-paths-per-word", type=int, default=80)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--policy-trap-top-k", type=int, default=20)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "ga_reverse"),
    )
    parser.add_argument("--output-name", default="")

    parser.add_argument("--use-policy-trap", action="store_true")
    parser.add_argument("--policy-trap-weight", type=float, default=0.5)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()
'''


def ensure_word_modes_file():
    WORD_MODES.write_text(
        WORD_MODES_CODE.strip() + "\n",
        encoding="utf-8",
    )
    print("Wrote:", WORD_MODES)


def ensure_import(text):
    import_line = "from .word_modes import load_words_by_mode, resolve_words_file\n"

    if import_line in text:
        return text

    marker = "from .core_solver import solve_numba\n"

    if marker not in text:
        raise RuntimeError("Could not find import marker: from .core_solver import solve_numba")

    text = text.replace(
        marker,
        marker + import_line,
        1,
    )

    print("Added word_modes import.")
    return text


def replace_parse_args(text):
    pattern = re.compile(
        r"def parse_args\(\) -> argparse\.Namespace:.*?(?=\ndef _create_dummy_words_file\()",
        re.DOTALL,
    )

    new_text, count = pattern.subn(
        NEW_PARSE_ARGS.strip() + "\n\n",
        text,
        count=1,
    )

    if count == 0:
        raise RuntimeError("Could not replace parse_args().")

    print("Replaced parse_args().")
    return new_text


def patch_main_word_loading(text):
    if "word_mode = args.word_mode" in text:
        print("main() word loading already patched.")
        return text

    pattern = re.compile(
        r'''    if args\.words_file:\n'''
        r'''        words_file = Path\(args\.words_file\)\.resolve\(\)\n'''
        r'''    else:\n'''
        r'''        words_file = _create_dummy_words_file\(\)\n\n'''
        r'''    words = load_words_file\(words_file\)\n''',
        re.MULTILINE,
    )

    replacement = '''    if args.words_file:
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

    new_text, count = pattern.subn(
        replacement,
        text,
        count=1,
    )

    if count == 0:
        raise RuntimeError("Could not patch main() word loading block.")

    print("Patched main() word loading.")
    return new_text


def patch_prints(text):
    marker = '    print("Words file:", words_file)\n'

    if marker in text and '    print("Word mode:", word_mode)\n' not in text:
        text = text.replace(
            marker,
            marker + '    print("Word mode:", word_mode)\n',
            1,
        )
        print("Added Word mode print.")

    return text


def patch_generator_fields(text):
    marker = "    best = generator.run()\n"

    if marker in text and "generator.word_mode = word_mode" not in text:
        text = text.replace(
            marker,
            '    generator.word_mode = word_mode\n'
            '    generator.words_file_path = str(words_file)\n\n'
            + marker,
            1,
        )
        print("Added generator.word_mode fields.")

    return text


def patch_save_config(text):
    seed_marker = '                "seed": self.seed,\n'

    if seed_marker in text and '"wordMode": getattr(self, "word_mode", None),' not in text:
        text = text.replace(
            seed_marker,
            seed_marker
            + '                "wordMode": getattr(self, "word_mode", None),\n'
            + '                "wordsFile": getattr(self, "words_file_path", None),\n',
            1,
        )
        print("Added wordMode/wordsFile to save config.")

    return text


def main():
    if not GA_REVERSE.exists():
        raise FileNotFoundError(GA_REVERSE)

    ensure_word_modes_file()

    text = GA_REVERSE.read_text(encoding="utf-8")

    text = ensure_import(text)
    text = replace_parse_args(text)
    text = patch_main_word_loading(text)
    text = patch_prints(text)
    text = patch_generator_fields(text)
    text = patch_save_config(text)

    GA_REVERSE.write_text(text, encoding="utf-8")

    print("Patched:", GA_REVERSE)


if __name__ == "__main__":
    main()