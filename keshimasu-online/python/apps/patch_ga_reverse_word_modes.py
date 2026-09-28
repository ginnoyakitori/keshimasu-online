# python/apps/patch_ga_reverse_word_modes.py

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "python" / "src" / "keshimasu_py" / "ga_reverse.py"


def main():
    text = TARGET.read_text(encoding="utf-8")

    # import 追加
    import_line = "from .word_modes import load_words_by_mode, resolve_words_file\n"

    if import_line not in text:
        marker = "from .core_solver import solve_numba\n"

        if marker not in text:
            raise RuntimeError("Could not find core_solver import marker.")

        text = text.replace(
            marker,
            marker + import_line,
            1,
        )

        print("Added word_modes import.")

    # argparse に --word-mode 追加
    if "--word-mode" not in text:
        marker = '    parser.add_argument("--words-file", default="")\n'

        if marker not in text:
            raise RuntimeError("Could not find --words-file argument.")

        insert = (
            '    parser.add_argument(\n'
            '        "--word-mode",\n'
            '        choices=["country", "capital", "pokemon", "custom"],\n'
            '        default="country",\n'
            '        help="Word list mode. Use custom with --words-file.",\n'
            '    )\n'
        )

        text = text.replace(
            marker,
            marker + insert,
            1,
        )

        print("Added --word-mode argument.")

    # main() の words_file / words 読み込みブロックを差し替え
    old_block = '''    if args.words_file:
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
        word_mode = args.word_mode
        words_file = resolve_words_file(word_mode, PROJECT_ROOT)
        words = load_words_by_mode(word_mode, PROJECT_ROOT)
'''

    if old_block in text:
        text = text.replace(old_block, new_block, 1)
        print("Replaced word loading block.")
    else:
        print("Word loading block not found. Skipped block replacement.")

    # print に Word mode を追加
    print_marker = '    print("Words file:", words_file)\n'

    if print_marker in text and '    print("Word mode:", word_mode)\n' not in text:
        text = text.replace(
            print_marker,
            print_marker + '    print("Word mode:", word_mode)\n',
            1,
        )

        print("Added Word mode print.")

    # generator.run() 前に generator へ word_mode / words_file_path を持たせる
    run_marker = "    best = generator.run()\n"

    if run_marker in text and "generator.word_mode = word_mode" not in text:
        text = text.replace(
            run_marker,
            '    generator.word_mode = word_mode\n'
            '    generator.words_file_path = str(words_file)\n\n'
            + run_marker,
            1,
        )

        print("Added generator word mode fields.")

    # save() の config に wordMode / wordsFile を入れる
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

    TARGET.write_text(text, encoding="utf-8")

    print("Patched:", TARGET)


if __name__ == "__main__":
    main()