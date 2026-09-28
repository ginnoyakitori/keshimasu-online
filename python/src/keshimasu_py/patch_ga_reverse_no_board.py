# python/apps/patch_ga_reverse_no_board.py

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "python" / "src" / "keshimasu_py" / "ga_reverse.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    old = '''    best = generator.run()

    out_path = generator.save(
        best,
        output_name=args.output_name or None,
    )

    print("\\n=== BEST ===")
    print("Solved:", best.solved)
    print("Fitness:", best.fitness)
    print("Policy trap bonus:", best.policy_trap_bonus)
    print("Saved:", out_path)

    if best.board is not None:
        print("\\n=== BOARD ===\\n")
        print(board_to_string(best.board, generator.codec))
'''

    new = '''    best = generator.run()

    print("\\n=== BEST ===")
    print("Solved:", best.solved)
    print("Fitness:", best.fitness)
    print("Policy trap bonus:", best.policy_trap_bonus)

    if best.board is None:
        print("\\nNO VALID BOARD GENERATED.")
        print("No file was saved because the best chromosome has no board.")
        print("Try larger --population / --generations, or reduce --steps.")
        print("")
        print("Recommended quick retry:")
        print("python -m keshimasu_py.ga_reverse --target-f 3 --generations 10 --population 40 --steps 9 --use-policy-trap --policy-trap-weight 0.3")
        return

    out_path = generator.save(
        best,
        output_name=args.output_name or None,
    )

    print("Saved:", out_path)

    print("\\n=== BOARD ===\\n")
    print(board_to_string(best.board, generator.codec))
'''

    if old not in text:
        print("Target main() block not found. No change applied.")
        return

    text = text.replace(old, new, 1)
    TARGET.write_text(text, encoding="utf-8")

    print("Patched:", TARGET)


if __name__ == "__main__":
    main()
    