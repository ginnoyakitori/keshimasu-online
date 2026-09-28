# python/tests/test_rule_sync.py

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PY_SRC = ROOT / "python" / "src"

if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))


from keshimasu_py.config import DIR_H, DIR_V  # noqa: E402
from keshimasu_py.core_solver import _remove_move_and_apply_gravity  # noqa: E402


CASE_DIR = ROOT / "data" / "testcases" / "rule_sync"


def normalize_cell(cell: Any) -> str | None:
    if cell is None:
        return None

    if cell == "" or cell == "・":
        return None

    if cell == "Ｆ":
        return "F"

    return str(cell)


def display_cell(cell: str | None) -> str:
    if cell is None:
        return "・"

    if cell == "F":
        return "Ｆ"

    return str(cell)


class SimpleCodec:
    def __init__(self) -> None:
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def add_char(self, ch: str) -> int:
        if ch not in self.char_to_id:
            idx = len(self.char_to_id) + 1
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

        return self.char_to_id[ch]


def build_simple_codec(*boards: List[List[Any]]) -> SimpleCodec:
    codec = SimpleCodec()

    for board in boards:
        for row in board:
            for cell in row:
                ch = normalize_cell(cell)

                if ch is not None:
                    codec.add_char(ch)

    return codec


def encode_board_simple(
    board_rows: List[List[Any]],
    codec: SimpleCodec,
) -> np.ndarray:
    rows = len(board_rows)
    cols = len(board_rows[0])

    out = np.zeros((rows, cols), dtype=np.int16)

    for r in range(rows):
        for c in range(cols):
            ch = normalize_cell(board_rows[r][c])

            if ch is None:
                out[r, c] = 0
            else:
                out[r, c] = codec.add_char(ch)

    return out


def decode_board_simple(
    board: np.ndarray,
    codec: SimpleCodec,
) -> List[List[str]]:
    rows: List[List[str]] = []

    for r in range(board.shape[0]):
        row: List[str] = []

        for c in range(board.shape[1]):
            value = int(board[r, c])

            if value == 0:
                row.append("・")
            else:
                row.append(display_cell(codec.id_to_char.get(value, "?")))

        rows.append(row)

    return rows


def board_to_text(
    board: np.ndarray,
    codec: SimpleCodec,
) -> str:
    rows = decode_board_simple(board, codec)
    return "\n".join(" ".join(row) for row in rows)


def load_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for path in sorted(CASE_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)

        case["_path"] = str(path)
        cases.append(case)

    return cases


def direction_from_path(path: List[List[int]]) -> int:
    if len(path) < 2:
        return int(DIR_H)

    r0, c0 = int(path[0][0]), int(path[0][1])
    r1, c1 = int(path[1][0]), int(path[1][1])

    if r1 == r0 and c1 == c0 + 1:
        return int(DIR_H)

    if r1 == r0 + 1 and c1 == c0:
        return int(DIR_V)

    raise ValueError(
        "Only H/V straight paths are supported in rule sync cases: "
        f"{path}"
    )


def make_encoded_move(case: Dict[str, Any]) -> np.ndarray:
    path = case["move"]["path"]

    row = int(path[0][0])
    col = int(path[0][1])
    direction = direction_from_path(path)

    return np.array(
        [0, row, col, direction],
        dtype=np.int16,
    )


def make_word_lengths(case: Dict[str, Any]) -> np.ndarray:
    path = case["move"]["path"]
    length = len(path)

    return np.array(
        [length],
        dtype=np.int16,
    )


def assert_board_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    codec: SimpleCodec,
    case_name: str,
) -> None:
    if not np.array_equal(actual, expected):
        actual_text = board_to_text(actual, codec)
        expected_text = board_to_text(expected, codec)

        raise AssertionError(
            f"\ncase={case_name}\n"
            f"actual:\n{actual_text}\n\n"
            f"expected:\n{expected_text}\n"
        )


def run_gravity_case(case: Dict[str, Any]) -> None:
    board_rows = case["board"]
    expected_rows = case["expectedBoardAfterGravity"]

    codec = build_simple_codec(
        board_rows,
        expected_rows,
    )

    board = encode_board_simple(board_rows, codec)
    expected_board = encode_board_simple(expected_rows, codec)

    encoded_move = make_encoded_move(case)
    word_lengths = make_word_lengths(case)

    actual_board = _remove_move_and_apply_gravity(
        board,
        encoded_move,
        word_lengths,
    )

    assert_board_equal(
        actual_board,
        expected_board,
        codec,
        case["name"],
    )


def run_no_repeat_case(case: Dict[str, Any]) -> None:
    used_words = set(case.get("usedWordsBefore", []))
    word = case["move"]["word"]
    expected = bool(case["expectedNoRepeatLegal"])

    actual = word not in used_words

    if actual != expected:
        raise AssertionError(
            f"case={case['name']} no-repeat mismatch: "
            f"word={word!r}, used={used_words}, "
            f"actual={actual}, expected={expected}"
        )


def test_rule_sync_cases() -> None:
    cases = load_cases()

    assert cases, f"No rule sync cases found in {CASE_DIR}"

    for case in cases:
        kind = case.get("kind", "gravity")

        if kind == "gravity":
            run_gravity_case(case)
        elif kind == "no-repeat":
            run_no_repeat_case(case)
        else:
            raise ValueError(
                f"Unknown rule sync case kind: {kind}"
            )


if __name__ == "__main__":
    test_rule_sync_cases()
    print("OK: Python rule sync cases passed.")