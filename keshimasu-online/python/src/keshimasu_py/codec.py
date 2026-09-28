# python/src/keshimasu_py/codec.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import (
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    WILDCARD_CHAR,
    WILDCARD_DISPLAY,
    EMPTY_DISPLAY,
    EMPTY_ID,
    WILD_ID,
    load_char_master,
)
from .utils import normalize_cell, split_word


@dataclass
class Codec:
    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]
    wild_id: int


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []

    for item in items:
        if item in seen:
            continue

        seen.add(item)
        out.append(item)

    return out


def load_master_chars() -> List[str]:
    """
    char-master.json から固定文字リストを作る。

    固定ルール:
      EMPTY_ID = 0 は内部専用なので char_to_id には入れない
      WILDCARD_CHAR = "F" は必ず ID 1
      その他は ID 2 以降
    """
    master = load_char_master()

    system_chars = master.get("systemChars", [])
    base_chars = master.get("baseChars", [])
    extra_chars = master.get("extraChars", [])

    chars: List[str] = []

    # systemChars のうち "" は EMPTY を意味するので char_to_id には入れない
    for ch in system_chars:
        if ch == "":
            continue

        chars.append(str(ch))

    for ch in base_chars:
        chars.append(str(ch))

    for ch in extra_chars:
        chars.append(str(ch))

    chars = _unique_keep_order(chars)

    # F は必ず先頭に固定する
    chars = [ch for ch in chars if ch != WILDCARD_CHAR]
    chars = [WILDCARD_CHAR] + chars

    return chars


def build_codec(
    words: List[str],
    wildcard_char: str = WILDCARD_CHAR,
) -> Codec:
    """
    文字 -> ID の Codec を作る。

    固定ルール:
      EMPTY_ID = 0 (内部専用、id_to_charには含めない)
      WILD_ID  = 1 (ワイルドカード文字)
      通常文字  = 2 以降

    char-master.json にある文字を先に登録し、
    words にだけ出てくる未知文字は末尾に追加する。
    """
    if wildcard_char != WILDCARD_CHAR:
        raise ValueError(
            f"wildcard_char must be {wildcard_char!r}, got {wildcard_char!r}"
        )

    master_chars = load_master_chars()

    char_to_id: Dict[str, int] = {}
    id_to_char: Dict[int, str] = {}

    # F は必ず ID 1
    char_to_id[WILDCARD_CHAR] = WILD_ID
    id_to_char[WILD_ID] = WILDCARD_CHAR

    next_id = 2

    for ch in master_chars:
        if ch == "":
            continue

        if ch == WILDCARD_CHAR:
            continue

        if ch in char_to_id:
            continue

        char_to_id[ch] = next_id
        id_to_char[next_id] = ch
        next_id += 1

    # words に含まれる未知文字を末尾に追加
    for word in words:
        for ch in split_word(word):
            if ch == "":
                continue

            if ch in char_to_id:
                continue

            char_to_id[ch] = next_id
            id_to_char[next_id] = ch
            next_id += 1

    wild_id = char_to_id[WILDCARD_CHAR]

    if wild_id != WILD_ID:
        raise RuntimeError(
            f"wild_id mismatch: expected {WILD_ID}, got {wild_id}"
        )

    return Codec(
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        wild_id=wild_id,
    )


def encode_words(
    words: List[str],
    codec: Codec,
    max_word_length: int = MAX_WORD_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    filtered = [
        word for word in words
        if 2 <= len(split_word(word)) <= max_word_length
    ]

    word_array = np.zeros(
        (len(filtered), max_word_length),
        dtype=np.int16,
    )

    word_lengths = np.zeros(
        (len(filtered),),
        dtype=np.int16,
    )

    for wi, word in enumerate(filtered):
        chars = split_word(word)
        word_lengths[wi] = len(chars)

        for i, ch in enumerate(chars):
            if ch not in codec.char_to_id:
                raise ValueError(f"unknown char in word: {ch}")

            word_array[wi, i] = codec.char_to_id[ch]

    return word_array, word_lengths, filtered


def encode_board(
    rows: List[Any],
    codec: Codec,
) -> np.ndarray:
    board = np.zeros((ROWS, COLS), dtype=np.int16)

    if len(rows) != ROWS:
        raise ValueError(f"board must have {ROWS} rows")

    for r in range(ROWS):
        row = rows[r]

        if isinstance(row, str):
            cells = row.split()
        else:
            cells = list(row)

        if len(cells) != COLS:
            raise ValueError(
                f"row {r} must have {COLS} cells: {cells}"
            )

        for c in range(COLS):
            cell = normalize_cell(cells[c])

            if cell is None:
                board[r, c] = EMPTY_ID
            else:
                if cell not in codec.char_to_id:
                    raise ValueError(f"unknown char in board: {cell}")

                board[r, c] = codec.char_to_id[cell]

    return board


def decode_cell(
    value: int,
    codec: Codec,
    display: bool = True,
) -> str:
    value = int(value)

    if value == EMPTY_ID:
        return EMPTY_DISPLAY if display else ""

    ch = codec.id_to_char.get(value, "?")

    if display and ch == WILDCARD_CHAR:
        return WILDCARD_DISPLAY

    return ch


def board_to_string(
    board: np.ndarray,
    codec: Codec,
) -> str:
    lines = []

    for r in range(board.shape[0]):
        cells = [
            decode_cell(int(board[r, c]), codec, display=True)
            for c in range(board.shape[1])
        ]

        lines.append(" ".join(cells))

    return "\n".join(lines)


def collect_board_chars(board_rows: List[List[Any]]) -> List[str]:
    chars: List[str] = []

    for row in board_rows:
        for cell in row:
            cell = normalize_cell(cell)

            if cell is None:
                continue

            chars.append(str(cell))

    return chars


def build_codec_for_board_and_words(
    words: List[str],
    board_rows: List[List[Any]],
) -> Codec:
    board_chars = collect_board_chars(board_rows)

    source = list(words)

    if board_chars:
        source.append("".join(board_chars))

    return build_codec(source, wildcard_char=WILDCARD_CHAR)


def validate_codec(codec: Codec) -> None:
    """
    開発時の安全確認用。

    必ず:
      EMPTY_ID = 0
      F = 1
    を満たすことを確認する。
    """
    if EMPTY_ID != 0:
        raise RuntimeError("EMPTY_ID must be 0")

    if codec.char_to_id.get(WILDCARD_CHAR) != WILD_ID:
        raise RuntimeError(
            f"{WILDCARD_CHAR} must be ID {WILD_ID}"
        )

    if codec.wild_id != WILD_ID:
        raise RuntimeError(
            f"codec.wild_id must be {WILD_ID}"
        )

    if EMPTY_ID in codec.id_to_char:
        raise RuntimeError("EMPTY_ID must not exist in id_to_char")