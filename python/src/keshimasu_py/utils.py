# python/src/keshimasu_py/utils.py

from __future__ import annotations

from typing import Any, List, Optional

from .config import (
    EMPTY_DISPLAY,
    WILDCARD_CHAR,
    WILDCARD_DISPLAY,
    MAX_WORD_LENGTH,
)


def normalize_cell(cell: Any) -> Optional[str]:
    """
    表示用セルを内部表現用の文字へ正規化する。

    None / EMPTY_DISPLAY は空白として None を返す。
    WILDCARD_DISPLAY は WILDCARD_CHAR に戻す。
    """
    if cell is None:
        return None

    if cell == EMPTY_DISPLAY:
        return None

    if cell == WILDCARD_DISPLAY:
        return WILDCARD_CHAR

    return str(cell)


def split_word(word: str) -> List[str]:
    """
    単語を1文字ずつに分解する。

    カタカナの濁音・半濁音は Python 上では1文字として扱われる。
    """
    return list(word)


def word_length(word: str) -> int:
    return len(split_word(word))


def is_playable_word(
    word: str,
    min_len: int = 2,
    max_len: int = MAX_WORD_LENGTH,
) -> bool:
    n = word_length(word)
    return min_len <= n <= max_len


def normalize_board_rows(rows: list) -> list:
    """
    board rows を normalize_cell 済みの2次元配列にする。
    """
    normalized = []

    for row in rows:
        if isinstance(row, str):
            cells = row.split()
        else:
            cells = list(row)

        normalized.append([
            normalize_cell(cell)
            for cell in cells
        ])

    return normalized