# python/src/keshimasu_py/config.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = PROJECT_ROOT / "python"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = DATA_DIR / "config"

PUZZLE_CONFIG_FILE = CONFIG_DIR / "puzzle-config.json"
CHAR_MASTER_FILE = CONFIG_DIR / "char-master.json"


DEFAULT_PUZZLE_CONFIG: Dict[str, Any] = {
    "ROWS": 8,
    "COLS": 5,
    "PLAYABLE_START_ROW": 3,
    "MAX_WORD_LENGTH": 5,
    "WILDCARD_CHAR": "F",
    "EMPTY_DISPLAY": "・",
    "WILDCARD_DISPLAY": "Ｆ",
    "DIRECTIONS": {
        "H": 0,
        "V": 1
    }
}


DEFAULT_CHAR_MASTER: Dict[str, Any] = {
    "systemChars": ["", "F"],
    "baseChars": [
        "ア", "イ", "ウ", "エ", "オ",
        "カ", "キ", "ク", "ケ", "コ",
        "サ", "シ", "ス", "セ", "ソ",
        "タ", "チ", "ツ", "テ", "ト",
        "ナ", "ニ", "ヌ", "ネ", "ノ",
        "ハ", "ヒ", "フ", "ヘ", "ホ",
        "マ", "ミ", "ム", "メ", "モ",
        "ヤ", "ユ", "ヨ",
        "ラ", "リ", "ル", "レ", "ロ",
        "ワ", "ヲ", "ン",
        "ー",
    ],
    "extraChars": [
        "ァ", "ィ", "ゥ", "ェ", "ォ",
        "ッ", "ャ", "ュ", "ョ", "ヮ",
        "ガ", "ギ", "グ", "ゲ", "ゴ",
        "ザ", "ジ", "ズ", "ゼ", "ゾ",
        "ダ", "ヂ", "ヅ", "デ", "ド",
        "バ", "ビ", "ブ", "ベ", "ボ",
        "パ", "ピ", "プ", "ペ", "ポ",
        "ヴ",
    ],
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_puzzle_config() -> Dict[str, Any]:
    if PUZZLE_CONFIG_FILE.exists():
        return load_json(PUZZLE_CONFIG_FILE)

    return dict(DEFAULT_PUZZLE_CONFIG)


def load_char_master() -> Dict[str, Any]:
    if CHAR_MASTER_FILE.exists():
        return load_json(CHAR_MASTER_FILE)

    return dict(DEFAULT_CHAR_MASTER)


PUZZLE_CONFIG = load_puzzle_config()

ROWS = int(PUZZLE_CONFIG["ROWS"])
COLS = int(PUZZLE_CONFIG["COLS"])

PLAYABLE_START_ROW = int(PUZZLE_CONFIG["PLAYABLE_START_ROW"])
PLAYABLE_ROWS = ROWS - PLAYABLE_START_ROW

MAX_WORD_LENGTH = int(PUZZLE_CONFIG["MAX_WORD_LENGTH"])

WILDCARD_CHAR = str(PUZZLE_CONFIG["WILDCARD_CHAR"])
EMPTY_DISPLAY = str(PUZZLE_CONFIG["EMPTY_DISPLAY"])
WILDCARD_DISPLAY = str(PUZZLE_CONFIG["WILDCARD_DISPLAY"])

DIR_H = int(PUZZLE_CONFIG["DIRECTIONS"]["H"])
DIR_V = int(PUZZLE_CONFIG["DIRECTIONS"]["V"])

# 固定IDルール
EMPTY_ID = 0
WILD_ID = 1