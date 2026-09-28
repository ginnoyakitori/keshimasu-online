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
