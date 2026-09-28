"use strict";

const ROWS = 8;
const COLS = 5;
const PLAYABLE_START_ROW = 3;

const EMPTY = "・";
const WILDS = new Set(["F", "Ｆ"]);


function toCharacters(value) {
  return [...String(value ?? "")];
}


function normalizePath(path) {
  if (!Array.isArray(path)) {
    return null;
  }

  const normalized = [];

  for (const position of path) {
    if (!Array.isArray(position) || position.length !== 2) {
      return null;
    }

    const row = Number(position[0]);
    const col = Number(position[1]);

    if (!Number.isInteger(row) || !Number.isInteger(col)) {
      return null;
    }

    normalized.push([row, col]);
  }

  return normalized;
}


function validatePath(path) {
  if (!path || path.length < 2 || path.length > 5) {
    return "2〜5マスを選択してください";
  }

  for (const [row, col] of path) {
    if (
      row < 0 ||
      row >= ROWS ||
      col < 0 ||
      col >= COLS
    ) {
      return "盤面外のマスが含まれています";
    }
  }

  if (path[0][0] < PLAYABLE_START_ROW) {
    return "表示されている下5段から選択してください";
  }

  const rowDirection =
    path[1][0] - path[0][0];

  const colDirection =
    path[1][1] - path[0][1];

  const horizontal =
    rowDirection === 0 &&
    Math.abs(colDirection) === 1;

  const vertical =
    colDirection === 0 &&
    Math.abs(rowDirection) === 1;

  if (!horizontal && !vertical) {
    return "縦または横に一直線で選択してください";
  }

  for (let index = 1; index < path.length; index += 1) {
    const previous = path[index - 1];
    const current = path[index];

    if (
      current[0] - previous[0] !== rowDirection ||
      current[1] - previous[1] !== colDirection
    ) {
      return "選択したマスが一直線に並んでいません";
    }
  }

  return null;
}


function countWildcards(board, path) {
  let count = 0;

  for (const [row, col] of path) {
    if (WILDS.has(board[row][col])) {
      count += 1;
    }
  }

  return count;
}


function buildCompletedWord({
  board,
  path,
  fText,
}) {
  const replacementCharacters =
    toCharacters(fText);

  const selectedCharacters = [];

  for (const [row, col] of path) {
    const cell = board[row][col];

    if (cell === EMPTY) {
      return {
        ok: false,
        error: "空マスは選択できません",
      };
    }

    selectedCharacters.push(cell);
  }

  const wildcardCount = selectedCharacters.filter(
    (character) => WILDS.has(character)
  ).length;

  if (
    wildcardCount === 0 &&
    replacementCharacters.length > 0
  ) {
    return {
      ok: false,
      error: "Fがないため補完文字は不要です",
    };
  }

  if (
    wildcardCount > 0 &&
    replacementCharacters.length !== wildcardCount
  ) {
    return {
      ok: false,
      error:
        `Fに入る文字を${wildcardCount}文字入力してください`,
      wildcardCount,
    };
  }

  const completedCharacters = [];
  let replacementIndex = 0;

  for (const character of selectedCharacters) {
    if (WILDS.has(character)) {
      completedCharacters.push(
        replacementCharacters[replacementIndex]
      );

      replacementIndex += 1;
    } else {
      completedCharacters.push(character);
    }
  }

  return {
    ok: true,
    word: completedCharacters.join(""),
    selectedText: selectedCharacters.join(""),
    wildcardCount,
  };
}


function applyGravity(board) {
  for (let col = 0; col < COLS; col += 1) {
    const remainingCharacters = [];

    for (let row = ROWS - 1; row >= 0; row -= 1) {
      const character = board[row][col];

      if (character !== EMPTY) {
        remainingCharacters.push(character);
      }
    }

    for (
      let row = ROWS - 1, index = 0;
      row >= 0;
      row -= 1, index += 1
    ) {
      board[row][col] =
        index < remainingCharacters.length
          ? remainingCharacters[index]
          : EMPTY;
    }
  }
}


function remainingCells(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell !== EMPTY) {
        count += 1;
      }
    }
  }

  return count;
}


function validateAndApply({
  board,
  path,
  fText,
  usedWords,
  isCountry,
}) {
  if (!Array.isArray(board) || board.length !== ROWS) {
    return {
      ok: false,
      error: "サーバー側の盤面情報が不正です",
    };
  }

  const normalizedPath = normalizePath(path);
  const pathError = validatePath(normalizedPath);

  if (pathError) {
    return {
      ok: false,
      error: pathError,
    };
  }

  const wordResult = buildCompletedWord({
    board,
    path: normalizedPath,
    fText,
  });

  if (!wordResult.ok) {
    return wordResult;
  }

  const completedWord = wordResult.word;

  if (!isCountry(completedWord)) {
    return {
      ok: false,
      error:
        `「${completedWord}」は国名リストにありません`,
      completedWord,
    };
  }

  if (usedWords.has(completedWord)) {
    return {
      ok: false,
      error:
        `「${completedWord}」はこの問題ですでに使用しています`,
      completedWord,
    };
  }

  for (const [row, col] of normalizedPath) {
    board[row][col] = EMPTY;
  }

  applyGravity(board);
  usedWords.add(completedWord);

  const remaining = remainingCells(board);

  return {
    ok: true,
    word: completedWord,
    board,
    remaining,
    cleared: remaining === 0,
    wildcardCount: wordResult.wildcardCount,
  };
}


module.exports = {
  validateAndApply,
  remainingCells,
  countWildcards,
};