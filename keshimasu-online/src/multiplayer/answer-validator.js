"use strict";

const ROWS = 8;
const COLS = 5;
const VISIBLE_ROWS = 5;
const PLAYABLE_START_ROW = ROWS - VISIBLE_ROWS;

const EMPTY = "・";
const WILDCARDS = new Set(["F", "Ｆ"]);


/* ==================================================
   基本処理
================================================== */

function toCharacters(value) {
  return [...String(value ?? "")];
}


function normalizeWildcard(value) {
  const text = String(value ?? "");

  if (text === "F" || text === "Ｆ") {
    return "F";
  }

  return text;
}


function validateBoard(board) {
  return (
    Array.isArray(board) &&
    board.length === ROWS &&
    board.every(
      (row) =>
        Array.isArray(row) &&
        row.length === COLS
    )
  );
}


/* ==================================================
   選択座標
================================================== */

function normalizePath(path) {
  if (!Array.isArray(path)) {
    return null;
  }

  const normalized = [];

  for (const position of path) {
    if (
      !Array.isArray(position) ||
      position.length !== 2
    ) {
      return null;
    }

    const row = Number(position[0]);
    const col = Number(position[1]);

    if (
      !Number.isInteger(row) ||
      !Number.isInteger(col)
    ) {
      return null;
    }

    normalized.push([row, col]);
  }

  return normalized;
}


function validatePath(path) {
  if (
    !Array.isArray(path) ||
    path.length < 2 ||
    path.length > 5
  ) {
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

    if (row < PLAYABLE_START_ROW) {
      return "表示されている下5行から選択してください";
    }
  }

  const uniquePositions = new Set(
    path.map(([row, col]) => `${row},${col}`)
  );

  if (uniquePositions.size !== path.length) {
    return "同じマスを2回選択することはできません";
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

  for (
    let index = 1;
    index < path.length;
    index += 1
  ) {
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


/*
 * 横は左から右、縦は上から下へ並べます。
 */
function sortPathInReadingOrder(path) {
  if (!Array.isArray(path)) {
    return [];
  }

  const sortedPath = path.map(
    ([row, col]) => [row, col]
  );

  if (sortedPath.length < 2) {
    return sortedPath;
  }

  const sameRow = sortedPath.every(
    ([row]) => row === sortedPath[0][0]
  );

  const sameColumn = sortedPath.every(
    ([, col]) => col === sortedPath[0][1]
  );

  if (sameRow) {
    sortedPath.sort(
      (left, right) => left[1] - right[1]
    );
  } else if (sameColumn) {
    sortedPath.sort(
      (left, right) => left[0] - right[0]
    );
  }

  return sortedPath;
}


/* ==================================================
   完成する国名を構築
================================================== */

function buildCompletedWord({
  board,
  path,
  fText,
}) {
  const selectedCharacters = [];

  for (const [row, col] of path) {
    const cell = normalizeWildcard(
      board[row]?.[col]
    );

    if (!cell || cell === EMPTY) {
      return {
        ok: false,
        error: "空マスは選択できません",
      };
    }

    selectedCharacters.push(cell);
  }

  const wildcardCount =
    selectedCharacters.filter(
      (character) =>
        WILDCARDS.has(character)
    ).length;

  const replacementCharacters =
    toCharacters(
      String(fText ?? "").trim()
    );

  if (
    wildcardCount === 0 &&
    replacementCharacters.length > 0
  ) {
    return {
      ok: false,
      error:
        "Fが含まれていないため、補完文字の入力は不要です",
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
    if (WILDCARDS.has(character)) {
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
    pattern: selectedCharacters.join(""),
    wildcardCount,
  };
}


/* ==================================================
   重力
================================================== */

function applyGravity(board) {
  for (let col = 0; col < COLS; col += 1) {
    const remainingCharacters = [];

    for (
      let row = ROWS - 1;
      row >= 0;
      row -= 1
    ) {
      const character = board[row][col];

      if (
        character !== "" &&
        character !== EMPTY
      ) {
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
      if (cell !== "" && cell !== EMPTY) {
        count += 1;
      }
    }
  }

  return count;
}


/* ==================================================
   正誤判定
================================================== */

function validateAndApply({
  board,
  path,
  fText = "",
  usedWords,
  isCountry,
}) {
  if (!validateBoard(board)) {
    return {
      ok: false,
      error: "サーバー側の盤面情報が不正です",
    };
  }

  if (!(usedWords instanceof Set)) {
    return {
      ok: false,
      error: "使用済み単語情報が不正です",
    };
  }

  if (typeof isCountry !== "function") {
    return {
      ok: false,
      error: "国名判定処理が設定されていません",
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

  /*
   * 選択操作の方向に関係なく読み順を統一します。
   */
  const readingPath =
    sortPathInReadingOrder(normalizedPath);

  const wordResult = buildCompletedWord({
    board,
    path: readingPath,
    fText,
  });

  if (!wordResult.ok) {
    return wordResult;
  }

  const completedWord = wordResult.word;

  if (
    toCharacters(completedWord).length !==
    readingPath.length
  ) {
    return {
      ok: false,
      error:
        "完成した国名の文字数を確認できませんでした",
      completedWord,
    };
  }

  if (!isCountry(completedWord)) {
    return {
      ok: false,
      error:
        `「${completedWord}」は国名リストにありません`,
      completedWord,
      pattern: wordResult.pattern,
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

  for (const [row, col] of readingPath) {
    board[row][col] = EMPTY;
  }

  applyGravity(board);
  usedWords.add(completedWord);

  const remaining = remainingCells(board);

  return {
    ok: true,
    word: completedWord,
    pattern: wordResult.pattern,
    wildcardCount: wordResult.wildcardCount,
    board,
    remaining,
    cleared: remaining === 0,
  };
}


module.exports = {
  validateAndApply,
  remainingCells,
  applyGravity,
  sortPathInReadingOrder,
};