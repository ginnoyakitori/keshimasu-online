"use strict";

/*
 * 回答判定と盤面更新。
 *
 * 盤面は board[row][col] の2次元配列で、空きマスは "" です。
 * （ポケモンの「カプ・コケコ」の「・」と区別するため、空きマスに「・」は使いません）
 * 盤面の大きさは問題ごとに異なり、board の形から求めます。
 */

const EMPTY = "";
const WILDCARDS = new Set(["F", "Ｆ"]);

const MIN_SELECTION = 2;
const MAX_SELECTION = 5;


/* ==================================================
   文字の正規化
================================================== */

function toCharacters(value) {
  return [...String(value ?? "")];
}

/*
 * Fに入力された文字を、盤面と同じ表記へそろえます。
 *
 *  - 全角・半角をNFKCで統一（Ｚ→Z、２→2、･→・）
 *  - ひらがなをカタカナへ変換
 *  - 英字は大文字へ（ポリゴンZ 用）
 */
function normalizeInput(value) {
  const normalized = String(value ?? "").normalize("NFKC");

  return [...normalized]
    .map((character) => {
      const code = character.codePointAt(0);

      if (code >= 0x3041 && code <= 0x3096) {
        return String.fromCodePoint(code + 0x60);
      }

      return character.toUpperCase();
    })
    .join("");
}

// 以前の名前との互換用
const hiraganaToKatakana = normalizeInput;

function normalizeWildcard(value) {
  const text = String(value ?? "");

  return WILDCARDS.has(text) ? "F" : text;
}


/* ==================================================
   盤面
================================================== */

function boardSize(board) {
  if (
    !Array.isArray(board) ||
    board.length === 0 ||
    !Array.isArray(board[0]) ||
    board[0].length === 0
  ) {
    return null;
  }

  const cols = board[0].length;

  if (!board.every((row) => Array.isArray(row) && row.length === cols)) {
    return null;
  }

  return { rows: board.length, cols };
}

function applyGravity(board) {
  const size = boardSize(board);

  if (!size) {
    return;
  }

  for (let col = 0; col < size.cols; col += 1) {
    const remaining = [];

    for (let row = size.rows - 1; row >= 0; row -= 1) {
      if (board[row][col] !== EMPTY) {
        remaining.push(board[row][col]);
      }
    }

    for (let row = size.rows - 1, index = 0; row >= 0; row -= 1, index += 1) {
      board[row][col] = index < remaining.length ? remaining[index] : EMPTY;
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


/* ==================================================
   選択座標
================================================== */

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

function validatePath(path, size, visibleRows) {
  if (
    !Array.isArray(path) ||
    path.length < MIN_SELECTION ||
    path.length > MAX_SELECTION
  ) {
    return `${MIN_SELECTION}〜${MAX_SELECTION}マスを選択してください`;
  }

  const firstVisibleRow = size.rows - visibleRows;

  for (const [row, col] of path) {
    if (row < 0 || row >= size.rows || col < 0 || col >= size.cols) {
      return "盤面外のマスが含まれています";
    }

    if (row < firstVisibleRow) {
      return "表示されているマスから選択してください";
    }
  }

  if (new Set(path.map(([row, col]) => `${row},${col}`)).size !== path.length) {
    return "同じマスを2回選択することはできません";
  }

  const rowStep = path[1][0] - path[0][0];
  const colStep = path[1][1] - path[0][1];

  const horizontal = rowStep === 0 && Math.abs(colStep) === 1;
  const vertical = colStep === 0 && Math.abs(rowStep) === 1;

  if (!horizontal && !vertical) {
    return "縦または横に一直線で選択してください";
  }

  for (let index = 1; index < path.length; index += 1) {
    const previous = path[index - 1];
    const current = path[index];

    if (
      current[0] - previous[0] !== rowStep ||
      current[1] - previous[1] !== colStep
    ) {
      return "選択したマスが一直線に並んでいません";
    }
  }

  return null;
}

// 横は左から右、縦は上から下の読み順にそろえます。
function sortPathInReadingOrder(path) {
  if (!Array.isArray(path)) {
    return [];
  }

  const sorted = path.map(([row, col]) => [row, col]);

  if (sorted.length < 2) {
    return sorted;
  }

  if (sorted.every(([row]) => row === sorted[0][0])) {
    sorted.sort((left, right) => left[1] - right[1]);
  } else if (sorted.every(([, col]) => col === sorted[0][1])) {
    sorted.sort((left, right) => left[0] - right[0]);
  }

  return sorted;
}


/* ==================================================
   完成する単語を組み立てる
================================================== */

function buildCompletedWord({ board, path, fText }) {
  const selected = [];

  for (const [row, col] of path) {
    const cell = normalizeWildcard(board[row]?.[col]);

    if (!cell || cell === EMPTY) {
      return { ok: false, error: "空マスは選択できません" };
    }

    selected.push(cell);
  }

  const wildcardCount = selected.filter((character) =>
    WILDCARDS.has(character)
  ).length;

  const replacements = toCharacters(normalizeInput(String(fText ?? "").trim()));

  if (wildcardCount === 0 && replacements.length > 0) {
    return {
      ok: false,
      error: "Fが含まれていないため、補完文字の入力は不要です",
    };
  }

  if (wildcardCount > 0 && replacements.length !== wildcardCount) {
    return {
      ok: false,
      error: `Fに入る文字を${wildcardCount}文字入力してください`,
      wildcardCount,
    };
  }

  let replacementIndex = 0;

  const completed = selected.map((character) => {
    if (WILDCARDS.has(character)) {
      const replacement = replacements[replacementIndex];

      replacementIndex += 1;

      return replacement;
    }

    return character;
  });

  return {
    ok: true,
    word: completed.join(""),
    pattern: selected.join(""),
    wildcardCount,
  };
}


/* ==================================================
   正誤判定と盤面更新
================================================== */

/*
 * @param {string[][]} board     現在の盤面（成功時はこの配列を直接更新します）
 * @param {number[][]} path      選択したマス [行, 列]
 * @param {string}     fText     Fに入れる文字
 * @param {Set}        usedWords この盤面で使用済みの単語
 * @param {Function}   isWord    単語が辞書にあるか
 * @param {number}     visibleRows 見えている下側の行数
 */
function validateAndApply({
  board,
  path,
  fText = "",
  usedWords,
  isWord,
  visibleRows,
}) {
  const size = boardSize(board);

  if (!size) {
    return { ok: false, error: "サーバー側の盤面情報が不正です" };
  }

  if (!(usedWords instanceof Set)) {
    return { ok: false, error: "使用済み単語情報が不正です" };
  }

  if (typeof isWord !== "function") {
    return { ok: false, error: "単語の判定処理が設定されていません" };
  }

  const visible = Number.isInteger(visibleRows)
    ? Math.min(visibleRows, size.rows)
    : Math.min(5, size.rows);

  const normalizedPath = normalizePath(path);
  const pathError = validatePath(normalizedPath, size, visible);

  if (pathError) {
    return { ok: false, error: pathError };
  }

  const readingPath = sortPathInReadingOrder(normalizedPath);
  const wordResult = buildCompletedWord({ board, path: readingPath, fText });

  if (!wordResult.ok) {
    return wordResult;
  }

  const word = wordResult.word;

  if (toCharacters(word).length !== readingPath.length) {
    return {
      ok: false,
      error: "完成したワードの文字数を確認できませんでした",
    };
  }

  if (!isWord(word)) {
    return {
      ok: false,
      error: `「${word}」はリストにありません`,
      pattern: wordResult.pattern,
    };
  }

  if (usedWords.has(word)) {
    return {
      ok: false,
      error: `「${word}」はこの問題ですでに使用しています`,
    };
  }

  for (const [row, col] of readingPath) {
    board[row][col] = EMPTY;
  }

  applyGravity(board);
  usedWords.add(word);

  const remaining = remainingCells(board);

  return {
    ok: true,
    word,
    pattern: wordResult.pattern,
    wildcardCount: wordResult.wildcardCount,
    board,
    remaining,
    cleared: remaining === 0,
  };
}

module.exports = {
  EMPTY,
  validateAndApply,
  remainingCells,
  applyGravity,
  sortPathInReadingOrder,
  normalizeInput,
  hiraganaToKatakana,
};
