"use strict";

/*
 * 盤面サイズを変えられる問題生成器です。
 *
 * 逆向き構築: 単語を列の下から積み上げて盤面を作り、積んだ順に消せることを
 * 想定解として保存します。そのため生成された盤面は必ず全消しできます。
 *
 * 生成はオフラインで行い、結果を data/puzzles/pools/ に保存して使います
 * （tools/generate-pools.js）。対戦中にサーバーで生成することはありません。
 */

const crypto = require("node:crypto");

const EMPTY = "";
const WILDCARD = "F";


/* ==================================================
   汎用処理
================================================== */

function randomInt(min, max) {
  if (max <= min) {
    return min;
  }

  return crypto.randomInt(min, max + 1);
}

function shuffle(values) {
  const result = values.slice();

  for (let index = result.length - 1; index > 0; index -= 1) {
    const other = randomInt(0, index);

    [result[index], result[other]] = [result[other], result[index]];
  }

  return result;
}

function cloneBoard(board) {
  return board.map((row) => row.slice());
}

function countRemaining(board) {
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

function applyGravity(board) {
  const rows = board.length;
  const cols = board[0].length;

  for (let col = 0; col < cols; col += 1) {
    const characters = [];

    for (let row = rows - 1; row >= 0; row -= 1) {
      if (board[row][col] !== EMPTY) {
        characters.push(board[row][col]);
      }
    }

    for (let row = rows - 1, index = 0; row >= 0; row -= 1, index += 1) {
      board[row][col] =
        index < characters.length ? characters[index] : EMPTY;
    }
  }
}

function matchesPattern(pattern, word) {
  const patternCharacters = [...pattern];
  const wordCharacters = [...word];

  if (patternCharacters.length !== wordCharacters.length) {
    return false;
  }

  return patternCharacters.every(
    (character, index) =>
      character === WILDCARD || character === wordCharacters[index]
  );
}


/* ==================================================
   辞書
================================================== */

function normalizeWords(words, minLength, maxLength) {
  return [
    ...new Set(
      (Array.isArray(words) ? words : [])
        .filter((word) => typeof word === "string")
        .map((word) => word.trim())
        .filter((word) => {
          const length = [...word].length;

          return length >= minLength && length <= maxLength;
        })
    ),
  ];
}

function groupByLength(words) {
  const byLength = new Map();

  for (const word of words) {
    const length = [...word].length;

    if (!byLength.has(length)) {
      byLength.set(length, []);
    }

    byLength.get(length).push(word);
  }

  return byLength;
}

function pickUnusedWord(words, usedWords) {
  for (const word of shuffle(words)) {
    if (!usedWords.has(word)) {
      return word;
    }
  }

  return null;
}


/* ==================================================
   積み上げ（逆向き構築）
================================================== */

/*
 * columns[col][0] が最下段の文字です。
 */
function createMoves({ columns, cfg, byLength, usedWords }) {
  const moves = [];
  const { rows, cols, visibleRows, maxWordLength } = cfg;

  // 横: 同じ高さの連続した列へ1文字ずつ積む
  for (let length = 2; length <= Math.min(maxWordLength, cols); length += 1) {
    const words = byLength.get(length) || [];

    if (words.length === 0) {
      continue;
    }

    for (let startCol = 0; startCol <= cols - length; startCol += 1) {
      const height = columns[startCol].length;
      let canPlace = height < rows;

      for (let col = startCol; canPlace && col < startCol + length; col += 1) {
        if (columns[col].length !== height) {
          canPlace = false;
        }
      }

      if (!canPlace) {
        continue;
      }

      const word = pickUnusedWord(words, usedWords);

      if (word) {
        moves.push({ direction: "H", word, length, startCol, height });
      }
    }
  }

  // 縦: 1つの列へ積む。消すときに見えている行に収まる長さだけ
  for (
    let length = 2;
    length <= Math.min(maxWordLength, visibleRows);
    length += 1
  ) {
    const words = byLength.get(length) || [];

    if (words.length === 0) {
      continue;
    }

    for (let col = 0; col < cols; col += 1) {
      if (columns[col].length + length > rows) {
        continue;
      }

      const word = pickUnusedWord(words, usedWords);

      if (word) {
        moves.push({
          direction: "V",
          word,
          length,
          col,
          height: columns[col].length,
        });
      }
    }
  }

  return moves;
}

function applyMove(columns, move) {
  const characters = [...move.word];

  if (move.direction === "H") {
    characters.forEach((character, index) => {
      columns[move.startCol + index].push(character);
    });

    return;
  }

  // 縦単語は先頭の文字が上になるよう、逆順に積む
  for (let index = characters.length - 1; index >= 0; index -= 1) {
    columns[move.col].push(characters[index]);
  }
}

function scoreMove({ move, rawMoves, columns, cfg }) {
  const horizontal = rawMoves.filter((entry) => entry.direction === "H").length;
  const vertical = rawMoves.length - horizontal;

  let score = Math.random() * 20;

  if (horizontal < cfg.minHorizontalMoves) {
    score += move.direction === "H" ? 300 : -80;
  }

  if (vertical < cfg.minVerticalMoves && move.direction === "V") {
    score += 140;
  }

  const verticalRate =
    (vertical + (move.direction === "V" ? 1 : 0)) / (rawMoves.length + 1);

  if (verticalRate > cfg.maxVerticalMoveRate) {
    score -= (verticalRate - cfg.maxVerticalMoveRate) * 500;
  }

  score += move.length * 15;

  const testColumns = columns.map((column) => column.slice());

  applyMove(testColumns, move);

  const heights = testColumns.map((column) => column.length);

  score -= (Math.max(...heights) - Math.min(...heights)) * 14;

  if (move.direction === "H") {
    score += 35;
  }

  return score;
}

function buildBoardFromColumns(columns, cfg) {
  const board = Array.from({ length: cfg.rows }, () =>
    Array(cfg.cols).fill(EMPTY)
  );

  for (let col = 0; col < cfg.cols; col += 1) {
    if (columns[col].length !== cfg.rows) {
      return null;
    }

    columns[col].forEach((character, index) => {
      board[cfg.rows - 1 - index][col] = character;
    });
  }

  return board;
}

function buildSolution(rawMoves, cfg) {
  return rawMoves.map((move) =>
    move.direction === "H"
      ? [move.word, "H", cfg.rows - 1, move.startCol]
      : [move.word, "V", cfg.rows - move.length, move.col]
  );
}

function constructBoard({ cfg, byLength }) {
  const columns = Array.from({ length: cfg.cols }, () => []);
  const usedWords = new Set();
  const rawMoves = [];
  const totalCells = cfg.rows * cfg.cols;

  for (let step = 0; step < totalCells; step += 1) {
    if (columns.every((column) => column.length === cfg.rows)) {
      break;
    }

    const moves = createMoves({ columns, cfg, byLength, usedWords });

    if (moves.length === 0) {
      return null;
    }

    const scored = moves
      .map((move) => ({
        move,
        score: scoreMove({ move, rawMoves, columns, cfg }),
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 8);

    const move = scored[randomInt(0, scored.length - 1)].move;

    applyMove(columns, move);
    rawMoves.push(move);
    usedWords.add(move.word);
  }

  const board = buildBoardFromColumns(columns, cfg);

  if (!board) {
    return null;
  }

  return { board, solution: buildSolution(rawMoves, cfg), rawMoves };
}


/* ==================================================
   想定解の再生（生成時の検証・検証ツールで共用）
================================================== */

function replaySolution({ board, solution, cfg, isWord }) {
  const work = cloneBoard(board);
  const used = new Set();
  const firstVisibleRow = cfg.rows - cfg.visibleRows;

  for (const [word, direction, startRow, startCol] of solution) {
    const length = [...word].length;
    const cells = [];

    for (let index = 0; index < length; index += 1) {
      cells.push(
        direction === "H"
          ? [startRow, startCol + index]
          : [startRow + index, startCol]
      );
    }

    if (used.has(word)) {
      return { ok: false, reason: `duplicate word ${word}` };
    }

    if (typeof isWord === "function" && !isWord(word)) {
      return { ok: false, reason: `not in dictionary ${word}` };
    }

    for (const [row, col] of cells) {
      if (
        row < firstVisibleRow ||
        row >= cfg.rows ||
        col < 0 ||
        col >= cfg.cols ||
        work[row][col] === EMPTY
      ) {
        return { ok: false, reason: `bad cell for ${word}` };
      }
    }

    const pattern = cells.map(([row, col]) => work[row][col]).join("");

    if (!matchesPattern(pattern, word)) {
      return { ok: false, reason: `pattern mismatch ${pattern}/${word}` };
    }

    for (const [row, col] of cells) {
      work[row][col] = EMPTY;
    }

    applyGravity(work);
    used.add(word);
  }

  return {
    ok: countRemaining(work) === 0,
    remaining: countRemaining(work),
  };
}


/* ==================================================
   Fの配置・難易度条件・縦だけ判定
================================================== */

function addWildcards(board, count) {
  const result = cloneBoard(board);
  const positions = shuffle(
    Array.from({ length: board.length * board[0].length }, (_, index) => [
      Math.floor(index / board[0].length),
      index % board[0].length,
    ])
  );

  for (const [row, col] of positions.slice(0, count)) {
    result[row][col] = WILDCARD;
  }

  return result;
}

function boardKey(board, usedWords) {
  return (
    board.map((row) => row.join("") || "_").join("/") +
    "::" +
    [...usedWords].sort().join(",")
  );
}

function canClearWithVerticalOnly({ board, byLength, cfg, maxNodes }) {
  let nodes = 0;
  const visited = new Set();
  const firstVisibleRow = cfg.rows - cfg.visibleRows;

  function candidates(current, used) {
    const found = [];

    for (let col = 0; col < cfg.cols; col += 1) {
      for (let startRow = firstVisibleRow; startRow < cfg.rows; startRow += 1) {
        let pattern = "";

        for (let endRow = startRow; endRow < cfg.rows; endRow += 1) {
          const character = current[endRow][col];

          if (character === EMPTY) {
            break;
          }

          pattern += character;

          const length = [...pattern].length;

          if (length > cfg.maxWordLength) {
            break;
          }

          if (length < 2) {
            continue;
          }

          for (const word of byLength.get(length) || []) {
            if (!used.has(word) && matchesPattern(pattern, word)) {
              found.push({ word, col, startRow, endRow });
            }
          }
        }
      }
    }

    return found;
  }

  function search(current, used) {
    nodes += 1;

    if (nodes > maxNodes) {
      // 打ち切ったときは安全側（縦だけで解ける扱い）にして除外する
      return true;
    }

    if (countRemaining(current) === 0) {
      return true;
    }

    const key = boardKey(current, used);

    if (visited.has(key)) {
      return false;
    }

    visited.add(key);

    for (const candidate of candidates(current, used)) {
      const next = cloneBoard(current);

      for (let row = candidate.startRow; row <= candidate.endRow; row += 1) {
        next[row][candidate.col] = EMPTY;
      }

      applyGravity(next);

      if (search(next, new Set([...used, candidate.word]))) {
        return true;
      }
    }

    return false;
  }

  return search(cloneBoard(board), new Set());
}


/* ==================================================
   1問の生成
================================================== */

/*
 * @param {object} cfg    levels.js のレベル設定
 * @param {string[]} words 使用できる単語
 * @returns {object|null}  生成できなければ null
 */
function generatePuzzle({
  cfg,
  words,
  maxAttempts = 400,
  verticalOnlyMaxNodes = 30000,
}) {
  const usable = normalizeWords(words, 2, cfg.maxWordLength);
  const byLength = groupByLength(usable);
  const isWord = (() => {
    const set = new Set(usable);

    return (word) => set.has(word);
  })();

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const built = constructBoard({ cfg, byLength });

    if (!built) {
      continue;
    }

    const steps = built.rawMoves.length;
    const horizontalMoves = built.rawMoves.filter(
      (move) => move.direction === "H"
    ).length;
    const verticalMoves = steps - horizontalMoves;

    if (
      steps < cfg.minSteps ||
      steps > cfg.maxSteps ||
      horizontalMoves < cfg.minHorizontalMoves ||
      verticalMoves < cfg.minVerticalMoves ||
      verticalMoves / steps > cfg.maxVerticalMoveRate
    ) {
      continue;
    }

    const wildcardCount = randomInt(cfg.minWildcards, cfg.maxWildcards);
    const board = addWildcards(built.board, wildcardCount);

    const replay = replaySolution({
      board,
      solution: built.solution,
      cfg,
      isWord,
    });

    if (!replay.ok) {
      continue;
    }

    if (
      cfg.rejectVerticalOnly &&
      canClearWithVerticalOnly({
        board,
        byLength,
        cfg,
        maxNodes: verticalOnlyMaxNodes,
      })
    ) {
      continue;
    }

    return {
      level: cfg.level,
      cols: cfg.cols,
      rows: cfg.rows,
      visibleRows: cfg.visibleRows,
      board,
      wildcards: wildcardCount,
      steps,
      horizontalMoves,
      verticalMoves,
      solution: built.solution,
    };
  }

  return null;
}

function puzzleId(prefix, board) {
  return (
    prefix +
    "-" +
    crypto
      .createHash("sha1")
      .update(board.map((row) => row.join(",")).join("/"))
      .digest("hex")
      .slice(0, 10)
  );
}

module.exports = {
  generatePuzzle,
  canClearWithVerticalOnly,
  groupByLength,
  replaySolution,
  puzzleId,
  normalizeWords,
  cloneBoard,
};
