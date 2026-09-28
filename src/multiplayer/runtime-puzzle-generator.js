"use strict";

const crypto = require("node:crypto");

const ROWS = 8;
const COLS = 5;
const VISIBLE_ROWS = 5;
const EMPTY = "・";

const DEFAULT_OPTIONS = Object.freeze({
  minWildcards: 2,
  maxWildcards: 8,
  maxAttempts: 100,
});

const PARTITIONS_OF_EIGHT = Object.freeze([
  [3, 5],
  [5, 3],
  [4, 4],
  [2, 2, 4],
  [2, 4, 2],
  [4, 2, 2],
  [2, 3, 3],
  [3, 2, 3],
  [3, 3, 2],
]);

function cloneBoard(board) {
  return board.map((row) => row.slice());
}

function randomInt(min, max) {
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

function normalizeWords(words) {
  return [...new Set(
    (Array.isArray(words) ? words : [])
      .filter((word) => typeof word === "string")
      .map((word) => word.trim())
      .filter((word) => {
        const length = [...word].length;
        return length >= 2 && length <= 5;
      })
  )];
}

function groupWordsByLength(words) {
  const byLength = new Map();
  for (const word of words) {
    const length = [...word].length;
    if (!byLength.has(length)) byLength.set(length, []);
    byLength.get(length).push(word);
  }
  return byLength;
}

function matchesPattern(pattern, word) {
  const patternChars = [...pattern];
  const wordChars = [...word];
  if (patternChars.length !== wordChars.length) return false;
  for (let index = 0; index < patternChars.length; index += 1) {
    if (patternChars[index] !== "F" && patternChars[index] !== wordChars[index]) {
      return false;
    }
  }
  return true;
}

function applyGravity(board) {
  for (let col = 0; col < COLS; col += 1) {
    const characters = [];
    for (let row = ROWS - 1; row >= 0; row -= 1) {
      if (board[row][col] !== EMPTY) characters.push(board[row][col]);
    }
    for (let row = ROWS - 1, index = 0; row >= 0; row -= 1, index += 1) {
      board[row][col] = index < characters.length ? characters[index] : EMPTY;
    }
  }
}

function countRemaining(board) {
  let count = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell !== EMPTY) count += 1;
    }
  }
  return count;
}

class RuntimePuzzleGenerator {
  constructor({ getWords, minWildcards, maxWildcards, maxAttempts } = {}) {
    if (typeof getWords !== "function") {
      throw new TypeError("RuntimePuzzleGenerator requires getWords()");
    }
    this.getWords = getWords;
    this.options = {
      ...DEFAULT_OPTIONS,
      ...(Number.isInteger(minWildcards) ? { minWildcards } : {}),
      ...(Number.isInteger(maxWildcards) ? { maxWildcards } : {}),
      ...(Number.isInteger(maxAttempts) ? { maxAttempts } : {}),
    };
  }

  generateVerifiedPuzzle() {
    const words = normalizeWords(this.getWords());
    const byLength = groupWordsByLength(words);
    const usablePartitions = PARTITIONS_OF_EIGHT.filter((partition) =>
      partition.every((length) => (byLength.get(length) || []).length > 0)
    );

    if (usablePartitions.length === 0) {
      throw new Error("8文字を構成できる長さの国名が不足しています");
    }

    for (let attempt = 1; attempt <= this.options.maxAttempts; attempt += 1) {
      const generated = this.generateOne(byLength, usablePartitions, attempt);
      if (!generated) continue;

      const verification = this.verifyExpectedPlan(
        generated.board,
        generated.expectedPlan
      );

      if (!verification.ok) continue;

      return {
        id: `runtime-${Date.now()}-${crypto.randomBytes(4).toString("hex")}`,
        source: "runtime-reverse-construction",
        board: cloneBoard(generated.board),
        targetWildcards: generated.targetWildcards,
        expectedPlan: generated.expectedPlan,
        verifiedSolution: generated.expectedPlan,
        generatedAt: new Date().toISOString(),
        generationAttempt: attempt,
        solved: true,
      };
    }

    throw new Error("全消去可能な盤面を生成できませんでした");
  }

  generateOne(byLength, usablePartitions, attempt) {
    const usedWords = new Set();
    const columns = [];
    const columnChunks = [];

    for (let col = 0; col < COLS; col += 1) {
      const partition = usablePartitions[randomInt(0, usablePartitions.length - 1)];
      const bottomToTopChunks = [];

      for (const length of partition) {
        const candidates = shuffle(byLength.get(length) || []).filter(
          (word) => !usedWords.has(word)
        );
        const word = candidates[0];
        if (!word) return null;
        usedWords.add(word);
        bottomToTopChunks.push({ word, length });
      }

      const topToBottomWords = bottomToTopChunks.slice().reverse();
      const column = topToBottomWords.flatMap((chunk) => [...chunk.word]);
      if (column.length !== ROWS) return null;

      columns.push(column);
      columnChunks.push(bottomToTopChunks);
    }

    const board = Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));
    for (let col = 0; col < COLS; col += 1) {
      for (let row = 0; row < ROWS; row += 1) {
        board[row][col] = columns[col][row];
      }
    }

    const minF = Math.max(0, this.options.minWildcards);
    const maxF = Math.min(ROWS * COLS, Math.max(minF, this.options.maxWildcards));
    const targetWildcards = randomInt(minF, maxF);
    const positions = shuffle(
      Array.from({ length: ROWS * COLS }, (_, index) => index)
    ).slice(0, targetWildcards);

    for (const position of positions) {
      const row = Math.floor(position / COLS);
      const col = position % COLS;
      board[row][col] = "F";
    }

    const expectedPlan = this.buildExpectedPlan(board, columnChunks);
    if (!expectedPlan) return null;

    return { board, expectedPlan, targetWildcards, attempt };
  }

  buildExpectedPlan(initialBoard, columnChunks) {
    const board = cloneBoard(initialBoard);
    const nextChunkIndex = Array(COLS).fill(0);
    const plan = [];
    const availableColumns = () =>
      Array.from({ length: COLS }, (_, col) => col).filter(
        (col) => nextChunkIndex[col] < columnChunks[col].length
      );

    while (availableColumns().length > 0) {
      const choices = availableColumns();
      const col = choices[randomInt(0, choices.length - 1)];
      const chunk = columnChunks[col][nextChunkIndex[col]];
      const startRow = ROWS - chunk.length;

      if (startRow < ROWS - VISIBLE_ROWS) return null;

      const cells = [];
      for (let row = startRow; row < ROWS; row += 1) cells.push([row, col]);
      const pattern = cells.map(([row, cellCol]) => board[row][cellCol]).join("");
      if (!matchesPattern(pattern, chunk.word)) return null;

      plan.push({
        word: chunk.word,
        pattern,
        direction: "V",
        cells: cells.map(([row, cellCol]) => [row, cellCol]),
        start: cells[0].slice(),
        end: cells[cells.length - 1].slice(),
      });

      for (const [row, cellCol] of cells) board[row][cellCol] = EMPTY;
      applyGravity(board);
      nextChunkIndex[col] += 1;
    }

    return countRemaining(board) === 0 ? plan : null;
  }

  verifyExpectedPlan(initialBoard, expectedPlan) {
    const board = cloneBoard(initialBoard);
    const usedWords = new Set();

    for (const step of expectedPlan) {
      if (usedWords.has(step.word)) {
        return { ok: false, reason: "duplicate word" };
      }
      if (!Array.isArray(step.cells) || step.cells.length < 2 || step.cells.length > 5) {
        return { ok: false, reason: "invalid cell length" };
      }
      if (step.cells.some(([row]) => row < ROWS - VISIBLE_ROWS)) {
        return { ok: false, reason: "hidden row selected" };
      }

      const pattern = step.cells.map(([row, col]) => board[row]?.[col]).join("");
      if (!matchesPattern(pattern, step.word)) {
        return { ok: false, reason: "pattern mismatch" };
      }

      for (const [row, col] of step.cells) board[row][col] = EMPTY;
      applyGravity(board);
      usedWords.add(step.word);
    }

    const remaining = countRemaining(board);
    return {
      ok: remaining === 0,
      remaining,
      steps: expectedPlan.length,
    };
  }
}

module.exports = {
  RuntimePuzzleGenerator,
  cloneBoard,
};
