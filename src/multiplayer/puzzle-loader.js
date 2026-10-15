"use strict";

const fs = require("node:fs");
const path = require("node:path");

const {
  LEVELS,
  MODES,
  MIN_WORD_LENGTH,
  MAX_WORD_LENGTH,
  levelConfig,
} = require("./levels");

const EMPTY = "";


function cloneBoard(board) {
  return board.map((row) => row.slice());
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

function readWordList(filePath) {
  const data = readJson(filePath);

  if (Array.isArray(data)) {
    return data.map(String);
  }

  for (const key of ["words", "countries", "capitals", "data"]) {
    if (Array.isArray(data[key])) {
      return data[key].map(String);
    }
  }

  throw new Error(`Unsupported word file format: ${filePath}`);
}

/*
 * プール内の1問を検査します。
 * 形が正しくないものは使いません。
 */
function normalizePuzzle(source, level) {
  const cfg = levelConfig(level);

  if (!source || typeof source !== "object" || !Array.isArray(source.board)) {
    return null;
  }

  const { board } = source;

  if (
    board.length !== cfg.rows ||
    !board.every((row) => Array.isArray(row) && row.length === cfg.cols)
  ) {
    return null;
  }

  const normalized = board.map((row) => row.map((cell) => String(cell ?? "")));

  if (normalized.some((row) => row.some((cell) => cell === EMPTY))) {
    return null;
  }

  return {
    id: String(source.id),
    level: cfg.level,
    cols: cfg.cols,
    rows: cfg.rows,
    visibleRows: cfg.visibleRows,
    board: normalized,
    targetWildcards: normalized.flat().filter((cell) => cell === "F").length,
    steps: Number(source.steps) || 0,
    difficulty: Number(source.difficulty) || 0,
  };
}


class PuzzleLoader {
  constructor({ wordsDir, poolsDir }) {
    if (typeof wordsDir !== "string" || wordsDir.trim() === "") {
      throw new TypeError("PuzzleLoader requires wordsDir");
    }

    if (typeof poolsDir !== "string" || poolsDir.trim() === "") {
      throw new TypeError("PuzzleLoader requires poolsDir");
    }

    this.wordsDir = path.resolve(wordsDir);
    this.poolsDir = path.resolve(poolsDir);

    // mode → Set(単語)
    this.words = new Map();

    // mode → level → 問題の配列
    this.pools = new Map();
  }

  load() {
    for (const [mode, info] of Object.entries(MODES)) {
      try {
        this.loadMode(mode, info);
      } catch (error) {
        console.warn(`[PuzzleLoader] mode "${mode}" is disabled: ${error.message}`);
        this.words.delete(mode);
        this.pools.delete(mode);
      }
    }

    if (this.pools.size === 0) {
      throw new Error(
        "No playable mode was loaded. Run `npm run generate` first. " +
          `words=${this.wordsDir} pools=${this.poolsDir}`
      );
    }

    for (const mode of this.pools.keys()) {
      console.log(
        `[PuzzleLoader] ${mode}: words=${this.words.get(mode).size} ` +
          `puzzles=${this.countPuzzles(mode)}`
      );
    }
  }

  loadMode(mode, info) {
    const wordFile = path.join(this.wordsDir, info.wordFile);

    if (!fs.existsSync(wordFile)) {
      throw new Error(`word file not found: ${wordFile}`);
    }

    // 画面から選べるのは2〜5マスなので、その長さの単語だけを使う
    const words = new Set(
      readWordList(wordFile)
        .map((word) => word.trim())
        .filter((word) => {
          const length = [...word].length;

          return length >= MIN_WORD_LENGTH && length <= MAX_WORD_LENGTH;
        })
    );

    const poolFile = path.join(this.poolsDir, `${mode}.json`);

    if (!fs.existsSync(poolFile)) {
      throw new Error(`pool file not found: ${poolFile}`);
    }

    const rawLevels = readJson(poolFile).levels || {};
    const levels = new Map();
    let invalid = 0;

    for (const cfg of LEVELS) {
      const loaded = [];
      const seen = new Set();

      for (const source of rawLevels[String(cfg.level)] || []) {
        const puzzle = normalizePuzzle(source, cfg.level);

        if (!puzzle || seen.has(puzzle.id)) {
          invalid += 1;
          continue;
        }

        seen.add(puzzle.id);
        loaded.push(puzzle);
      }

      if (loaded.length === 0) {
        throw new Error(`no puzzles for level ${cfg.level}`);
      }

      levels.set(cfg.level, loaded);
    }

    if (invalid > 0) {
      console.warn(`[PuzzleLoader] ${mode}: ${invalid} invalid puzzles skipped`);
    }

    this.words.set(mode, words);
    this.pools.set(mode, levels);
  }

  countPuzzles(mode) {
    let total = 0;

    for (const puzzles of this.pools.get(mode)?.values() || []) {
      total += puzzles.length;
    }

    return total;
  }

  get count() {
    let total = 0;

    for (const mode of this.pools.keys()) {
      total += this.countPuzzles(mode);
    }

    return total;
  }

  get countryCount() {
    return this.words.get("country")?.size ?? 0;
  }

  hasMode(mode) {
    return this.pools.has(mode);
  }

  availableModes() {
    return [...this.pools.keys()].map((id) => ({
      id,
      label: MODES[id].label,
    }));
  }

  // 画面で「消せます」の表示に使う、選べる長さの単語一覧
  wordList(mode) {
    return [...(this.words.get(mode) || [])];
  }

  isWord(mode, word) {
    return this.words.get(mode)?.has(String(word ?? "").trim()) ?? false;
  }

  /*
   * 指定したモードとレベルから1問選びます。
   * excludeIds にある問題は、候補が残っている限り避けます。
   * レベルが最大を超えたときは、最大レベルの問題を使います。
   */
  pick(mode, level, excludeIds = new Set()) {
    const levels = this.pools.get(mode);

    if (!levels) {
      throw new Error(`Unknown mode: ${mode}`);
    }

    const pool = levels.get(levelConfig(level).level);
    const fresh = pool.filter((puzzle) => !excludeIds.has(puzzle.id));
    const candidates = fresh.length > 0 ? fresh : pool;
    const puzzle = candidates[Math.floor(Math.random() * candidates.length)];

    return { ...puzzle, board: cloneBoard(puzzle.board) };
  }
}

module.exports = {
  PuzzleLoader,
  cloneBoard,
};