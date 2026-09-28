"use strict";

const fs = require("node:fs");
const path = require("node:path");


const ROWS = 8;
const COLS = 5;
const EMPTY = "・";


function cloneBoard(board) {
  return board.map(
    (row) => row.slice()
  );
}


function normalizeCell(value) {
  const text =
    String(value ?? "").trim();

  if (
    text === "" ||
    text === EMPTY
  ) {
    return EMPTY;
  }

  if (
    text === "F" ||
    text === "Ｆ"
  ) {
    return "F";
  }

  return text;
}


function normalizeBoard(board) {
  if (
    !Array.isArray(board) ||
    board.length !== ROWS
  ) {
    return null;
  }

  const normalized = [];

  for (const row of board) {
    if (
      !Array.isArray(row) ||
      row.length !== COLS
    ) {
      return null;
    }

    const normalizedRow =
      row.map(normalizeCell);

    normalized.push(
      normalizedRow
    );
  }

  return normalized;
}


function countFilledCells(board) {
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


function countWildcards(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell === "F") {
        count += 1;
      }
    }
  }

  return count;
}


function readJson(filePath) {
  const text = fs.readFileSync(
    filePath,
    "utf8"
  );

  return JSON.parse(
    text.replace(/^\uFEFF/, "")
  );
}


function readWordList(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(
      `Country file not found: ${filePath}`
    );
  }

  const data = readJson(filePath);

  if (Array.isArray(data)) {
    return data.map(String);
  }

  for (const key of [
    "words",
    "countries",
    "data",
  ]) {
    if (Array.isArray(data[key])) {
      return data[key].map(String);
    }
  }

  throw new Error(
    `Unsupported country file format: ${filePath}`
  );
}


function normalizePuzzle(
  source,
  index
) {
  if (
    !source ||
    typeof source !== "object"
  ) {
    return null;
  }

  const board =
    normalizeBoard(source.board);

  if (!board) {
    return null;
  }

  if (
    countFilledCells(board) !==
    ROWS * COLS
  ) {
    return null;
  }

  const rawId =
    source.id ?? index + 1;

  return {
    id: `json-${String(rawId)}`,

    source:
      "hard_puzzle_boards.json",

    board,

    targetWildcards:
      countWildcards(board),

    hardnessScore: Number(
      source.hardnessScore ?? 0
    ),

    hardnessLabel: String(
      source.hardnessLabel ??
      "unknown"
    ),

    difficulty: Number(
      source.difficulty ?? 0
    ),
  };
}


class PuzzleLoader {
  constructor({
    puzzleFile,
    countryFile,
  }) {
    if (
      typeof puzzleFile !== "string" ||
      puzzleFile.trim() === ""
    ) {
      throw new TypeError(
        "PuzzleLoader requires puzzleFile"
      );
    }

    if (
      typeof countryFile !== "string" ||
      countryFile.trim() === ""
    ) {
      throw new TypeError(
        "PuzzleLoader requires countryFile"
      );
    }

    this.puzzleFile =
      path.resolve(puzzleFile);

    this.countryFile =
      path.resolve(countryFile);

    this.puzzles = [];
    this.countries = new Set();
  }


  load() {
    this.loadCountries();
    this.loadPuzzles();

    if (this.puzzles.length === 0) {
      throw new Error(
        "No playable puzzles were loaded from: " +
        this.puzzleFile
      );
    }
  }


  loadCountries() {
    const words =
      readWordList(
        this.countryFile
      );

    this.countries =
      new Set(
        words
          .map(
            (word) =>
              String(word).trim()
          )
          .filter(
            (word) => {
              const length =
                [...word].length;

              return (
                length >= 2 &&
                length <= 5
              );
            }
          )
      );
  }


  loadPuzzles() {
    if (
      !fs.existsSync(
        this.puzzleFile
      )
    ) {
      throw new Error(
        `Puzzle JSON not found: ${this.puzzleFile}`
      );
    }

    const extension =
      path.extname(
        this.puzzleFile
      ).toLowerCase();

    if (extension !== ".json") {
      throw new Error(
        "Puzzle file must be JSON: " +
        this.puzzleFile
      );
    }

    const data =
      readJson(
        this.puzzleFile
      );

    if (!Array.isArray(data)) {
      throw new Error(
        "Puzzle JSON root must be an array"
      );
    }

    const loaded = [];
    const usedIds = new Set();

    let invalidCount = 0;
    let duplicateCount = 0;

    for (
      let index = 0;
      index < data.length;
      index += 1
    ) {
      const puzzle =
        normalizePuzzle(
          data[index],
          index
        );

      if (!puzzle) {
        invalidCount += 1;
        continue;
      }

      if (usedIds.has(puzzle.id)) {
        duplicateCount += 1;
        continue;
      }

      usedIds.add(puzzle.id);
      loaded.push(puzzle);
    }

    this.puzzles = loaded;

    console.log(
      "[PuzzleLoader] JSON records:",
      data.length
    );

    console.log(
      "[PuzzleLoader] Loaded puzzles:",
      loaded.length
    );

    console.log(
      "[PuzzleLoader] Invalid:",
      invalidCount
    );

    console.log(
      "[PuzzleLoader] Duplicates:",
      duplicateCount
    );
  }


  get count() {
    return this.puzzles.length;
  }


  get countryCount() {
    return this.countries.size;
  }


  isCountry(word) {
    return this.countries.has(
      String(word ?? "").trim()
    );
  }


  getById(puzzleId) {
    const id =
      String(puzzleId ?? "");

    const puzzle =
      this.puzzles.find(
        (candidate) =>
          candidate.id === id
      );

    if (!puzzle) {
      return null;
    }

    return {
      ...puzzle,
      board:
        cloneBoard(puzzle.board),
    };
  }


  random(excludeId = null) {
    let candidates =
      this.puzzles;

    if (
      excludeId &&
      this.puzzles.length > 1
    ) {
      const filtered =
        this.puzzles.filter(
          (puzzle) =>
            puzzle.id !== excludeId
        );

      if (filtered.length > 0) {
        candidates = filtered;
      }
    }

    const randomIndex =
      Math.floor(
        Math.random() *
        candidates.length
      );

    const puzzle =
      candidates[randomIndex];

    return {
      ...puzzle,

      board:
        cloneBoard(puzzle.board),
    };
  }
}


module.exports = {
  PuzzleLoader,
  cloneBoard,
};