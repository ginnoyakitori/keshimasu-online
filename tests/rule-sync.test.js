// tests/rule-sync.test.js

const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const test = require("node:test");

const {
  removePathAndApplyGravity,
} = require("../src/solver/gravity");

const ROOT = path.resolve(__dirname, "..");
const CASE_DIR = path.join(ROOT, "data", "testcases", "rule_sync");


function loadCases() {
  const files = fs
    .readdirSync(CASE_DIR)
    .filter((name) => name.endsWith(".json"))
    .sort();

  return files.map((name) => {
    const fullPath = path.join(CASE_DIR, name);
    const text = fs.readFileSync(fullPath, "utf8");
    const data = JSON.parse(text);
    data._path = fullPath;
    return data;
  });
}


function normalizeCell(cell) {
  if (cell === null || cell === undefined) {
    return null;
  }

  if (cell === "・" || cell === "") {
    return null;
  }

  if (cell === "Ｆ") {
    return "F";
  }

  return String(cell);
}


function normalizeBoard(board) {
  return board.map((row) => row.map(normalizeCell));
}


function displayCell(cell) {
  if (cell === null || cell === undefined) {
    return "・";
  }

  if (cell === "F") {
    return "Ｆ";
  }

  return String(cell);
}


function boardToDisplayRows(board) {
  return board.map((row) => row.map(displayCell));
}


function boardToText(board) {
  return boardToDisplayRows(board)
    .map((row) => row.join(" "))
    .join("\n");
}


function callRemovePathAndApplyGravity(board, pathCells) {
  const result = removePathAndApplyGravity(board, pathCells);

  if (Array.isArray(result)) {
    return result;
  }

  if (result && Array.isArray(result.board)) {
    return result.board;
  }

  throw new Error(
    "removePathAndApplyGravity returned unsupported value: " +
      JSON.stringify(result)
  );
}


function assertBoardsEqual(actual, expected, caseName) {
  const actualDisplay = boardToDisplayRows(actual);
  const expectedDisplay = boardToDisplayRows(expected);

  try {
    assert.deepEqual(actualDisplay, expectedDisplay);
  } catch (error) {
    error.message =
      "\ncase=" + caseName +
      "\nactual:\n" + boardToText(actual) +
      "\n\nexpected:\n" + boardToText(expected) +
      "\n\n" + error.message;

    throw error;
  }
}


function runGravityCase(caseData) {
  const board = normalizeBoard(caseData.board);
  const expected = normalizeBoard(caseData.expectedBoardAfterGravity);
  const pathCells = caseData.move.path;

  const actual = callRemovePathAndApplyGravity(
    board,
    pathCells
  );

  assertBoardsEqual(
    actual,
    expected,
    caseData.name
  );
}


function runNoRepeatCase(caseData) {
  const usedWords = new Set(caseData.usedWordsBefore || []);
  const word = caseData.move.word;
  const expected = Boolean(caseData.expectedNoRepeatLegal);

  const actual = !usedWords.has(word);

  assert.equal(
    actual,
    expected,
    `case=${caseData.name} no-repeat mismatch`
  );
}


test("JS rule sync cases", () => {
  const cases = loadCases();

  assert.ok(
    cases.length > 0,
    `No rule sync cases found in ${CASE_DIR}`
  );

  for (const caseData of cases) {
    const kind = caseData.kind || "gravity";

    if (kind === "gravity") {
      runGravityCase(caseData);
    } else if (kind === "no-repeat") {
      runNoRepeatCase(caseData);
    } else {
      throw new Error(
        `Unknown rule sync case kind: ${kind}`
      );
    }
  }
});