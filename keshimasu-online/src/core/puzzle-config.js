// src/core/puzzle-config.js

const {
  loadPuzzleConfig,
} = require("./config-loader");

const config = loadPuzzleConfig();

const ROWS = config.ROWS;
const COLS = config.COLS;
const PLAYABLE_START_ROW = config.PLAYABLE_START_ROW;
const PLAYABLE_ROWS = ROWS - PLAYABLE_START_ROW;
const MAX_WORD_LENGTH = config.MAX_WORD_LENGTH;

const WILDCARD_CHAR = config.WILDCARD_CHAR;
const EMPTY_DISPLAY = config.EMPTY_DISPLAY;
const WILDCARD_DISPLAY = config.WILDCARD_DISPLAY;

function isPlayableCell(row, col) {
  return (
    row >= PLAYABLE_START_ROW &&
    row < ROWS &&
    col >= 0 &&
    col < COLS
  );
}

module.exports = {
  ROWS,
  COLS,
  PLAYABLE_START_ROW,
  PLAYABLE_ROWS,
  MAX_WORD_LENGTH,
  WILDCARD_CHAR,
  EMPTY_DISPLAY,
  WILDCARD_DISPLAY,
  isPlayableCell,
};