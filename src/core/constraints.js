// src/core/constraints.js
//
// 制約伝播
//

const {
  isValidMove,
} = require("./rules");

const LETTERS = [
  "あ",
  "い",
  "う",
  "え",
  "お",
];

function getCandidates(
  board,
  row,
  col
) {
  return LETTERS.filter((v) =>
    isValidMove(board, row, col, v)
  );
}

function hasFutureValidity(board) {
  for (let r = 0; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      if (board[r][c] === null) {
        const cand = getCandidates(
          board,
          r,
          c
        );

        if (cand.length === 0) {
          return false;
        }
      }
    }
  }

  return true;
}

//
// Constraint propagation
//

function propagateConstraints(board) {
  let changed = true;

  while (changed) {
    changed = false;

    for (let r = 0; r < board.length; r++) {
      for (let c = 0; c < board[r].length; c++) {
        if (board[r][c] !== null) {
          continue;
        }

        const candidates =
          getCandidates(board, r, c);

        if (candidates.length === 1) {
          board[r][c] = candidates[0];
          changed = true;
        }
      }
    }
  }

  return board;
}

module.exports = {
  LETTERS,
  getCandidates,
  hasFutureValidity,
  propagateConstraints,
};