// src/core/rules.js
//
// パズルルール
//

function isSolved(board) {
  for (const row of board) {
    for (const cell of row) {
      if (cell === null) {
        return false;
      }
    }
  }

  return true;
}

//
// 行列重複禁止
//

function isValidMove(
  board,
  row,
  col,
  value
) {
  // row check
  for (let c = 0; c < board[row].length; c++) {
    if (c !== col && board[row][c] === value) {
      return false;
    }
  }

  // col check
  for (let r = 0; r < board.length; r++) {
    if (r !== row && board[r][col] === value) {
      return false;
    }
  }

  return true;
}

module.exports = {
  isSolved,
  isValidMove,
};