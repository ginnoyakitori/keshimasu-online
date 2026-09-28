// src/solver/heuristics.js
//
// 探索ヒューリスティック
//

const {
  getCandidates,
} = require("../core");

//
// MRV
// 候補数最小
//

function selectNextCell(board) {
  let best = null;
  let bestCount = Infinity;

  for (let r = 0; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      if (board[r][c] !== null) {
        continue;
      }

      const candidates =
        getCandidates(board, r, c);

      if (candidates.length < bestCount) {
        bestCount = candidates.length;

        best = {
          row: r,
          col: c,
          candidates,
        };
      }
    }
  }

  return best;
}

//
// NN化する場所
//

function orderCells(cells) {
  return cells.sort(
    (a, b) =>
      a.candidates.length -
      b.candidates.length
  );
}

function orderCandidates(candidates) {
  return candidates;
}

module.exports = {
  selectNextCell,
  orderCells,
  orderCandidates,
};