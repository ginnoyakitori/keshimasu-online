// src/neural/neural-heuristics.js
//
// NN-guided heuristics
//

const {
  boardToTensor,
} = require("../core");

const {
  predictNextCell,
} = require("./inference");

async function selectCellWithNN(
  board,
  cells
) {
  const tensor =
    boardToTensor(board);

  const prediction =
    await predictNextCell(
      tensor
    );

  const [targetRow, targetCol] =
    prediction.cell;

  //
  // NN が選んだセルを優允E
  //

  const exact = cells.find(
    (c) =>
      c.row === targetRow &&
      c.col === targetCol
  );

  if (exact) {
    return exact;
  }

  //
  // fallback
  //

  return cells.sort(
    (a, b) =>
      a.candidates.length -
      b.candidates.length
  )[0];
}

module.exports = {
  selectCellWithNN,
};
