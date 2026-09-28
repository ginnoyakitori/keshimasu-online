// src/solver/gravity.js
//
// gravity utilities
//
// 表示仕様:
// - null は「・」として表示
// - 半角 "F" は表示時だけ全角 "Ｆ" に変換
// - 内部データは変更しない
//

function createEmptyBoard(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array(cols).fill(null)
  );
}

function cloneBoard(board) {
  return board.map((row) => [...row]);
}

function boardHeight(board) {
  return board.length;
}

function boardWidth(board) {
  return board[0].length;
}

/**
 * 下方向重力
 *
 * null は上へ
 * 文字は下へ詰める
 */
function applyGravity(board) {
  const rows = boardHeight(board);
  const cols = boardWidth(board);

  const next = createEmptyBoard(rows, cols);

  for (let c = 0; c < cols; c++) {
    let writeRow = rows - 1;

    for (let r = rows - 1; r >= 0; r--) {
      if (board[r][c] !== null) {
        next[writeRow][c] = board[r][c];
        writeRow--;
      }
    }
  }

  return next;
}

/**
 * 指定 path の文字を消して重力をかける
 */
function removePathAndApplyGravity(board, path) {
  const next = cloneBoard(board);

  for (const [r, c] of path) {
    next[r][c] = null;
  }

  return applyGravity(next);
}

function isBoardEmpty(board) {
  for (const row of board) {
    for (const cell of row) {
      if (cell !== null) {
        return false;
      }
    }
  }

  return true;
}

function countFilledCells(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell !== null) {
        count++;
      }
    }
  }

  return count;
}

/**
 * 表示用セル変換
 *
 * 内部:
 *   null -> 空白
 *   "F"  -> ワイルドカード
 *
 * 表示:
 *   null -> ・
 *   "F"  -> Ｆ
 */
function displayCell(cell) {
  if (cell === null) {
    return "・";
  }

  if (cell === "F") {
    return "Ｆ";
  }

  return cell;
}

function boardToString(board) {
  return board
    .map((row) =>
      row.map((cell) => displayCell(cell)).join(" ")
    )
    .join("\n");
}

module.exports = {
  createEmptyBoard,
  cloneBoard,
  applyGravity,
  removePathAndApplyGravity,
  isBoardEmpty,
  countFilledCells,
  boardToString,
  displayCell,
};