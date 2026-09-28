// src/core/board.js
//
// board utilities
//
// 仕様:
// - 盤面は基本 8行 x 5列
// - 内部データでは null を空白として扱う
// - 内部データでは "F" をワイルドカードとして扱う
//
// 表示:
// - null -> ・
// - "F"  -> Ｆ
// - その他の文字はそのまま表示
//

const {
  ROWS,
  COLS,
  WILDCARD_CHAR,
} = require("./puzzle-config");

/**
 * 空盤面を作る
 */
function createBoard(
  rows = ROWS,
  cols = COLS,
  fill = null
) {
  return Array.from({ length: rows }, () =>
    Array(cols).fill(fill)
  );
}

/**
 * 盤面を複製する
 */
function cloneBoard(board) {
  return board.map((row) => [...row]);
}

/**
 * 盤面サイズ
 */
function getBoardSize(board) {
  return {
    rows: board.length,
    cols: board[0]?.length ?? 0,
  };
}

/**
 * 指定座標が盤面内か
 */
function isInside(board, row, col) {
  return (
    row >= 0 &&
    row < board.length &&
    col >= 0 &&
    col < board[0].length
  );
}

/**
 * セル取得
 */
function getCell(board, row, col) {
  if (!isInside(board, row, col)) {
    return undefined;
  }

  return board[row][col];
}

/**
 * セル設定
 */
function setCell(board, row, col, value) {
  if (!isInside(board, row, col)) {
    return false;
  }

  board[row][col] = value;
  return true;
}

/**
 * セルを空にする
 */
function clearCell(board, row, col) {
  return setCell(board, row, col, null);
}

/**
 * path 上のセルを空にする
 */
function clearPath(board, path) {
  for (const [row, col] of path) {
    clearCell(board, row, col);
  }

  return board;
}

/**
 * 空セルかどうか
 */
function isEmptyCell(cell) {
  return cell === null;
}

/**
 * ワイルドカードかどうか
 */
function isWildcardCell(cell) {
  return cell === WILDCARD_CHAR;
}

/**
 * 盤面が全部空か
 */
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

/**
 * 盤面が全マス埋まっているか
 */
function isBoardFull(board) {
  for (const row of board) {
    for (const cell of row) {
      if (cell === null) {
        return false;
      }
    }
  }

  return true;
}

/**
 * 埋まっているセル数
 */
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
 * 空セル数
 */
function countEmptyCells(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell === null) {
        count++;
      }
    }
  }

  return count;
}

/**
 * ワイルドカード数
 */
function countWildcards(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell === WILDCARD_CHAR) {
        count++;
      }
    }
  }

  return count;
}

/**
 * 空セル一覧
 */
function getEmptyCells(board) {
  const cells = [];

  for (let r = 0; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      if (board[r][c] === null) {
        cells.push({
          row: r,
          col: c,
        });
      }
    }
  }

  return cells;
}

/**
 * 埋まっているセル一覧
 */
function getFilledCells(board) {
  const cells = [];

  for (let r = 0; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      if (board[r][c] !== null) {
        cells.push({
          row: r,
          col: c,
          value: board[r][c],
        });
      }
    }
  }

  return cells;
}

/**
 * path から文字列を読む
 *
 * F は内部値のまま "F" として返す
 */
function readPath(board, path) {
  return path
    .map(([row, col]) => {
      if (!isInside(board, row, col)) {
        return "";
      }

      return board[row][col] ?? "";
    })
    .join("");
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

  if (cell === WILDCARD_CHAR) {
    return "Ｆ";
  }

  return cell;
}

/**
 * 盤面を文字列化
 */
function boardToString(board) {
  return board
    .map((row) =>
      row.map((cell) => displayCell(cell)).join(" ")
    )
    .join("\n");
}

/**
 * console.log 用
 */
function printBoard(board) {
  console.log(boardToString(board));
}

/**
 * 盤面ハッシュ
 *
 * 探索の visited 管理用
 * 表示用ではなく内部用なので:
 * - null -> "."
 * - F は "F" のまま
 */
function boardHash(board) {
  return board
    .map((row) =>
      row.map((cell) => cell ?? ".").join("")
    )
    .join("|");
}

/**
 * 盤面が同じか
 */
function boardsEqual(a, b) {
  if (!a || !b) {
    return false;
  }

  if (a.length !== b.length) {
    return false;
  }

  if (a[0].length !== b[0].length) {
    return false;
  }

  for (let r = 0; r < a.length; r++) {
    for (let c = 0; c < a[0].length; c++) {
      if (a[r][c] !== b[r][c]) {
        return false;
      }
    }
  }

  return true;
}

/**
 * JSON保存用
 */
function boardToJSON(board) {
  return JSON.stringify(board);
}

/**
 * JSON復元用
 */
function boardFromJSON(json) {
  return JSON.parse(json);
}

module.exports = {
  createBoard,
  cloneBoard,
  getBoardSize,
  isInside,
  getCell,
  setCell,
  clearCell,
  clearPath,

  isEmptyCell,
  isWildcardCell,

  isBoardEmpty,
  isBoardFull,
  countFilledCells,
  countEmptyCells,
  countWildcards,

  getEmptyCells,
  getFilledCells,

  readPath,

  displayCell,
  boardToString,
  printBoard,

  boardHash,
  boardsEqual,

  boardToJSON,
  boardFromJSON,
};