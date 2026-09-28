// src/generator/reverse-generator.js
//
// 8x5 full-board reverse generator
//
// 仕様:
// - 盤面は 8行 x 5列
// - 全40マスを文字で埋める
// - 消去可能エリアは下5段のみ
// - 単語長は最大5文字
// - pathは 左→右 / 上→下 のみ
// - F はワイルドカード
//
// 今回の重要変更:
// - targetWildcards 個の F をランダム配置しない
// - まず F なし盤面を作る
// - その後「候補moveが増えやすい場所」を評価して F を置く
//

const {
  ROWS,
  COLS,
  PLAYABLE_START_ROW,
  MAX_WORD_LENGTH,
  WILDCARD_CHAR,
  isPlayableCell,
} = require("../core/puzzle-config");

const { WORDS } = require("./words");

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function choice(array) {
  return array[randomInt(array.length)];
}

function shuffle(array) {
  const a = [...array];

  for (let i = a.length - 1; i > 0; i--) {
    const j = randomInt(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }

  return a;
}

function createEmptyBoard(rows = ROWS, cols = COLS) {
  return Array.from({ length: rows }, () =>
    Array(cols).fill(null)
  );
}

function cloneBoard(board) {
  return board.map((row) => [...row]);
}

function displayCell(cell) {
  if (cell === null) {
    return "・";
  }

  if (cell === WILDCARD_CHAR) {
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

function isFullBoard(board) {
  return (
    countFilledCells(board) ===
    board.length * board[0].length
  );
}

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

function applyGravity(board) {
  const rows = board.length;
  const cols = board[0].length;

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

function removePathAndApplyGravity(board, path) {
  const next = cloneBoard(board);

  for (const [r, c] of path) {
    next[r][c] = null;
  }

  return applyGravity(next);
}

function getColumnValues(board, col) {
  const values = [];

  for (let r = 0; r < board.length; r++) {
    if (board[r][col] !== null) {
      values.push(board[r][col]);
    }
  }

  return values;
}

function getColumnHeights(board) {
  const heights = [];

  for (let c = 0; c < board[0].length; c++) {
    heights[c] = getColumnValues(board, c).length;
  }

  return heights;
}

function canInsertIntoColumns(board, path) {
  const rows = board.length;
  const insertCountByCol = {};

  for (const [, c] of path) {
    insertCountByCol[c] =
      (insertCountByCol[c] || 0) + 1;
  }

  for (const colText of Object.keys(insertCountByCol)) {
    const c = Number(colText);

    const existing =
      getColumnValues(board, c).length;

    const inserts =
      insertCountByCol[c];

    if (existing + inserts > rows) {
      return false;
    }
  }

  return true;
}

function isPathPlayable(path) {
  for (const [r, c] of path) {
    if (!isPlayableCell(r, c)) {
      return false;
    }
  }

  return true;
}

function isAllowedStraightPath(path) {
  if (path.length <= 1) {
    return true;
  }

  const [r0, c0] = path[0];
  const [r1, c1] = path[1];

  const dr = r1 - r0;
  const dc = c1 - c0;

  const allowed =
    (dr === 0 && dc === 1) ||
    (dr === 1 && dc === 0);

  if (!allowed) {
    return false;
  }

  for (let i = 1; i < path.length; i++) {
    const [pr, pc] = path[i - 1];
    const [cr, cc] = path[i];

    if (cr - pr !== dr || cc - pc !== dc) {
      return false;
    }
  }

  return true;
}

/**
 * 左→右 / 上→下 のpathだけ作る
 */
function generatePlayableAdjacentPath(
  board,
  length,
  attempts = 3000
) {
  if (length < 1 || length > MAX_WORD_LENGTH) {
    return null;
  }

  const dirs = [
    [0, 1], // 左 -> 右
    [1, 0], // 上 -> 下
  ];

  const heights = getColumnHeights(board);

  function buildPath(sr, sc, dr, dc) {
    const path = [];

    for (let i = 0; i < length; i++) {
      path.push([
        sr + dr * i,
        sc + dc * i,
      ]);
    }

    return path;
  }

  for (let attempt = 0; attempt < attempts; attempt++) {
    let r =
      PLAYABLE_START_ROW +
      randomInt(ROWS - PLAYABLE_START_ROW);

    const candidateCols = [...Array(COLS).keys()]
      .sort((a, b) => heights[a] - heights[b]);

    let c =
      Math.random() < 0.75
        ? candidateCols[
            randomInt(
              Math.min(3, candidateCols.length)
            )
          ]
        : randomInt(COLS);

    const shuffledDirs = shuffle(dirs);

    for (const [dr, dc] of shuffledDirs) {
      const path = buildPath(r, c, dr, dc);

      if (!isPathPlayable(path)) {
        continue;
      }

      if (!isAllowedStraightPath(path)) {
        continue;
      }

      if (!canInsertIntoColumns(board, path)) {
        continue;
      }

      return path;
    }
  }

  return null;
}

/**
 * afterBoard に word/path を逆挿入する
 *
 * ここでは F を入れない。
 * F は盤面完成後に「難しくなりやすい場所」へ配置する。
 */
function insertWordAsReverseMove(afterBoard, word, path) {
  const rows = afterBoard.length;
  const cols = afterBoard[0].length;
  const chars = [...word];

  if (chars.length !== path.length) {
    return null;
  }

  if (chars.length > MAX_WORD_LENGTH) {
    return null;
  }

  if (!isPathPlayable(path)) {
    return null;
  }

  if (!isAllowedStraightPath(path)) {
    return null;
  }

  const insertByCol = {};

  for (let i = 0; i < path.length; i++) {
    const [r, c] = path[i];

    if (!insertByCol[c]) {
      insertByCol[c] = [];
    }

    insertByCol[c].push({
      row: r,
      char: chars[i],
    });
  }

  const beforeBoard = createEmptyBoard(rows, cols);

  for (let c = 0; c < cols; c++) {
    const inserts = insertByCol[c] || [];

    const insertRows = new Set(
      inserts.map((x) => x.row)
    );

    if (insertRows.size !== inserts.length) {
      return null;
    }

    const existingValues =
      getColumnValues(afterBoard, c);

    if (
      existingValues.length + inserts.length >
      rows
    ) {
      return null;
    }

    for (const ins of inserts) {
      beforeBoard[ins.row][c] = ins.char;
    }

    let valueIndex = existingValues.length - 1;

    for (let r = rows - 1; r >= 0; r--) {
      if (insertRows.has(r)) {
        continue;
      }

      if (valueIndex >= 0) {
        beforeBoard[r][c] =
          existingValues[valueIndex];
        valueIndex--;
      }
    }
  }

  const check =
    removePathAndApplyGravity(beforeBoard, path);

  if (!boardsEqual(check, afterBoard)) {
    return null;
  }

  return beforeBoard;
}

/**
 * 合計文字数が targetCells になる単語列を作る
 */
function createWordPlan(
  words,
  targetCells = ROWS * COLS,
  maxAttempts = 500
) {
  const usable = words.filter((word) => {
    const len = [...word].length;

    return (
      len >= 2 &&
      len <= MAX_WORD_LENGTH
    );
  });

  if (usable.length === 0) {
    return null;
  }

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const plan = [];
    let sum = 0;

    while (sum < targetCells) {
      const remaining = targetCells - sum;

      const candidates = usable.filter((word) => {
        const len = [...word].length;
        return len <= remaining;
      });

      if (candidates.length === 0) {
        break;
      }

      const exact = candidates.filter(
        (word) => [...word].length === remaining
      );

      let word;

      if (exact.length > 0) {
        word = choice(exact);
      } else {
        const longCandidates = candidates.filter(
          (w) => [...w].length >= 4
        );

        word =
          longCandidates.length > 0 &&
          Math.random() < 0.65
            ? choice(longCandidates)
            : choice(candidates);
      }

      plan.push(word);
      sum += [...word].length;
    }

    if (sum === targetCells) {
      return shuffle(plan);
    }
  }

  return null;
}

/**
 * 指定された単語列を逆再生で盤面に挿入する
 */
function buildBoardFromWordPlan(
  wordPlan,
  options = {}
) {
  const maxPlacementAttempts =
    options.maxPlacementAttempts ?? 5000;

  let board = createEmptyBoard(ROWS, COLS);
  const reverseMoves = [];

  for (const word of wordPlan) {
    const len = [...word].length;

    if (len > MAX_WORD_LENGTH) {
      return null;
    }

    let placed = false;

    for (
      let attempt = 0;
      attempt < maxPlacementAttempts;
      attempt++
    ) {
      const path = generatePlayableAdjacentPath(
        board,
        len,
        300
      );

      if (!path) {
        continue;
      }

      const beforeBoard =
        insertWordAsReverseMove(
          board,
          word,
          path
        );

      if (!beforeBoard) {
        continue;
      }

      board = beforeBoard;

      reverseMoves.push({
        word,
        path,
      });

      placed = true;
      break;
    }

    if (!placed) {
      return null;
    }
  }

  if (!isFullBoard(board)) {
    return null;
  }

  return {
    board,
    reverseMoves,
    solutionMoves: [...reverseMoves].reverse(),
  };
}

/**
 * ここから F 配置評価
 */

function getCandidateWildcardCells(board) {
  const cells = [];

  for (let r = 0; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      if (board[r][c] !== null && board[r][c] !== WILDCARD_CHAR) {
        cells.push([r, c]);
      }
    }
  }

  return cells;
}

function charMatches(cell, targetChar) {
  return cell === targetChar || cell === WILDCARD_CHAR;
}

function isPlayableWord(word) {
  const len = [...word].length;

  return (
    len >= 2 &&
    len <= MAX_WORD_LENGTH
  );
}

function buildStraightPath(sr, sc, dr, dc, length) {
  const path = [];

  for (let i = 0; i < length; i++) {
    path.push([
      sr + dr * i,
      sc + dc * i,
    ]);
  }

  return path;
}

/**
 * 簡易 move 数カウント
 *
 * solver を直接呼ぶと循環参照になりやすいので、
 * generator 内で軽量に候補数だけ数える。
 */
function countPotentialMoves(board, words) {
  const dirs = [
    [0, 1],
    [1, 0],
  ];

  let count = 0;

  for (const word of words) {
    if (!isPlayableWord(word)) {
      continue;
    }

    const chars = [...word];

    for (let r = PLAYABLE_START_ROW; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        for (const [dr, dc] of dirs) {
          const path = buildStraightPath(
            r,
            c,
            dr,
            dc,
            chars.length
          );

          if (!isPathPlayable(path)) {
            continue;
          }

          let ok = true;

          for (let i = 0; i < path.length; i++) {
            const [pr, pc] = path[i];

            if (!charMatches(board[pr][pc], chars[i])) {
              ok = false;
              break;
            }
          }

          if (ok) {
            count++;
          }
        }
      }
    }
  }

  return count;
}

/**
 * そのマスが何本の合法ラインに含まれやすいか
 */
function countLineCoverage(row, col) {
  let count = 0;

  for (let len = 2; len <= MAX_WORD_LENGTH; len++) {
    // 横ライン
    for (let startCol = 0; startCol <= COLS - len; startCol++) {
      if (
        row >= PLAYABLE_START_ROW &&
        row < ROWS &&
        col >= startCol &&
        col < startCol + len
      ) {
        count++;
      }
    }

    // 縦ライン
    for (
      let startRow = PLAYABLE_START_ROW;
      startRow <= ROWS - len;
      startRow++
    ) {
      if (
        col >= 0 &&
        col < COLS &&
        row >= startRow &&
        row < startRow + len
      ) {
        count++;
      }
    }
  }

  return count;
}

/**
 * F を置いたときの「難しくなりやすさ」を評価する
 */
function scoreWildcardCell(board, row, col, words, baseMoveCount) {
  const original = board[row][col];

  if (original === null || original === WILDCARD_CHAR) {
    return -Infinity;
  }

  const temp = cloneBoard(board);
  temp[row][col] = WILDCARD_CHAR;

  const newMoveCount =
    countPotentialMoves(temp, words);

  const moveIncrease =
    newMoveCount - baseMoveCount;

  const playableBonus =
    row >= PLAYABLE_START_ROW ? 30 : 8;

  const lowerBonus =
    row >= PLAYABLE_START_ROW
      ? (row - PLAYABLE_START_ROW + 1) * 4
      : row;

  const centerCol =
    (COLS - 1) / 2;

  const centerBonus =
    10 - Math.abs(col - centerCol) * 3;

  const coverageBonus =
    countLineCoverage(row, col) * 2;

  return (
    moveIncrease * 50 +
    playableBonus +
    lowerBonus +
    centerBonus +
    coverageBonus +
    Math.random() * 0.001
  );
}

/**
 * targetWildcards 個の F を難しくなりやすい場所に置く
 *
 * greedy:
 * 1個置く
 * ↓
 * 盤面更新
 * ↓
 * 次に一番効く場所を再評価
 */
function placeWildcardsForDifficulty(
  inputBoard,
  targetWildcards = 0,
  options = {}
) {
  const words = options.words || WORDS;

  const board = cloneBoard(inputBoard);

  const target = Math.max(
    0,
    Math.min(
      targetWildcards,
      ROWS * COLS
    )
  );

  for (let placed = 0; placed < target; placed++) {
    const cells =
      getCandidateWildcardCells(board);

    if (cells.length === 0) {
      break;
    }

    const baseMoveCount =
      countPotentialMoves(board, words);

    let bestCell = null;
    let bestScore = -Infinity;

    for (const [r, c] of cells) {
      const score =
        scoreWildcardCell(
          board,
          r,
          c,
          words,
          baseMoveCount
        );

      if (score > bestScore) {
        bestScore = score;
        bestCell = [r, c];
      }
    }

    if (!bestCell) {
      break;
    }

    const [br, bc] = bestCell;
    board[br][bc] = WILDCARD_CHAR;
  }

  if (countWildcards(board) !== target) {
    return null;
  }

  return board;
}

/**
 * 8x5 全マス文字入りの解ける盤面を生成する
 */
function generateFullBoardByReverseSimulation(options = {}) {
  const words = options.words || WORDS;

  const targetCells =
    options.targetCells ?? ROWS * COLS;

  const targetWildcards =
    options.targetWildcards ?? 0;

  const maxPlanAttempts =
    options.maxPlanAttempts ?? 500;

  const maxPlacementAttempts =
    options.maxPlacementAttempts ?? 5000;

  for (
    let attempt = 0;
    attempt < maxPlanAttempts;
    attempt++
  ) {
    const plan = createWordPlan(
      words,
      targetCells,
      500
    );

    if (!plan) {
      continue;
    }

    const result = buildBoardFromWordPlan(plan, {
      maxPlacementAttempts,
    });

    if (!result) {
      continue;
    }

    const boardWithWildcards =
      placeWildcardsForDifficulty(
        result.board,
        targetWildcards,
        { words }
      );

    if (!boardWithWildcards) {
      continue;
    }

    return {
      board: boardWithWildcards,
      reverseMoves: result.reverseMoves,
      solutionMoves: result.solutionMoves,
      targetWildcards,
      actualWildcards:
        countWildcards(boardWithWildcards),
    };
  }

  return null;
}

module.exports = {
  createEmptyBoard,
  cloneBoard,
  applyGravity,
  removePathAndApplyGravity,
  insertWordAsReverseMove,
  generateFullBoardByReverseSimulation,
  boardToString,
  displayCell,
  countFilledCells,
  countWildcards,
  isFullBoard,
  createWordPlan,
  buildBoardFromWordPlan,
  generatePlayableAdjacentPath,
  isAllowedStraightPath,
  placeWildcardsForDifficulty,
  countPotentialMoves,
  scoreWildcardCell,
};