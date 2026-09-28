// src/solver/word-search.js
//
// Word search utilities for 8x5 Keshimasu-style puzzle
//
// ルール:
// - 盤面全体は 8行 x 5列
// - 消去できるのは下5段のみ
// - 単語は最大5文字
// - 許可方向は:
//   1. 左 -> 右
//   2. 上 -> 下
// - F は任意の文字として一致するワイルドカード
// - 同じ単語は2回以上使わない場合、usedWords で除外する

const wordsModule = require("../generator/words");
const puzzleConfig = require("../core/puzzle-config");

const PLAYABLE_START_ROW =
  puzzleConfig.PLAYABLE_START_ROW ?? 3;

const MAX_WORD_LENGTH =
  puzzleConfig.MAX_WORD_LENGTH ?? 5;

const WILDCARD_CHAR =
  puzzleConfig.WILDCARD_CHAR ?? "F";

const isPlayableCellFromConfig =
  puzzleConfig.isPlayableCell;

const DEFAULT_WORDS =
  wordsModule.WORDS ||
  wordsModule.words ||
  (
    typeof wordsModule.getPlayableWords === "function"
      ? wordsModule.getPlayableWords({
          minLen: 2,
          maxLen: MAX_WORD_LENGTH,
        })
      : []
  );

const DIRECTIONS_2 = [
  [0, 1], // H: 左 -> 右
  [1, 0], // V: 上 -> 下
];

function isInside(board, r, c) {
  return (
    r >= 0 &&
    r < board.length &&
    c >= 0 &&
    c < board[0].length
  );
}

function wordLength(word) {
  return [...String(word)].length;
}

function splitWord(word) {
  return [...String(word)];
}

function isPlayableWord(word) {
  const len = wordLength(word);

  return (
    len >= 2 &&
    len <= MAX_WORD_LENGTH
  );
}

function isWildcard(cell) {
  return cell === WILDCARD_CHAR;
}

function normalizeCell(cell) {
  if (cell === undefined || cell === null) {
    return null;
  }

  if (cell === "・" || cell === "") {
    return null;
  }

  if (cell === "Ｆ") {
    return WILDCARD_CHAR;
  }

  return String(cell);
}

function charMatches(cell, targetChar) {
  const normalized = normalizeCell(cell);

  return (
    normalized === targetChar ||
    isWildcard(normalized)
  );
}

function isPlayableCell(r, c, playableStartRow = PLAYABLE_START_ROW) {
  if (typeof isPlayableCellFromConfig === "function") {
    return isPlayableCellFromConfig(r, c);
  }

  return (
    r >= playableStartRow &&
    r < 8 &&
    c >= 0 &&
    c < 5
  );
}

function isPathPlayable(path, playableStartRow = PLAYABLE_START_ROW) {
  for (const [r, c] of path) {
    if (!isPlayableCell(r, c, playableStartRow)) {
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

function getBoardCharCount(
  board,
  playableStartRow = PLAYABLE_START_ROW
) {
  const count = {};

  for (let r = playableStartRow; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      const cell = normalizeCell(board[r][c]);

      if (cell === null) {
        continue;
      }

      count[cell] = (count[cell] || 0) + 1;
    }
  }

  return count;
}

function canFormWordByCounts(
  board,
  word,
  playableStartRow = PLAYABLE_START_ROW
) {
  if (!isPlayableWord(word)) {
    return false;
  }

  const boardCount = getBoardCharCount(
    board,
    playableStartRow
  );

  const wildcardCount =
    boardCount[WILDCARD_CHAR] || 0;

  const need = {};

  for (const ch of splitWord(word)) {
    need[ch] = (need[ch] || 0) + 1;
  }

  let missing = 0;

  for (const ch of Object.keys(need)) {
    const available = boardCount[ch] || 0;

    if (available < need[ch]) {
      missing += need[ch] - available;
    }
  }

  return missing <= wildcardCount;
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

function pathMatchesWord(board, word, path) {
  const chars = splitWord(word);

  if (path.length !== chars.length) {
    return false;
  }

  for (let i = 0; i < chars.length; i++) {
    const [r, c] = path[i];

    if (!isInside(board, r, c)) {
      return false;
    }

    if (!charMatches(board[r][c], chars[i])) {
      return false;
    }
  }

  return true;
}

function findPathsForWordStraight(
  board,
  word,
  maxPathsPerWord = 100,
  playableStartRow = PLAYABLE_START_ROW
) {
  if (!isPlayableWord(word)) {
    return [];
  }

  const chars = splitWord(word);
  const length = chars.length;

  const rows = board.length;
  const cols = board[0].length;

  const results = [];

  for (let r = playableStartRow; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      for (const [dr, dc] of DIRECTIONS_2) {
        const endR = r + dr * (length - 1);
        const endC = c + dc * (length - 1);

        if (!isInside(board, endR, endC)) {
          continue;
        }

        const path = buildStraightPath(
          r,
          c,
          dr,
          dc,
          length
        );

        if (!isPathPlayable(path, playableStartRow)) {
          continue;
        }

        if (!isAllowedStraightPath(path)) {
          continue;
        }

        if (!pathMatchesWord(board, word, path)) {
          continue;
        }

        results.push(path);

        if (results.length >= maxPathsPerWord) {
          return results;
        }
      }
    }
  }

  return results;
}

function findPathsForWordAdjacent(
  board,
  word,
  maxPathsPerWord = 100,
  playableStartRow = PLAYABLE_START_ROW
) {
  return findPathsForWordStraight(
    board,
    word,
    maxPathsPerWord,
    playableStartRow
  );
}

function scoreMove(board, word, path) {
  const len = wordLength(word);
  const lengthScore = len * 20;

  const averageRow =
    path.reduce((sum, [r]) => sum + r, 0) /
    path.length;

  const lowerScore = averageRow * 2;

  const [r0, c0] = path[0];
  const [r1, c1] = path[1] || path[0];

  const horizontalBonus =
    r1 === r0 && c1 === c0 + 1 ? 2 : 0;

  const wildcardPenalty =
    path.filter(([r, c]) => normalizeCell(board[r][c]) === WILDCARD_CHAR)
      .length * 3;

  return (
    lengthScore +
    lowerScore +
    horizontalBonus -
    wildcardPenalty
  );
}

function directionFromPath(path) {
  if (path.length < 2) {
    return "H";
  }

  const [r0, c0] = path[0];
  const [r1, c1] = path[1];

  if (r1 === r0 && c1 === c0 + 1) {
    return "H";
  }

  if (r1 === r0 + 1 && c1 === c0) {
    return "V";
  }

  return "?";
}

function findAllMoves(board, options = {}) {
  const words = options.words || DEFAULT_WORDS;
  const maxPathsPerWord =
    options.maxPathsPerWord ?? 100;
  const playableStartRow =
    options.playableStartRow ?? PLAYABLE_START_ROW;

  const useWordOnce =
    options.useWordOnce ?? false;

  const usedWords =
    options.usedWords instanceof Set
      ? options.usedWords
      : new Set(options.usedWords || []);

  const moves = [];

  for (const word of words) {
    if (!isPlayableWord(word)) {
      continue;
    }

    if (useWordOnce && usedWords.has(word)) {
      continue;
    }

    if (!canFormWordByCounts(board, word, playableStartRow)) {
      continue;
    }

    const paths = findPathsForWordStraight(
      board,
      word,
      maxPathsPerWord,
      playableStartRow
    );

    for (const path of paths) {
      moves.push({
        word,
        path,
        score: scoreMove(board, word, path),
        length: wordLength(word),
        direction: directionFromPath(path),
      });
    }
  }

  return moves;
}

function orderMoves(moves) {
  return [...moves].sort((a, b) => {
    const scoreDiff =
      (b.score ?? 0) - (a.score ?? 0);

    if (scoreDiff !== 0) {
      return scoreDiff;
    }

    const lenA = wordLength(a.word);
    const lenB = wordLength(b.word);

    if (lenB !== lenA) {
      return lenB - lenA;
    }

    const aRow = a.path[0]?.[0] ?? 0;
    const bRow = b.path[0]?.[0] ?? 0;

    return bRow - aRow;
  });
}

function pathKey(path) {
  return path
    .map(([r, c]) => `${r},${c}`)
    .join(";");
}

function moveToString(move) {
  return `${move.word}: ${pathKey(move.path)}`;
}

function readPath(board, path) {
  return path
    .map(([r, c]) => board[r][c])
    .join("");
}

module.exports = {
  DIRECTIONS_2,
  isInside,
  wordLength,
  splitWord,
  isPlayableWord,
  isWildcard,
  normalizeCell,
  charMatches,
  isPathPlayable,
  isAllowedStraightPath,
  getBoardCharCount,
  canFormWordByCounts,
  buildStraightPath,
  findPathsForWordStraight,
  findPathsForWordAdjacent,
  findAllMoves,
  orderMoves,
  scoreMove,
  moveToString,
  readPath,
};
``