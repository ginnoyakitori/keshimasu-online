// src/solver/word-search.js
//
// Word search utilities for 8x5 Keshimasu-style puzzle
//
// 仕様:
// - 盤面全体は 8行 x 5列
// - 消去できるのは下5段のみ
// - 単語は最大5文字
// - 許可方向は:
//   1. 左 -> 右
//   2. 上 -> 下
// - F は任意の文字として一致するワイルドカード
//

const { WORDS } = require("../generator/words");

const {
  PLAYABLE_START_ROW,
  MAX_WORD_LENGTH,
  WILDCARD_CHAR,
  isPlayableCell,
} = require("../core/puzzle-config");

// 許可方向は2つだけ
const DIRECTIONS_2 = [
  [0, 1], // 左 -> 右
  [1, 0], // 上 -> 下
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
  return [...word].length;
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

function charMatches(cell, targetChar) {
  return cell === targetChar || isWildcard(cell);
}

function isPathPlayable(path) {
  for (const [r, c] of path) {
    if (!isPlayableCell(r, c)) {
      return false;
    }
  }

  return true;
}

/**
 * path が「左→右」または「上→下」の直線かどうか
 */
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
 * 消去可能エリア内だけの文字頻度を数える
 */
function getBoardCharCount(
  board,
  playableStartRow = PLAYABLE_START_ROW
) {
  const count = {};

  for (let r = playableStartRow; r < board.length; r++) {
    for (let c = 0; c < board[r].length; c++) {
      const cell = board[r][c];

      if (cell === null) {
        continue;
      }

      count[cell] = (count[cell] || 0) + 1;
    }
  }

  return count;
}

/**
 * F込みで、単語が作れる可能性があるかを軽く判定
 */
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

  for (const ch of [...word]) {
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

/**
 * 左→右 / 上→下 だけで単語pathを探す
 */
function findPathsForWordStraight(
  board,
  word,
  maxPathsPerWord = 100,
  playableStartRow = PLAYABLE_START_ROW
) {
  if (!isPlayableWord(word)) {
    return [];
  }

  const chars = [...word];
  const rows = board.length;
  const cols = board[0].length;

  const results = [];

  function localIsPlayableCell(r, c) {
    return (
      r >= playableStartRow &&
      r < rows &&
      c >= 0 &&
      c < cols
    );
  }

  for (let r = playableStartRow; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!charMatches(board[r][c], chars[0])) {
        continue;
      }

      for (const [dr, dc] of DIRECTIONS_2) {
        if (results.length >= maxPathsPerWord) {
          return results;
        }

        const path = buildStraightPath(
          r,
          c,
          dr,
          dc,
          chars.length
        );

        let ok = true;

        for (let i = 0; i < path.length; i++) {
          const [pr, pc] = path[i];

          if (!isInside(board, pr, pc)) {
            ok = false;
            break;
          }

          if (!localIsPlayableCell(pr, pc)) {
            ok = false;
            break;
          }

          if (!charMatches(board[pr][pc], chars[i])) {
            ok = false;
            break;
          }
        }

        if (!ok) {
          continue;
        }

        if (!isPathPlayable(path)) {
          continue;
        }

        if (!isAllowedStraightPath(path)) {
          continue;
        }

        results.push(path);
      }
    }
  }

  return results;
}

/**
 * 後方互換用
 */
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
    path.reduce((sum, [r]) => sum + r, 0) / path.length;

  const lowerScore = averageRow * 2;

  const [r0, c0] = path[0];
  const [r1, c1] = path[1] || path[0];

  const horizontalBonus =
    r1 === r0 && c1 === c0 + 1 ? 2 : 0;

  // F を使う move は少し減点
  const wildcardPenalty =
    path.filter(([r, c]) => board[r][c] === WILDCARD_CHAR).length * 3;

  return (
    lengthScore +
    lowerScore +
    horizontalBonus -
    wildcardPenalty
  );
}

function findAllMoves(board, options = {}) {
  const words = options.words || WORDS;

  const maxPathsPerWord =
    options.maxPathsPerWord ?? 100;

  const playableStartRow =
    options.playableStartRow ?? PLAYABLE_START_ROW;

  const moves = [];

  for (const word of words) {
    const len = wordLength(word);

    if (len < 2) {
      continue;
    }

    if (len > MAX_WORD_LENGTH) {
      continue;
    }

    if (
      !canFormWordByCounts(
        board,
        word,
        playableStartRow
      )
    ) {
      continue;
    }

    const paths = findPathsForWordStraight(
      board,
      word,
      maxPathsPerWord,
      playableStartRow
    );

    for (const path of paths) {
      if (!isPathPlayable(path)) {
        continue;
      }

      if (!isAllowedStraightPath(path)) {
        continue;
      }

      moves.push({
        word,
        path,
        score: scoreMove(board, word, path),
      });
    }
  }

  return moves;
}

function orderMoves(moves) {
  return [...moves].sort((a, b) => {
    const lenA = wordLength(a.word);
    const lenB = wordLength(b.word);

    if (lenA !== lenB) {
      return lenB - lenA;
    }

    if (b.score !== a.score) {
      return b.score - a.score;
    }

    return pathKey(a.path).localeCompare(
      pathKey(b.path)
    );
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
  isPlayableWord,
  isWildcard,
  charMatches,
  isPathPlayable,
  isAllowedStraightPath,
  getBoardCharCount,
  canFormWordByCounts,
  findPathsForWordStraight,
  findPathsForWordAdjacent,
  findAllMoves,
  orderMoves,
  scoreMove,
  moveToString,
  readPath,
};