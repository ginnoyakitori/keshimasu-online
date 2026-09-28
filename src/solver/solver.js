// src/solver/solver.js
//
// 8x5 word puzzle solver
//
// ルール:
// - 盤面は縦8 × 横5
// - 消去できるのは下5段のみ
// - row 0,1,2: 消去不可
// - row 3,4,5,6,7: 消去可能
// - 単語は H/V の直線のみ
// - F はワイルドカード
// - 同じ単語は useWordOnce=true の場合、2回以上使えない
// - 単語を消すと盤面全体に重力がかかる
// - 全部消えたら solved

const { performance } = require("node:perf_hooks");

const {
  cloneBoard,
  removePathAndApplyGravity,
  isBoardEmpty,
  countFilledCells,
} = require("./gravity");

const {
  findAllMoves,
  orderMoves,
} = require("./word-search");


const DEFAULT_OPTIONS = {
  maxDepth: 120,
  maxNodes: 500000,
  maxPathsPerWord: 80,
  playableStartRow: 3,
  allowLoose: false,
  useVisited: true,

  // 新ルール:
  // true の場合、同じ単語は2回以上使えない
  useWordOnce: true,
};


function createStats() {
  return {
    exploredNodes: 0,
    backtracks: 0,
    deadEnds: 0,
    forcedMoves: 0,
    maxDepth: 0,
    elapsedMs: 0,
    branchingHistogram: {},
    solvedMoves: [],
    choices: [],
  };
}

function recordNode(stats) {
  stats.exploredNodes += 1;
}

function recordDepth(stats, depth) {
  if (depth > stats.maxDepth) {
    stats.maxDepth = depth;
  }
}

function recordDeadEnd(stats) {
  stats.deadEnds += 1;
}

function recordBacktrack(stats) {
  stats.backtracks += 1;
}

function recordForcedMove(stats) {
  stats.forcedMoves += 1;
}

function recordBranching(stats, branch) {
  const key = String(branch);
  stats.branchingHistogram[key] =
    (stats.branchingHistogram[key] || 0) + 1;
}

function finalizeStats(stats, elapsedMs) {
  stats.elapsedMs = elapsedMs;
}

function boardHash(board) {
  return board
    .map((row) =>
      row.map((cell) => cell ?? ".").join("")
    )
    .join("|");
}

function usedWordsKey(usedWords) {
  return [...usedWords].sort().join(",");
}

function stateHash(board, usedWords) {
  return `${boardHash(board)}::${usedWordsKey(usedWords)}`;
}

function boardToTensor(board) {
  // 既存互換用の簡易表現。
  // JS neural 側で使わない場合でも export は残す。
  const charMap = new Map();
  let nextId = 1;

  return board.map((row) =>
    row.map((cell) => {
      if (cell === null || cell === undefined) {
        return 0;
      }

      const ch = String(cell);

      if (!charMap.has(ch)) {
        charMap.set(ch, nextId);
        nextId += 1;
      }

      return charMap.get(ch);
    })
  );
}

function isMoveInsidePlayableArea(move, playableStartRow) {
  for (const [r] of move.path) {
    if (r < playableStartRow) {
      return false;
    }
  }

  return true;
}

function shouldStopSearch(stats, options, depth) {
  if (stats.exploredNodes >= options.maxNodes) {
    return true;
  }

  if (depth > options.maxDepth) {
    return true;
  }

  return false;
}

function moveToSolvedMove(move) {
  return {
    word: move.word,
    path: move.path.map(([r, c]) => [r, c]),
    direction: move.direction,
    score: move.score,
    length: move.length,
  };
}

function hasDuplicateSolvedMoves(moves) {
  const seen = new Set();

  for (const move of moves) {
    if (!move || typeof move.word !== "string") {
      continue;
    }

    if (seen.has(move.word)) {
      return true;
    }

    seen.add(move.word);
  }

  return false;
}

function dfs(
  board,
  stats,
  options,
  depth,
  visited,
  usedWords
) {
  recordNode(stats);
  recordDepth(stats, depth);

  if (shouldStopSearch(stats, options, depth)) {
    recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (isBoardEmpty(board)) {
    return {
      solved: true,
      board,
    };
  }

  const hash = stateHash(board, usedWords);

  if (options.useVisited && visited.has(hash)) {
    recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (options.useVisited) {
    visited.add(hash);
  }

  let moves = findAllMoves(board, {
    words: options.words,
    maxPathsPerWord: options.maxPathsPerWord,
    playableStartRow: options.playableStartRow,
    useWordOnce: options.useWordOnce,
    usedWords,
  });

  moves = moves.filter((move) =>
    isMoveInsidePlayableArea(
      move,
      options.playableStartRow
    )
  );

  if (options.useWordOnce) {
    moves = moves.filter((move) => !usedWords.has(move.word));
  }

  moves = orderMoves(moves);

  const branch = moves.length;
  recordBranching(stats, branch);

  if (branch === 0) {
    recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (branch === 1) {
    recordForcedMove(stats);
  }

  const beforeFilled = countFilledCells(board);

  for (const move of moves) {
    if (options.useWordOnce && usedWords.has(move.word)) {
      continue;
    }

    const nextBoard = removePathAndApplyGravity(
      board,
      move.path
    );

    const afterFilled = countFilledCells(nextBoard);

    if (afterFilled >= beforeFilled) {
      continue;
    }

    const nextUsedWords = new Set(usedWords);

    if (options.useWordOnce) {
      nextUsedWords.add(move.word);
    }

    const childVisited =
      options.useVisited
        ? new Set(visited)
        : visited;

    const result = dfs(
      nextBoard,
      stats,
      options,
      depth + 1,
      childVisited,
      nextUsedWords
    );

    if (result.solved) {
      stats.solvedMoves.unshift(
        moveToSolvedMove(move)
      );

      return result;
    }
  }

  recordBacktrack(stats);

  return {
    solved: false,
    board,
  };
}

function solve(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const board = cloneBoard(inputBoard);
  const stats = createStats();

  const start = performance.now();

  const result = dfs(
    board,
    stats,
    options,
    0,
    new Set(),
    new Set()
  );

  const elapsed = performance.now() - start;

  finalizeStats(stats, elapsed);

  if (options.useWordOnce && hasDuplicateSolvedMoves(stats.solvedMoves)) {
    return {
      solved: false,
      board: result.board,
      stats,
      reason: "duplicate word in solvedMoves",
    };
  }

  return {
    solved: result.solved,
    board: result.board,
    stats,
  };
}

function countSolutions(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const limit = options.limit ?? 2;
  let count = 0;

  function inner(board, depth, visited, usedWords) {
    if (count >= limit) {
      return;
    }

    if (depth > options.maxDepth) {
      return;
    }

    if (isBoardEmpty(board)) {
      count += 1;
      return;
    }

    const hash = stateHash(board, usedWords);

    if (options.useVisited && visited.has(hash)) {
      return;
    }

    if (options.useVisited) {
      visited.add(hash);
    }

    let moves = findAllMoves(board, {
      words: options.words,
      maxPathsPerWord: options.maxPathsPerWord,
      playableStartRow: options.playableStartRow,
      useWordOnce: options.useWordOnce,
      usedWords,
    });

    moves = moves.filter((move) =>
      isMoveInsidePlayableArea(
        move,
        options.playableStartRow
      )
    );

    if (options.useWordOnce) {
      moves = moves.filter((move) => !usedWords.has(move.word));
    }

    moves = orderMoves(moves);

    const beforeFilled = countFilledCells(board);

    for (const move of moves) {
      if (count >= limit) {
        return;
      }

      const nextBoard = removePathAndApplyGravity(
        board,
        move.path
      );

      if (countFilledCells(nextBoard) >= beforeFilled) {
        continue;
      }

      const nextUsedWords = new Set(usedWords);

      if (options.useWordOnce) {
        nextUsedWords.add(move.word);
      }

      const childVisited =
        options.useVisited
          ? new Set(visited)
          : visited;

      inner(
        nextBoard,
        depth + 1,
        childVisited,
        nextUsedWords
      );
    }
  }

  inner(
    cloneBoard(inputBoard),
    0,
    new Set(),
    new Set()
  );

  return count;
}

module.exports = {
  solve,
  countSolutions,
  boardToTensor,
  boardHash,
  stateHash,
};