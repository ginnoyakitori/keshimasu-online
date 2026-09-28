// src/solver/solver-policy.js
//
// ONNX Policy-guided word puzzle solver
//
// 目的:
// - Python で学習・ONNX export した PolicyMLP を Node.js 側で使う
// - move ordering を改善する
// - 既存 solver.js は壊さない
//
// 仕様:
// - 盤面: 8 x 5
// - 消去可能エリア: 下5段
// - 単語方向: 左→右 / 上→下
// - F はワイルドカード
// - 同じ単語は useWordOnce=true の場合、2回以上使わない
//
// 依存:
// - src/neural/onnx-policy.js
// - src/solver/gravity.js
// - src/solver/word-search.js
// - src/solver/telemetry.js

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

const telemetry = require("./telemetry");

const {
  OnnxPolicy,
} = require("../neural/onnx-policy");

const {
  ROWS,
  COLS,
  PLAYABLE_START_ROW,
  WILDCARD_CHAR,
} = require("../core/puzzle-config");

const DEFAULT_OPTIONS = {
  maxDepth: 120,
  maxNodes: 500000,
  maxPathsPerWord: 80,

  playableStartRow: PLAYABLE_START_ROW,

  usePolicy: true,
  fallbackToHeuristic: true,

  // 候補が少ない局面では heuristic の方が安定
  policyMinBranch: 20,

  // Policy を強くしすぎない
  policyWeight: 0.25,
  heuristicWeight: 0.75,

  // heuristic 上位だけを Policy で再ランキングする
  policyTopK: 8,

  targetWildcards: 0,

  useVisited: true,
  useWordOnce: true,

  verbose: false,

  policy: null,
};


let DEFAULT_POLICY = null;


function normalizeCell(cell) {
  if (cell === null || cell === undefined) {
    return null;
  }

  if (cell === "" || cell === "・") {
    return null;
  }

  if (cell === "Ｆ") {
    return WILDCARD_CHAR;
  }

  return String(cell);
}


function boardHash(board) {
  return board
    .map((row) =>
      row.map((cell) => normalizeCell(cell) ?? ".").join("")
    )
    .join("|");
}


function usedWordsKey(usedWords) {
  return Array.from(usedWords).sort().join(",");
}


function stateHash(board, usedWords) {
  return `${boardHash(board)}::${usedWordsKey(usedWords)}`;
}


function countWildcards(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      const normalized = normalizeCell(cell);

      if (normalized === WILDCARD_CHAR) {
        count += 1;
      }
    }
  }

  return count;
}


function boardToTensor(board) {
  const map = new Map();
  let nextId = 1;

  return board.map((row) =>
    row.map((cell) => {
      const normalized = normalizeCell(cell);

      if (normalized === null) {
        return 0;
      }

      if (!map.has(normalized)) {
        map.set(normalized, nextId);
        nextId += 1;
      }

      return map.get(normalized);
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


function moveDirection(move) {
  if (move.direction === "H" || move.direction === 0) {
    return "H";
  }

  if (move.direction === "V" || move.direction === 1) {
    return "V";
  }

  if (move.path && move.path.length >= 2) {
    const [r0, c0] = move.path[0];
    const [r1, c1] = move.path[1];

    if (r1 === r0 && c1 === c0 + 1) {
      return "H";
    }

    if (r1 === r0 + 1 && c1 === c0) {
      return "V";
    }
  }

  return "H";
}


function encodedDirection(move) {
  return moveDirection(move) === "V" ? 1 : 0;
}


function moveToEncoded(move, wordToId) {
  if (!move || !move.path || move.path.length === 0) {
    throw new Error(`Invalid move: ${JSON.stringify(move)}`);
  }

  const word = String(move.word);
  const wordId = wordToId.get(word);

  if (wordId === undefined) {
    throw new Error(`Unknown word for ONNX policy: ${word}`);
  }

  const [row, col] = move.path[0];

  return [
    Number(wordId),
    Number(row),
    Number(col),
    encodedDirection(move),
  ];
}


function buildWordToId(policy, wordsFromOptions) {
  const words =
    policy && Array.isArray(policy.words) && policy.words.length > 0
      ? policy.words
      : wordsFromOptions || [];

  const map = new Map();

  words.forEach((word, index) => {
    map.set(String(word), index);
  });

  return map;
}


function normalizeHeuristicScores(moves) {
  return moves.map((move) => Number(move.score || 0));
}


function attachPolicyInfo(originalMoves, orderedResult, encodedToMove) {
  const out = [];

  for (let i = 0; i < orderedResult.moves.length; i++) {
    const encoded = orderedResult.moves[i];
    const key = encoded.join(",");
    const move = encodedToMove.get(key);

    if (!move) {
      continue;
    }

    out.push({
      ...move,
      policyScore: orderedResult.policyScores[i],
      heuristicScore: orderedResult.heuristicScores[i],
      combinedScore: orderedResult.combinedScores[i],
    });
  }

  // 万が一、変換漏れがあれば元の順で後ろに残す
  if (out.length !== originalMoves.length) {
    const seen = new Set(
      out.map((move) => moveToPathKey(move))
    );

    for (const move of originalMoves) {
      const key = moveToPathKey(move);

      if (!seen.has(key)) {
        out.push(move);
      }
    }
  }

  return out;
}


function moveToPathKey(move) {
  return `${move.word}:${move.path.map(([r, c]) => `${r},${c}`).join(";")}`;
}


async function getPolicy(options) {
  if (options.policy) {
    return options.policy;
  }

  if (DEFAULT_POLICY) {
    return DEFAULT_POLICY;
  }

  const policy = new OnnxPolicy();

  await policy.load();

  DEFAULT_POLICY = policy;

  return DEFAULT_POLICY;
}

async function orderMovesSmart(board, moves, options, depth) {
  if (!moves || moves.length === 0) {
    return [];
  }

  const heuristicOrdered = orderMoves(moves);

  if (!options.usePolicy) {
    return heuristicOrdered;
  }

  if (moves.length < options.policyMinBranch) {
    return heuristicOrdered;
  }

  const policyTopK = Math.max(
    1,
    Number(options.policyTopK || heuristicOrdered.length)
  );

  // 全候補を Policy に投げると、Policy が外した時に大悪化しやすい。
  // そのため heuristic 上位だけを Policy で再ランキングする。
  const policyCandidates = heuristicOrdered.slice(0, policyTopK);
  const tailMoves = heuristicOrdered.slice(policyTopK);

  try {
    const policy = await getPolicy(options);
    const wordToId = buildWordToId(policy, options.words);

    const encodedMoves = [];
    const encodedToMove = new Map();

    for (const move of policyCandidates) {
      const encoded = moveToEncoded(move, wordToId);
      encodedMoves.push(encoded);
      encodedToMove.set(encoded.join(","), move);
    }

    const heuristicScores = normalizeHeuristicScores(policyCandidates);

    const orderedResult = await policy.orderEncodedMoves({
      board,
      moves: encodedMoves,
      heuristicScores,
      targetF:
        options.targetWildcards ??
        countWildcards(board),
      depth,
      branchCount: moves.length,
      policyWeight: options.policyWeight,
      heuristicWeight: options.heuristicWeight,
    });

    const rerankedTop = attachPolicyInfo(
      policyCandidates,
      orderedResult,
      encodedToMove
    );

    return [
      ...rerankedTop,
      ...tailMoves,
    ];
  } catch (error) {
    if (options.verbose) {
      console.log(
        "[solver-policy] Policy ordering failed:",
        error && error.message ? error.message : error
      );
    }

    if (!options.fallbackToHeuristic) {
      throw error;
    }

    return heuristicOrdered;
  }
}

function moveToSolvedMove(move) {
  return {
    word: move.word,
    path: move.path.map(([r, c]) => [r, c]),
    direction: moveDirection(move),
    score: move.score,
    policyScore: move.policyScore,
    heuristicScore: move.heuristicScore,
    combinedScore: move.combinedScore,
    length: move.length,
  };
}


function hasDuplicateSolvedMoves(moves) {
  const seen = new Set();

  for (const move of moves) {
    if (!move || !move.word) {
      continue;
    }

    const word = String(move.word);

    if (seen.has(word)) {
      return true;
    }

    seen.add(word);
  }

  return false;
}


async function dfsPolicy(
  board,
  stats,
  options,
  depth,
  visited,
  usedWords
) {
  telemetry.recordNode(stats);
  telemetry.recordDepth(stats, depth);

  if (shouldStopSearch(stats, options, depth)) {
    telemetry.recordDeadEnd(stats);

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
    telemetry.recordDeadEnd(stats);

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
    moves = moves.filter((move) => !usedWords.has(String(move.word)));
  }

  const branch = moves.length;
  telemetry.recordBranching(stats, branch);

  if (branch === 0) {
    telemetry.recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (branch === 1) {
    telemetry.recordForcedMove(stats);
  }

  moves = await orderMovesSmart(
    board,
    moves,
    options,
    depth
  );

  const beforeFilled = countFilledCells(board);

  for (const move of moves) {
    telemetry.recordChoice(
      stats,
      boardToTensor(board),
      move,
      moves,
      depth
    );

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
      nextUsedWords.add(String(move.word));
    }

    const childVisited =
      options.useVisited
        ? new Set(visited)
        : visited;

    const result = await dfsPolicy(
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

  telemetry.recordBacktrack(stats);

  return {
    solved: false,
    board,
  };
}


async function solvePolicy(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const board = cloneBoard(inputBoard);
  const stats = telemetry.createStats();

  const start = performance.now();

  const result = await dfsPolicy(
    board,
    stats,
    options,
    0,
    new Set(),
    new Set()
  );

  const elapsed = performance.now() - start;

  telemetry.finalizeStats(stats, elapsed);

  if (options.useWordOnce && hasDuplicateSolvedMoves(stats.solvedMoves)) {
    return {
      solved: false,
      board: result.board,
      stats,
      usedPolicy: options.usePolicy,
      reason: "duplicate word in solvedMoves",
    };
  }

  return {
    solved: result.solved,
    board: result.board,
    stats,
    usedPolicy: Boolean(options.usePolicy),
  };
}


async function compareSolvers(inputBoard, options = {}) {
  const {
    solve,
  } = require("./solver");

  const normal = solve(inputBoard, options);

  const policy = await solvePolicy(inputBoard, {
    ...options,
    usePolicy: true,
  });

  return {
    normal,
    policy,
    improvement: {
      exploredNodes:
        normal.stats.exploredNodes -
        policy.stats.exploredNodes,
      backtracks:
        normal.stats.backtracks -
        policy.stats.backtracks,
      deadEnds:
        normal.stats.deadEnds -
        policy.stats.deadEnds,
      elapsedMs:
        normal.stats.elapsedMs -
        policy.stats.elapsedMs,
    },
  };
}


async function countSolutionsPolicy(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const limit = options.limit ?? 2;
  let count = 0;

  async function inner(board, depth, visited, usedWords) {
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
      moves = moves.filter((move) => !usedWords.has(String(move.word)));
    }

    moves = await orderMovesSmart(
      board,
      moves,
      options,
      depth
    );

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
        nextUsedWords.add(String(move.word));
      }

      const childVisited =
        options.useVisited
          ? new Set(visited)
          : visited;

      await inner(
        nextBoard,
        depth + 1,
        childVisited,
        nextUsedWords
      );
    }
  }

  await inner(
    cloneBoard(inputBoard),
    0,
    new Set(),
    new Set()
  );

  return count;
}


function resetDefaultPolicyCache() {
  DEFAULT_POLICY = null;
}


module.exports = {
  solvePolicy,
  dfsPolicy,
  orderMovesSmart,
  compareSolvers,
  countSolutionsPolicy,

  boardToTensor,
  boardHash,
  stateHash,

  resetDefaultPolicyCache,
};