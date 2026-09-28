// src/neural/dataset.js
//
// Replay Buffer 縺九ｉ Policy 蟄ｦ鄙堤畑繝・・繧ｿ繧剃ｽ懊ｋ
//
// 蜈･蜉・
//   data/training/selfplay/f0/*.json
//   data/training/selfplay/f1/*.json
//   ...
//
// 蜃ｺ蜉帷畑繝・・繧ｿ:
//   1. policyExamples
//      board + candidates + labelIndex
//
//   2. moveExamples
//      board + candidateMove + label
//
// 逕ｨ騾・
//   - policyExamples:
//       蛟呵｣徇ove縺ｮ荳ｭ縺九ｉ豁｣隗｣move繧貞・鬘槭☆繧句ｭｦ鄙・
//
//   - moveExamples:
//       board + move 縺瑚憶縺・焔縺九←縺・°繧・蛟､蛻・｡槭☆繧句ｭｦ鄙・
//

const fs = require("fs");
const path = require("path");

const {
  findAllMoves,
  orderMoves,
  readPath,
} = require("../solver/word-search");

const {
  removePathAndApplyGravity,
  cloneBoard,
} = require("../solver/gravity");

const {
  MAX_WORD_LENGTH,
  WILDCARD_CHAR,
  PLAYABLE_START_ROW,
} = require("../core/puzzle-config");

const {
  getPlayableWords,
} = require("../generator/words");

function ensureDir(dir) {
  fs.mkdirSync(dir, {
    recursive: true,
  });
}

function getSelfplayRoot() {
  return path.join(
    __dirname,
    "../../data/training/selfplay"
  );
}

function getDatasetDir() {
  return path.join(
    __dirname,
    "../../data/datasets/policy"
  );
}

function listReplayFiles(options = {}) {
  const root =
    options.root || getSelfplayRoot();

  const targetWildcards =
    options.targetWildcards;

  if (!fs.existsSync(root)) {
    return [];
  }

  const leagues =
    targetWildcards === undefined
      ? fs
          .readdirSync(root)
          .filter((name) => name.startsWith("f"))
      : [`f${targetWildcards}`];

  const files = [];

  for (const league of leagues) {
    const dir = path.join(root, league);

    if (!fs.existsSync(dir)) {
      continue;
    }

    for (const file of fs.readdirSync(dir)) {
      if (!file.endsWith(".json")) {
        continue;
      }

      files.push(path.join(dir, file));
    }
  }

  return files;
}

function loadReplay(file) {
  const json = fs.readFileSync(file, "utf8");
  return JSON.parse(json);
}

function pathKey(path) {
  return path
    .map(([r, c]) => `${r},${c}`)
    .join(";");
}

function movesEqual(a, b) {
  if (!a || !b) {
    return false;
  }

  if (a.word !== b.word) {
    return false;
  }

  if (!a.path || !b.path) {
    return false;
  }

  return pathKey(a.path) === pathKey(b.path);
}

function getMoveDirection(move) {
  const path = move.path;

  if (!path || path.length < 2) {
    return "unknown";
  }

  const [r0, c0] = path[0];
  const [r1, c1] = path[1];

  if (r1 === r0 && c1 === c0 + 1) {
    return "H";
  }

  if (r1 === r0 + 1 && c1 === c0) {
    return "V";
  }

  return "invalid";
}

function countWildcardsInBoard(board) {
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

function createCharMap(words) {
  const chars = new Set();

  for (const word of words) {
    for (const ch of [...word]) {
      chars.add(ch);
    }
  }

  chars.add(WILDCARD_CHAR);

  const map = {
    null: 0,
  };

  let id = 1;

  for (const ch of [...chars].sort()) {
    map[ch] = id;
    id++;
  }

  return map;
}

function boardToTensor(board, charMap) {
  return board.map((row) =>
    row.map((cell) => {
      if (cell === null) {
        return 0;
      }

      return charMap[cell] || 0;
    })
  );
}

function moveToFeature(move, board) {
  const path = move.path;
  const direction = getMoveDirection(move);

  const start = path[0];
  const end = path[path.length - 1];

  const boardText = readPath(board, path);
  const wildcardCount = [...boardText].filter(
    (ch) => ch === WILDCARD_CHAR
  ).length;

  return {
    word: move.word,
    path: move.path,
    direction,
    start,
    end,
    length: path.length,
    boardText,
    wildcardCount,
    score: move.score ?? 0,
  };
}

function findLabelIndex(candidates, chosenMove) {
  for (let i = 0; i < candidates.length; i++) {
    if (movesEqual(candidates[i], chosenMove)) {
      return i;
    }
  }

  return -1;
}

/**
 * 1 replay 縺九ｉ蟄ｦ鄙剃ｾ九ｒ菴懊ｋ
 *
 * replay.board:
 *   蛻晄悄逶､髱｢
 *
 * replay.solver.solvedMoves:
 *   solver 縺悟ｮ滄圀縺ｫ隗｣縺・◆謇矩・
 */
function buildExamplesFromReplay(replay, options = {}) {
  const words =
    options.words ||
    replay.words ||
    getPlayableWords({
      minLen: 2,
      maxLen: MAX_WORD_LENGTH,
    });

  const charMap =
    options.charMap || createCharMap(words);

  const maxPathsPerWord =
    options.maxPathsPerWord ?? 80;

  const policyExamples = [];
  const moveExamples = [];

  const solvedMoves =
    replay.solver?.solvedMoves ||
    replay.solverSolutionMoves ||
    [];

  if (!replay.board || solvedMoves.length === 0) {
    return {
      policyExamples,
      moveExamples,
      skipped: true,
      reason: "missing board or solvedMoves",
    };
  }

  let board = cloneBoard(replay.board);

  for (let depth = 0; depth < solvedMoves.length; depth++) {
    const chosenMove = solvedMoves[depth];

    let candidates = findAllMoves(board, {
      words,
      playableStartRow: PLAYABLE_START_ROW,
      maxPathsPerWord,
    });

    candidates = orderMoves(candidates);

    const labelIndex =
      findLabelIndex(candidates, chosenMove);

    if (labelIndex === -1) {
      return {
        policyExamples,
        moveExamples,
        skipped: true,
        reason: `chosen move not found at depth ${depth}`,
      };
    }

    const boardTensor =
      boardToTensor(board, charMap);

    const candidateFeatures =
      candidates.map((move) =>
        moveToFeature(move, board)
      );

    const chosenFeature =
      moveToFeature(chosenMove, board);

    const policyExample = {
      puzzleId: replay.id || null,
      epoch: replay.epoch,
      targetWildcards: replay.targetWildcards,
      actualWildcards:
        replay.actualWildcards ??
        countWildcardsInBoard(replay.board),

      depth,
      board: cloneBoard(board),
      boardTensor,

      candidates: candidateFeatures,
      labelIndex,

      chosenMove: chosenFeature,

      branchCount: candidates.length,

      stats: {
        totalExploredNodes:
          replay.solver?.stats?.exploredNodes,
        totalBacktracks:
          replay.solver?.stats?.backtracks,
        totalDeadEnds:
          replay.solver?.stats?.deadEnds,
      },
    };

    policyExamples.push(policyExample);

    for (let i = 0; i < candidates.length; i++) {
      const move = candidates[i];

      moveExamples.push({
        puzzleId: replay.id || null,
        epoch: replay.epoch,
        targetWildcards: replay.targetWildcards,
        actualWildcards:
          replay.actualWildcards ??
          countWildcardsInBoard(replay.board),

        depth,
        board: cloneBoard(board),
        boardTensor,

        move: moveToFeature(move, board),

        label: i === labelIndex ? 1 : 0,

        branchCount: candidates.length,
      });
    }

    board = removePathAndApplyGravity(
      board,
      chosenMove.path
    );
  }

  return {
    policyExamples,
    moveExamples,
    skipped: false,
  };
}

function writeJsonl(file, rows) {
  const text =
    rows.map((row) => JSON.stringify(row)).join("\n") +
    (rows.length > 0 ? "\n" : "");

  fs.writeFileSync(file, text, "utf8");
}

function saveDataset({
  policyExamples,
  moveExamples,
  summary,
  outputDir = getDatasetDir(),
}) {
  ensureDir(outputDir);

  const policyFile = path.join(
    outputDir,
    "policy-examples.jsonl"
  );

  const moveFile = path.join(
    outputDir,
    "move-examples.jsonl"
  );

  const summaryFile = path.join(
    outputDir,
    "summary.json"
  );

  writeJsonl(policyFile, policyExamples);
  writeJsonl(moveFile, moveExamples);

  fs.writeFileSync(
    summaryFile,
    JSON.stringify(summary, null, 2),
    "utf8"
  );

  return {
    policyFile,
    moveFile,
    summaryFile,
  };
}

function buildDataset(options = {}) {
  const files = listReplayFiles(options);

  const words = getPlayableWords({
    minLen: 2,
    maxLen: MAX_WORD_LENGTH,
  });

  const charMap = createCharMap(words);

  const policyExamples = [];
  const moveExamples = [];

  const skipped = [];

  for (const file of files) {
    const replay = loadReplay(file);

    const result = buildExamplesFromReplay(
      replay,
      {
        words,
        charMap,
        maxPathsPerWord:
          options.maxPathsPerWord ?? 80,
      }
    );

    if (result.skipped) {
      skipped.push({
        file,
        reason: result.reason,
      });

      continue;
    }

    policyExamples.push(
      ...result.policyExamples
    );

    moveExamples.push(
      ...result.moveExamples
    );
  }

  const byF = {};

  for (const ex of policyExamples) {
    const key = `f${ex.targetWildcards}`;

    if (!byF[key]) {
      byF[key] = {
        policyExamples: 0,
        moveExamples: 0,
        avgBranch: 0,
      };
    }

    byF[key].policyExamples++;
    byF[key].avgBranch += ex.branchCount;
  }

  for (const ex of moveExamples) {
    const key = `f${ex.targetWildcards}`;

    if (!byF[key]) {
      byF[key] = {
        policyExamples: 0,
        moveExamples: 0,
        avgBranch: 0,
      };
    }

    byF[key].moveExamples++;
  }

  for (const key of Object.keys(byF)) {
    if (byF[key].policyExamples > 0) {
      byF[key].avgBranch =
        byF[key].avgBranch /
        byF[key].policyExamples;
    }
  }

  const summary = {
    createdAt: new Date().toISOString(),

    inputReplayFiles: files.length,
    skippedFiles: skipped.length,

    policyExamples: policyExamples.length,
    moveExamples: moveExamples.length,

    byF,
    skipped,

    charMap,
  };

  const output = saveDataset({
    policyExamples,
    moveExamples,
    summary,
    outputDir:
      options.outputDir || getDatasetDir(),
  });

  return {
    ...output,
    summary,
  };
}

module.exports = {
  getSelfplayRoot,
  getDatasetDir,
  listReplayFiles,
  loadReplay,
  createCharMap,
  boardToTensor,
  moveToFeature,
  buildExamplesFromReplay,
  buildDataset,
  saveDataset,
};
