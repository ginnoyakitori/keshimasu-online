// src/selfplay/selfplay.js
//
// Generator vs Solver self-play
//
// 目的:
// - F数固定リーグで公平に比較
// - Generator は Solver が苦戦する盤面を作る
// - Solver は少ない探索で解く
// - 探索ログを replay-buffer に保存する
//

const { generatePuzzle } = require("../generator");

const {
  solve,
  boardToString,
} = require("../solver");

const {
  getPlayableWords,
} = require("../generator/words");

const {
  MAX_WORD_LENGTH,
  WILDCARD_CHAR,
} = require("../core/puzzle-config");

const {
  getEpochConfig,
} = require("./curriculum");

const {
  saveReplay,
  saveBestPuzzle,
} = require("./replay-buffer");

const {
  LeagueTable,
} = require("./league");

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

function getBranchStats(branchingHistogram = {}) {
  const keys = Object.keys(branchingHistogram).map(Number);

  const maxBranch =
    keys.length === 0 ? 0 : Math.max(...keys);

  const multiBranchCount = keys
    .filter((k) => k >= 2)
    .reduce(
      (sum, k) => sum + branchingHistogram[k],
      0
    );

  return {
    maxBranch,
    multiBranchCount,
  };
}

function computeGeneratorScore(stats, options = {}) {
  const targetWildcards =
    options.targetWildcards ?? 0;

  const actualWildcards =
    options.actualWildcards ?? 0;

  const {
    maxBranch,
    multiBranchCount,
  } = getBranchStats(stats.branchingHistogram);

  const wildcardMismatchPenalty =
    actualWildcards === targetWildcards
      ? 0
      : 100000;

  const timeoutPenalty =
    stats.exploredNodes >=
    (options.maxNodes ?? Infinity)
      ? 5000
      : 0;

  return (
    stats.exploredNodes * 2.0 +
    stats.backtracks * 35.0 +
    stats.deadEnds * 40.0 +
    stats.maxDepth * 5.0 +
    maxBranch * 10.0 +
    multiBranchCount * 25.0 -
    stats.forcedMoves * 10.0 -
    wildcardMismatchPenalty -
    timeoutPenalty
  );
}

function validateSolutionMoves(moves) {
  for (const move of moves) {
    const path = move.path;

    if (!path || path.length < 2) {
      continue;
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
  }

  return true;
}

function printShortStats({
  epoch,
  targetWildcards,
  score,
  result,
}) {
  const stats = result.stats;

  console.log(
    [
      `epoch=${epoch}`,
      `F=${targetWildcards}`,
      `score=${score.toFixed(2)}`,
      `nodes=${stats.exploredNodes}`,
      `backtracks=${stats.backtracks}`,
      `deadEnds=${stats.deadEnds}`,
      `forced=${stats.forcedMoves}`,
      `depth=${stats.maxDepth}`,
      `ms=${stats.elapsedMs.toFixed(1)}`,
    ].join(" | ")
  );
}

function printMovesCompact(title, moves) {
  console.log(`\n=== ${title} ===\n`);

  if (!moves || moves.length === 0) {
    console.log("(no moves)");
    return;
  }

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];

    const path = move.path;
    const start = path[0];
    const end = path[path.length - 1];

    const direction =
      path.length >= 2 &&
      path[0][0] === path[1][0]
        ? "横"
        : "縦";

    const index = String(i + 1).padStart(2, "0");

    console.log(
      `${index}. ${move.word.padEnd(8, " ")} ${direction} (${start[0]},${start[1]})→(${end[0]},${end[1]}) len=${path.length}`
    );
  }
}

async function selfPlay(epochs = 20, options = {}) {
  const leagueTable = new LeagueTable();

  const words = getPlayableWords({
    minLen: 2,
    maxLen: MAX_WORD_LENGTH,
  });

  let globalBest = null;
  let globalBestScore = -Infinity;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const config = {
      ...getEpochConfig(epoch),
      ...options,
    };

    const targetWildcards =
      config.targetWildcards;

    console.log("\n==============================");
    console.log(`EPOCH ${epoch}`);
    console.log("==============================");
    console.log("Config:", config);

    let generated;

    try {
      generated = generatePuzzle(
        config.generatorIterations,
        {
          timeLimitMs: config.timeLimitMs,
          targetWildcards,
        }
      );
    } catch (error) {
      console.log(
        "Generation failed:",
        error.message
      );

      continue;
    }

    const actualWildcards =
      countWildcards(generated.board);

    if (actualWildcards !== targetWildcards) {
      console.log(
        `Skip: wildcard mismatch target=${targetWildcards}, actual=${actualWildcards}`
      );

      continue;
    }

    if (
      !validateSolutionMoves(
        generated.solutionMoves
      )
    ) {
      console.log(
        "Skip: invalid generator solution moves"
      );

      continue;
    }

    const result = solve(generated.board, {
      words,
      playableStartRow: 3,
      allowLoose: false,
      maxNodes: config.solverMaxNodes,
      maxDepth: config.solverMaxDepth,
      maxPathsPerWord:
        config.solverMaxPathsPerWord,
    });

    if (!result.solved) {
      console.log("Solver failed");
      continue;
    }

    if (
      !validateSolutionMoves(
        result.stats.solvedMoves
      )
    ) {
      console.log(
        "Skip: invalid solver solution moves"
      );

      continue;
    }

    const score = computeGeneratorScore(
      result.stats,
      {
        targetWildcards,
        actualWildcards,
        maxNodes: config.solverMaxNodes,
      }
    );

    printShortStats({
      epoch,
      targetWildcards,
      score,
      result,
    });

    leagueTable.record({
      targetWildcards,
      puzzle: generated.board,
      score,
      solverStats: result.stats,
      solved: result.solved,
    });

    const replay = saveReplay({
      epoch,
      targetWildcards,
      puzzle: generated.board,
      generatorResult: generated,
      solverResult: result,
      score,
      words,
    });

    console.log("Saved replay:", replay.file);

    if (score > globalBestScore) {
      globalBestScore = score;

      globalBest = {
        epoch,
        targetWildcards,
        board: generated.board,
        generatorResult: generated,
        solverResult: result,
        score,
      };

      const best = saveBestPuzzle({
        epoch,
        targetWildcards,
        puzzle: generated.board,
        generatorResult: generated,
        solverResult: result,
        score,
      });

      console.log("NEW GLOBAL BEST:", best.file);
      console.log("\nBest board:\n");
      console.log(boardToString(generated.board));

      printMovesCompact(
        "BEST SOLVER MOVES",
        result.stats.solvedMoves
      );
    }
  }

  leagueTable.printSummary();

  return {
    best: globalBest,
    bestScore: globalBestScore,
    leagueSummary: leagueTable.getSummary(),
  };
}

module.exports = {
  selfPlay,
  computeGeneratorScore,
  countWildcards,
  validateSolutionMoves,
};