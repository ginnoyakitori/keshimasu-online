// src/index.js

const { generatePuzzle } = require("./generator");

const {
  solve,
  solvePolicy,
  boardToString,
} = require("./solver");

const {
  getPlayableWords,
} = require("./generator/words");

const {
  MAX_WORD_LENGTH,
} = require("./core/puzzle-config");

async function main() {
  const words = getPlayableWords({
    minLen: 2,
    maxLen: MAX_WORD_LENGTH,
  });

  const targetWildcards = 2;

  const generated = generatePuzzle(30, {
    timeLimitMs: 15000,
    targetWildcards,
  });

  console.log("\n=== PUZZLE ===\n");
  console.log(boardToString(generated.board));

  console.log("\n=== NORMAL SOLVER ===\n");

  const normal = solve(generated.board, {
    words,
    targetWildcards,
    maxNodes: 100000,
    maxDepth: 100,
    maxPathsPerWord: 50,
  });

  console.log("Solved:", normal.solved);
  console.log("Nodes:", normal.stats.exploredNodes);
  console.log("Backtracks:", normal.stats.backtracks);
  console.log("DeadEnds:", normal.stats.deadEnds);
  console.log("Elapsed:", normal.stats.elapsedMs);

  console.log("\n=== POLICY SOLVER ===\n");

  const policy = await solvePolicy(generated.board, {
    words,
    targetWildcards,
    maxNodes: 100000,
    maxDepth: 100,
    maxPathsPerWord: 50,
    usePolicy: true,
    verbose: true,
  });

  console.log("Solved:", policy.solved);
  console.log("Used Policy:", policy.usedPolicy);
  console.log("Nodes:", policy.stats.exploredNodes);
  console.log("Backtracks:", policy.stats.backtracks);
  console.log("DeadEnds:", policy.stats.deadEnds);
  console.log("Elapsed:", policy.stats.elapsedMs);

  console.log("\n=== IMPROVEMENT ===\n");

  console.log(
    "Nodes:",
    normal.stats.exploredNodes -
      policy.stats.exploredNodes
  );

  console.log(
    "Backtracks:",
    normal.stats.backtracks -
      policy.stats.backtracks
  );

  console.log(
    "DeadEnds:",
    normal.stats.deadEnds -
      policy.stats.deadEnds
  );
}

main();