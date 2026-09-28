// src/generator/evaluate.js

function computeDifficulty(stats) {
  return (
    stats.exploredNodes * 2.0 +
    stats.backtracks * 20.0 +
    stats.deadEnds * 25.0 +
    stats.maxDepth * 3.0 -
    stats.forcedMoves * 2.0 +
    stats.elapsedMs * 0.01
  );
}

function computeReverseDifficulty(puzzle) {
  const filled = puzzle.board
    .flat()
    .filter((v) => v !== null).length;

  const moveCount = puzzle.solutionMoves.length;

  const avgWordLength =
    moveCount === 0
      ? 0
      : puzzle.solutionMoves.reduce(
          (sum, move) => sum + [...move.word].length,
          0
        ) / moveCount;

  const oneMovePenalty = moveCount <= 1 ? 1000 : 0;

  return (
    filled * 1.5 +
    moveCount * 30 +
    avgWordLength * 4 -
    oneMovePenalty
  );
}

function computeHybridDifficulty(puzzle, stats) {
  const reverseScore = computeReverseDifficulty(puzzle);
  const solverScore = computeDifficulty(stats);

  const branchKeys = Object.keys(
    stats.branchingHistogram || {}
  ).map(Number);

  const maxBranch =
    branchKeys.length === 0
      ? 0
      : Math.max(...branchKeys);

  const branchBonus =
    maxBranch >= 2 ? maxBranch * 30 : -100;

  const forcedPenalty = stats.forcedMoves * 5;

  return (
    reverseScore +
    solverScore +
    branchBonus -
    forcedPenalty
  );
}

module.exports = {
  computeDifficulty,
  computeReverseDifficulty,
  computeHybridDifficulty,
};