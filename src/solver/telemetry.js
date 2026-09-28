// src/solver/telemetry.js
//
// word puzzle solver telemetry
//

function createStats() {
  return {
    exploredNodes: 0,
    backtracks: 0,
    deadEnds: 0,
    forcedMoves: 0,
    maxDepth: 0,
    elapsedMs: 0,

    branchingHistogram: {},

    path: [],

    solvedMoves: [],
  };
}

function recordNode(stats) {
  stats.exploredNodes++;
}

function recordBacktrack(stats) {
  stats.backtracks++;
}

function recordDeadEnd(stats) {
  stats.deadEnds++;
}

function recordForcedMove(stats) {
  stats.forcedMoves++;
}

function recordDepth(stats, depth) {
  stats.maxDepth = Math.max(stats.maxDepth, depth);
}

function recordBranching(stats, branch) {
  stats.branchingHistogram[branch] =
    (stats.branchingHistogram[branch] || 0) + 1;
}

function recordChoice(stats, boardTensor, move, candidates, depth) {
  stats.path.push({
    boardTensor,
    selectedWord: move.word,
    selectedPath: move.path,
    candidateCount: candidates.length,
    candidates: candidates.map((m) => ({
      word: m.word,
      path: m.path,
      score: m.score,
    })),
    depth,
  });
}

function recordSolvedMove(stats, move) {
  stats.solvedMoves.push({
    word: move.word,
    path: move.path,
  });
}

function finalizeStats(stats, elapsedMs) {
  stats.elapsedMs = elapsedMs;
}

module.exports = {
  createStats,
  recordNode,
  recordBacktrack,
  recordDeadEnd,
  recordForcedMove,
  recordDepth,
  recordBranching,
  recordChoice,
  recordSolvedMove,
  finalizeStats,
};