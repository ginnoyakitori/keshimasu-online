// src/generator/generator.js

const {
  generateFullBoardByReverseSimulation,
  countWildcards,
} = require("./reverse-generator");

const {
  computeHybridDifficulty,
  computeReverseDifficulty,
} = require("./evaluate");

const {
  getPlayableWords,
} = require("./words");

const {
  solve,
} = require("../solver");


function getMoveWord(move) {
  if (!move) {
    return null;
  }

  if (typeof move.word === "string" && move.word.length > 0) {
    return move.word;
  }

  return null;
}

function findDuplicateWordsInMoves(moves) {
  const counts = new Map();

  if (!Array.isArray(moves)) {
    return [];
  }

  for (const move of moves) {
    const word = getMoveWord(move);

    if (!word) {
      continue;
    }

    counts.set(
      word,
      (counts.get(word) || 0) + 1
    );
  }

  const duplicates = [];

  for (const [word, count] of counts.entries()) {
    if (count >= 2) {
      duplicates.push({
        word,
        count,
      });
    }
  }

  return duplicates;
}

function hasDuplicateWordsInMoves(moves) {
  return findDuplicateWordsInMoves(moves).length > 0;
}

function hasValidNoRepeatSolution(result) {
  if (!result || !result.solved) {
    return false;
  }

  const solvedMoves =
    result.stats && Array.isArray(result.stats.solvedMoves)
      ? result.stats.solvedMoves
      : [];

  if (!solvedMoves.length) {
    return false;
  }

  return !hasDuplicateWordsInMoves(solvedMoves);
}

function isGoodPuzzle(candidate, result) {
  if (!result.solved) {
    return false;
  }

  if (!candidate.solutionMoves) {
    return false;
  }

  if (candidate.solutionMoves.length < 6) {
    return false;
  }

  if (result.stats.maxDepth < 3) {
    return false;
  }

  if (hasDuplicateWordsInMoves(candidate.solutionMoves)) {
    return false;
  }

  if (!hasValidNoRepeatSolution(result)) {
    return false;
  }

  return true;
}

function solveCandidate(candidate, words, options) {
  return solve(candidate.board, {
    words,
    playableStartRow: 3,
    allowLoose: false,
    maxNodes: options.solverMaxNodes ?? 30000,
    maxDepth: options.solverMaxDepth ?? 80,
    maxPathsPerWord: options.solverMaxPathsPerWord ?? 30,
    useWordOnce: true,
  });
}

function generateCandidate(words, targetWildcards, options) {
  return generateFullBoardByReverseSimulation({
    words,
    targetCells: 40,
    targetWildcards,
    maxPlanAttempts: options.maxPlanAttempts ?? 80,
    maxPlacementAttempts: options.maxPlacementAttempts ?? 1200,
  });
}

function makeReturnPayload(candidate, score, result, targetWildcards) {
  const solverSolvedMoves =
    result &&
    result.stats &&
    Array.isArray(result.stats.solvedMoves)
      ? result.stats.solvedMoves
      : [];

  return {
    board: candidate.board,
    score,
    // 新ルール下では Solver が見つけた no-repeat 解を正として返す
    solutionMoves: solverSolvedMoves.length
      ? solverSolvedMoves
      : candidate.solutionMoves,
    solverStats: result ? result.stats : null,
    targetWildcards,
    actualWildcards: countWildcards(candidate.board),
  };
}

function generatePuzzle(iterations = 50, options = {}) {
  let best = null;
  let bestScore = -Infinity;
  let bestResult = null;

  const timeLimitMs =
    options.timeLimitMs ?? 15000;

  const targetWildcards =
    options.targetWildcards ?? 0;

  const startTime = Date.now();

  const words = getPlayableWords({
    minLen: 2,
    maxLen: 5,
  });

  let generatedCount = 0;
  let solvedCount = 0;
  let failedCount = 0;
  let duplicateRejectedCount = 0;

  for (let i = 0; i < iterations; i++) {
    if (Date.now() - startTime > timeLimitMs) {
      console.log("Generator time limit reached");
      break;
    }

    if (i % 5 === 0) {
      console.log(
        `[generator] ${i}/${iterations} ` +
        `best=${bestScore.toFixed(2)} ` +
        `generated=${generatedCount} ` +
        `solved=${solvedCount} ` +
        `failed=${failedCount} ` +
        `dupRejected=${duplicateRejectedCount} ` +
        `F=${targetWildcards}`
      );
    }

    const candidate = generateCandidate(
      words,
      targetWildcards,
      options
    );

    if (!candidate) {
      failedCount++;
      continue;
    }

    if (
      countWildcards(candidate.board) !==
      targetWildcards
    ) {
      failedCount++;
      continue;
    }

    if (hasDuplicateWordsInMoves(candidate.solutionMoves)) {
      duplicateRejectedCount++;
      failedCount++;
      continue;
    }

    generatedCount++;

    const result = solveCandidate(
      candidate,
      words,
      options
    );

    if (!result.solved) {
      failedCount++;
      continue;
    }

    if (!hasValidNoRepeatSolution(result)) {
      duplicateRejectedCount++;
      failedCount++;
      continue;
    }

    solvedCount++;

    const score =
      computeHybridDifficulty(
        candidate,
        result.stats
      );

    if (
      score > bestScore &&
      isGoodPuzzle(candidate, result)
    ) {
      bestScore = score;
      best = candidate;
      bestResult = result;
    }
  }

  if (!best) {
    console.log(
      "No strong puzzle found. Returning fallback puzzle..."
    );

    for (let i = 0; i < 30; i++) {
      const fallback =
        generateFullBoardByReverseSimulation({
          words,
          targetCells: 40,
          targetWildcards,
          maxPlanAttempts: options.fallbackMaxPlanAttempts ?? 100,
          maxPlacementAttempts: options.fallbackMaxPlacementAttempts ?? 1500,
        });

      if (!fallback) {
        continue;
      }

      if (
        countWildcards(fallback.board) !==
        targetWildcards
      ) {
        continue;
      }

      if (hasDuplicateWordsInMoves(fallback.solutionMoves)) {
        continue;
      }

      const fallbackResult = solveCandidate(
        fallback,
        words,
        options
      );

      if (!fallbackResult.solved) {
        continue;
      }

      if (!hasValidNoRepeatSolution(fallbackResult)) {
        continue;
      }

      const fallbackScore =
        computeHybridDifficulty(
          fallback,
          fallbackResult.stats
        );

      return makeReturnPayload(
        fallback,
        fallbackScore,
        fallbackResult,
        targetWildcards
      );
    }

    throw new Error(
      "Failed to generate puzzle with fixed wildcard count and no duplicate words."
    );
  }

  return makeReturnPayload(
    best,
    bestScore,
    bestResult,
    targetWildcards
  );
}

module.exports = {
  generatePuzzle,
  findDuplicateWordsInMoves,
  hasDuplicateWordsInMoves,
};