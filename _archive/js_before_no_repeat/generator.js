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

    counts.set(word, (counts.get(word) || 0) + 1);
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

  // 生成時の逆再生プランで同じ単語を2回以上使っていたら不採用
  if (hasDuplicateWordsInMoves(candidate.solutionMoves)) {
    return false;
  }

  // Solverが見つけた解でも同じ単語を2回以上使っていたら不採用
  if (
    result.stats &&
    Array.isArray(result.stats.solvedMoves) &&
    hasDuplicateWordsInMoves(result.stats.solvedMoves)
  ) {
    return false;
  }

  return true;
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

    const candidate =
      generateFullBoardByReverseSimulation({
        words,
        targetCells: 40,
        targetWildcards,
        maxPlanAttempts: 80,
        maxPlacementAttempts: 1200,
      });

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

    // 逆再生プラン自体に同じ単語の重複使用があれば、この候補は捨てる
    if (hasDuplicateWordsInMoves(candidate.solutionMoves)) {
      duplicateRejectedCount++;
      failedCount++;
      continue;
    }

    generatedCount++;

    const result = solve(candidate.board, {
      words,
      playableStartRow: 3,
      allowLoose: false,
      maxNodes: 30000,
      maxDepth: 80,
      maxPathsPerWord: 30,
      useWordOnce: true,
    });

    if (!result.solved) {
      failedCount++;
      continue;
    }

    // Solver側の解にも同じ単語の重複使用があれば捨てる
    if (
      result.stats &&
      Array.isArray(result.stats.solvedMoves) &&
      hasDuplicateWordsInMoves(result.stats.solvedMoves)
    ) {
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
          maxPlanAttempts: 100,
          maxPlacementAttempts: 1500,
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

      // fallbackでも同じ単語を2回以上使っていたら不採用
      if (hasDuplicateWordsInMoves(fallback.solutionMoves)) {
        continue;
      }

      const fallbackResult = solve(
        fallback.board,
        {
          words,
          playableStartRow: 3,
          allowLoose: false,
          maxNodes: 30000,
          maxDepth: 80,
          maxPathsPerWord: 30,
          useWordOnce: true,
        }
      );

      if (!fallbackResult.solved) {
        continue;
      }

      if (
        fallbackResult.stats &&
        Array.isArray(fallbackResult.stats.solvedMoves) &&
        hasDuplicateWordsInMoves(fallbackResult.stats.solvedMoves)
      ) {
        continue;
      }

      return {
        board: fallback.board,
        score: fallbackResult.solved
          ? computeHybridDifficulty(
              fallback,
              fallbackResult.stats
            )
          : computeReverseDifficulty(fallback),
        solutionMoves: fallback.solutionMoves,
        solverStats: fallbackResult.stats,
        targetWildcards,
        actualWildcards:
          countWildcards(fallback.board),
      };
    }

    throw new Error(
      "Failed to generate puzzle with fixed wildcard count and no duplicate words."
    );
  }

  return {
    board: best.board,
    score: bestScore,
    solutionMoves: best.solutionMoves,
    solverStats: bestResult.stats,
    targetWildcards,
    actualWildcards:
      countWildcards(best.board),
  };
}


module.exports = {
  generatePuzzle,

  // テスト・デバッグ用に export
  findDuplicateWordsInMoves,
  hasDuplicateWordsInMoves,
};