"use strict";

/*
 * 問題の難易度を測り、レベルごとに「だんだん難しくなる」ように選び直します。
 *
 *   測る      … Solver で想定解をたどり、当てずっぽう度（ビット）を求める
 *   点数にする … 手数・ワイルドカード数・ビットを重みつきで足す（levels.js）
 *   選ぶ      … 各レベルの目標難易度を、前のレベルより必ず上がるように決め、
 *               目標に近い問題から順にプールへ入れる
 */

const { DIFFICULTY_WEIGHTS } = require("../src/multiplayer/levels");

function round(value, digits = 2) {
  const factor = 10 ** digits;

  return Math.round(value * factor) / factor;
}

function difficultyOf(puzzle, analysis) {
  return (
    DIFFICULTY_WEIGHTS.step * puzzle.steps +
    DIFFICULTY_WEIGHTS.wildcard * puzzle.wildcards +
    DIFFICULTY_WEIGHTS.bit * analysis.guessBits
  );
}

/*
 * 1問を測り、難易度などを付けた新しい問題を返します。
 * 想定解を再現できなかったときは null です。
 */
function measurePuzzle({ solver, puzzle, cfg, nodeBudget }) {
  const analysis = solver.analyze({
    board: puzzle.board,
    solution: puzzle.solution,
    visibleRows: cfg.visibleRows,
    nodeBudget,
  });

  if (!analysis.ok) {
    return null;
  }

  return {
    ...puzzle,
    difficulty: round(difficultyOf(puzzle, analysis), 1),
    guessBits: round(analysis.guessBits, 2),
    rootMoves: analysis.rootMoves,
    rootWinning: analysis.rootWinning,
    measureExhausted: analysis.exhausted || undefined,
  };
}

function percentile(sortedValues, ratio) {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.round((sortedValues.length - 1) * ratio))
  );

  return sortedValues[index];
}

function summarize(puzzles) {
  const values = puzzles.map((puzzle) => puzzle.difficulty).sort((a, b) => a - b);

  if (values.length === 0) {
    return { count: 0, min: 0, p25: 0, median: 0, p75: 0, max: 0, mean: 0 };
  }

  return {
    count: values.length,
    min: values[0],
    p25: percentile(values, 0.25),
    median: percentile(values, 0.5),
    p75: percentile(values, 0.75),
    max: values[values.length - 1],
    mean: round(values.reduce((sum, value) => sum + value, 0) / values.length, 1),
  };
}

/*
 * candidatesByLevel: { 1: [測定済みの問題...], 2: [...], ... }
 *
 * 各レベルの目標 = そのレベルの候補の中央値。ただし前のレベルの目標より
 * minGap 以上高くなるよう押し上げます。目標に近い順に count 問を選びます。
 */
function selectBalanced(candidatesByLevel, count, minGap = 2.5) {
  const levels = Object.keys(candidatesByLevel)
    .map(Number)
    .sort((a, b) => a - b);

  const selected = {};
  const targets = {};
  let previousTarget = -Infinity;

  for (const level of levels) {
    const candidates = candidatesByLevel[level];

    if (candidates.length === 0) {
      selected[level] = [];
      continue;
    }

    const median = summarize(candidates).median;
    const target = Math.max(median, previousTarget + minGap);

    targets[level] = round(target, 1);
    previousTarget = target;

    selected[level] = candidates
      .slice()
      .sort(
        (left, right) =>
          Math.abs(left.difficulty - target) - Math.abs(right.difficulty - target)
      )
      .slice(0, count)
      // 同じ難易度の問題が並ばないよう、選んだあとは難易度順に整える
      .sort((left, right) => left.difficulty - right.difficulty);
  }

  return { selected, targets };
}

// 平均難易度が、レベルが上がるたびに上がっているか
function checkMonotonic(selected) {
  const problems = [];
  let previous = null;

  for (const level of Object.keys(selected).map(Number).sort((a, b) => a - b)) {
    const { mean } = summarize(selected[level]);

    if (previous && mean <= previous.mean) {
      problems.push(
        `Lv${level}（平均 ${mean}）が Lv${previous.level}（平均 ${previous.mean}）以下です`
      );
    }

    previous = { level, mean };
  }

  return problems;
}

module.exports = {
  measurePuzzle,
  summarize,
  selectBalanced,
  checkMonotonic,
  difficultyOf,
};