"use strict";

/*
 * モード・レベル・得点の定義です。
 * 遊びの調整は、基本的にこのファイルだけで行えます。
 */

const DEFAULT_MODE = "country";

const MODES = Object.freeze({
  country: { label: "国名", wordFile: "countries.json" },
  capital: { label: "首都名", wordFile: "capitals.json" },
  pokemon: { label: "ポケモン", wordFile: "pokemon.json" },
});

const COUNTDOWN_MS = 3000;
const MATCH_DURATION_MS = 3 * 60 * 1000;

// 全消しできなかった盤面でも、消したマス数に応じて入る点数
const PARTIAL_POINTS_PER_CELL = 2;

// 単語の長さ（選択できるのは最大5マス）
const MIN_WORD_LENGTH = 2;
const MAX_WORD_LENGTH = 5;

/*
 * 各レベルの盤面と生成条件。
 *
 * cols / rows        … 盤面の列数・行数
 * visibleRows        … 画面に見える下側の行数（それより上は落ちてくるまで見えない）
 * maxWordLength      … 想定解で使う単語の最大文字数
 * minWildcards ほか  … ワイルドカードF の個数
 * minSteps ほか      … 想定解の手数
 * minHorizontalMoves … 想定解に必要な横手・縦手の最小数
 * rejectVerticalOnly … 縦の手だけで全消しできる盤面を除外する（全レベルで true）
 * clearPoints        … 全消しで入る点数
 */
const LEVELS = Object.freeze([
  { level: 1, cols: 3, rows: 3, visibleRows: 3, maxWordLength: 3,
    minWildcards: 0, maxWildcards: 0, minSteps: 3, maxSteps: 4,
    minHorizontalMoves: 1, minVerticalMoves: 0, maxVerticalMoveRate: 1,
    rejectVerticalOnly: true, clearPoints: 100 },
  { level: 2, cols: 4, rows: 3, visibleRows: 3, maxWordLength: 4,
    minWildcards: 0, maxWildcards: 0, minSteps: 3, maxSteps: 6,
    minHorizontalMoves: 1, minVerticalMoves: 0, maxVerticalMoveRate: 1,
    rejectVerticalOnly: true, clearPoints: 150 },
  { level: 3, cols: 4, rows: 4, visibleRows: 4, maxWordLength: 4,
    minWildcards: 0, maxWildcards: 0, minSteps: 4, maxSteps: 7,
    minHorizontalMoves: 1, minVerticalMoves: 0, maxVerticalMoveRate: 1,
    rejectVerticalOnly: true, clearPoints: 200 },
  { level: 4, cols: 5, rows: 4, visibleRows: 4, maxWordLength: 5,
    minWildcards: 0, maxWildcards: 1, minSteps: 4, maxSteps: 8,
    minHorizontalMoves: 1, minVerticalMoves: 0, maxVerticalMoveRate: 1,
    rejectVerticalOnly: true, clearPoints: 300 },
  { level: 5, cols: 5, rows: 5, visibleRows: 5, maxWordLength: 5,
    minWildcards: 1, maxWildcards: 2, minSteps: 5, maxSteps: 10,
    minHorizontalMoves: 2, minVerticalMoves: 1, maxVerticalMoveRate: 0.7,
    rejectVerticalOnly: true, clearPoints: 400 },
  { level: 6, cols: 5, rows: 6, visibleRows: 5, maxWordLength: 5,
    minWildcards: 1, maxWildcards: 2, minSteps: 6, maxSteps: 12,
    minHorizontalMoves: 2, minVerticalMoves: 1, maxVerticalMoveRate: 0.7,
    rejectVerticalOnly: true, clearPoints: 550 },
  { level: 7, cols: 5, rows: 7, visibleRows: 5, maxWordLength: 5,
    minWildcards: 2, maxWildcards: 3, minSteps: 7, maxSteps: 14,
    minHorizontalMoves: 3, minVerticalMoves: 1, maxVerticalMoveRate: 0.65,
    rejectVerticalOnly: true, clearPoints: 700 },
  { level: 8, cols: 5, rows: 8, visibleRows: 5, maxWordLength: 5,
    minWildcards: 2, maxWildcards: 4, minSteps: 9, maxSteps: 16,
    minHorizontalMoves: 4, minVerticalMoves: 2, maxVerticalMoveRate: 0.65,
    rejectVerticalOnly: true, clearPoints: 900 },
  { level: 9, cols: 5, rows: 8, visibleRows: 5, maxWordLength: 5,
    minWildcards: 3, maxWildcards: 5, minSteps: 10, maxSteps: 18,
    minHorizontalMoves: 4, minVerticalMoves: 2, maxVerticalMoveRate: 0.65,
    rejectVerticalOnly: true, clearPoints: 1100 },
  { level: 10, cols: 5, rows: 8, visibleRows: 5, maxWordLength: 5,
    minWildcards: 4, maxWildcards: 6, minSteps: 11, maxSteps: 18,
    minHorizontalMoves: 4, minVerticalMoves: 2, maxVerticalMoveRate: 0.65,
    rejectVerticalOnly: true, clearPoints: 1300 },
  { level: 11, cols: 5, rows: 8, visibleRows: 5, maxWordLength: 5,
    minWildcards: 5, maxWildcards: 6, minSteps: 12, maxSteps: 18,
    minHorizontalMoves: 5, minVerticalMoves: 2, maxVerticalMoveRate: 0.6,
    rejectVerticalOnly: true, clearPoints: 1500 },
  { level: 12, cols: 5, rows: 8, visibleRows: 5, maxWordLength: 5,
    minWildcards: 6, maxWildcards: 6, minSteps: 13, maxSteps: 18,
    minHorizontalMoves: 5, minVerticalMoves: 3, maxVerticalMoveRate: 0.6,
    rejectVerticalOnly: true, clearPoints: 1800 },
]);

/*
 * 難易度の重み（tools/balance.js が使います）
 *
 *   難易度 = 手数 × step + ワイルドカード数 × wildcard + 当てずっぽう度（ビット）× bit
 *
 * 当てずっぽう度は、想定解をたどる各局面で
 *   log2(指せる手の数 / 全消しにつながる手の数)
 * を足したものです。罠の手が多く、正しい手が少ない盤面ほど大きくなります。
 */
const DIFFICULTY_WEIGHTS = Object.freeze({
  step: 1,
  wildcard: 1.5,
  bit: 2.5,
});

function levelConfig(level) {
  const index = Math.min(
    Math.max(1, Math.floor(Number(level) || 1)),
    LEVELS.length
  );

  return LEVELS[index - 1];
}

// 最終レベルをクリアしたあとも、最終レベルの点数で続きます。
function clearPoints(level) {
  return levelConfig(level).clearPoints;
}

module.exports = {
  DEFAULT_MODE,
  MODES,
  COUNTDOWN_MS,
  MATCH_DURATION_MS,
  PARTIAL_POINTS_PER_CELL,
  MIN_WORD_LENGTH,
  MAX_WORD_LENGTH,
  LEVELS,
  levelConfig,
  clearPoints,
  DIFFICULTY_WEIGHTS,
};