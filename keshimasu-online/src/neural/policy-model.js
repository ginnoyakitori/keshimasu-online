// src/neural/policy-model.js
//
// Policy model
//
// 蜈･蜉・
// - boardTensor 8x5 = 40蛟・
// - move迚ｹ蠕ｴ
//
// 蜃ｺ蜉・
// - 縺昴・ move 繧貞・縺ｫ隧ｦ縺吶∋縺咲｢ｺ邇・
//
// 蠖｢蠑・
//   board + candidateMove -> score
//

const tf = require("@tensorflow/tfjs");

const {
  ROWS,
  COLS,
  MAX_WORD_LENGTH,
} = require("../core/puzzle-config");

const BOARD_SIZE = ROWS * COLS;

// board 40
// + startRow
// + startCol
// + endRow
// + endCol
// + length
// + directionH
// + directionV
// + wildcardCount
// + moveScore
// + branchCount
// + targetWildcards
const EXTRA_FEATURES = 11;

const INPUT_SIZE = BOARD_SIZE + EXTRA_FEATURES;

function createPolicyModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [INPUT_SIZE],
      units: 128,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dropout({
      rate: 0.15,
    })
  );

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dropout({
      rate: 0.1,
    })
  );

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

/**
 * boardTensor 8x5 繧・flat feature 縺ｫ縺吶ｋ
 *
 * charId 縺ｯ charVocabSize 縺ｧ蜑ｲ縺｣縺ｦ 0縲・ 莉倩ｿ代↓豁｣隕丞喧
 */
function flattenBoardTensor(boardTensor, charVocabSize) {
  const divisor = Math.max(1, charVocabSize);

  const flat = [];

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const value =
        boardTensor?.[r]?.[c] ?? 0;

      flat.push(value / divisor);
    }
  }

  return flat;
}

function normalizeRow(row) {
  return row / Math.max(1, ROWS - 1);
}

function normalizeCol(col) {
  return col / Math.max(1, COLS - 1);
}

function normalizeLength(length) {
  return length / Math.max(1, MAX_WORD_LENGTH);
}

function normalizeScore(score) {
  // move.score 縺ｯ縺縺・◆縺・0縲・20 遞句ｺｦ諠ｳ螳・
  // 螟ｧ縺阪☆縺弱ｋ蝣ｴ蜷医ｂ證ｴ繧後↑縺・ｈ縺・↓蝨ｧ邵ｮ
  return Math.tanh((score || 0) / 100);
}

function normalizeBranchCount(branchCount) {
  // F縺ゅｊ縺縺ｨ蛟呵｣懈焚縺悟､ｧ縺阪￥縺ｪ繧九・縺ｧ log 豁｣隕丞喧
  return Math.log1p(branchCount || 0) / Math.log1p(100);
}

function normalizeTargetWildcards(targetWildcards) {
  return (targetWildcards || 0) / 10;
}

/**
 * moveExample -> feature vector
 *
 * move-example 縺ｯ dataset.js 縺御ｽ懊▲縺・陦・
 * {
 *   boardTensor,
 *   move: {
 *     direction,
 *     start,
 *     end,
 *     length,
 *     wildcardCount,
 *     score
 *   },
 *   branchCount,
 *   targetWildcards
 * }
 */
function exampleToFeatures(example, metadata = {}) {
  const charVocabSize =
    metadata.charVocabSize ||
    metadata.vocabSize ||
    100;

  const boardFeatures =
    flattenBoardTensor(
      example.boardTensor,
      charVocabSize
    );

  const move = example.move || {};

  const start = move.start || [0, 0];
  const end = move.end || [0, 0];

  const direction = move.direction || "unknown";

  const directionH =
    direction === "H" ? 1 : 0;

  const directionV =
    direction === "V" ? 1 : 0;

  const extra = [
    normalizeRow(start[0] || 0),
    normalizeCol(start[1] || 0),
    normalizeRow(end[0] || 0),
    normalizeCol(end[1] || 0),
    normalizeLength(move.length || 0),
    directionH,
    directionV,
    normalizeLength(move.wildcardCount || 0),
    normalizeScore(move.score || 0),
    normalizeBranchCount(example.branchCount || 0),
    normalizeTargetWildcards(example.targetWildcards || 0),
  ];

  const features = [
    ...boardFeatures,
    ...extra,
  ];

  if (features.length !== INPUT_SIZE) {
    throw new Error(
      `Invalid feature length: ${features.length}, expected ${INPUT_SIZE}`
    );
  }

  return features;
}

function createMetadataFromSummary(summary) {
  const charMap = summary.charMap || {};

  const charVocabSize = Math.max(
    1,
    ...Object.values(charMap).filter(
      (v) => typeof v === "number"
    )
  );

  return {
    inputSize: INPUT_SIZE,
    boardSize: BOARD_SIZE,
    rows: ROWS,
    cols: COLS,
    maxWordLength: MAX_WORD_LENGTH,
    extraFeatures: EXTRA_FEATURES,
    charMap,
    charVocabSize,
    createdAt: new Date().toISOString(),
  };
}

module.exports = {
  tf,
  BOARD_SIZE,
  EXTRA_FEATURES,
  INPUT_SIZE,
  createPolicyModel,
  exampleToFeatures,
  createMetadataFromSummary,
};
