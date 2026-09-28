// src/neural/onnx-policy.js
//
// ONNX Runtime policy inference for Keshimasu AI.
//
// Python side feature layout must match:
//   python/src/keshimasu_py/torch_policy.py
//
// Feature layout:
//   00-39: board flattened / char_vocab_size
//   40   : word_id / (num_words - 1)
//   41   : row / 7
//   42   : col / 4
//   43   : direction raw H=0 V=1
//   44   : is_horizontal
//   45   : is_vertical
//   46   : word_length / 5
//   47   : target_f / 3
//   48   : depth / 120
//   49   : log1p(branch_count) / log1p(256)
//   50   : start_index / 39
//   51   : end_row / 7
//   52   : end_col / 4
//   53   : filled_ratio
//
// Move format:
//   [word_id, row, col, direction]
//   direction: H=0, V=1

const fs = require("node:fs");
const path = require("node:path");

const ort = require("onnxruntime-node");


const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

const DEFAULT_MODEL_DIR = path.join(
  PROJECT_ROOT,
  "data",
  "models",
  "torch_policy"
);

const DEFAULT_ONNX_PATH = path.join(
  DEFAULT_MODEL_DIR,
  "policy_model.onnx"
);

const DEFAULT_ONNX_METADATA_PATH = path.join(
  DEFAULT_MODEL_DIR,
  "onnx_metadata.json"
);

const DEFAULT_METADATA_PATH = path.join(
  DEFAULT_MODEL_DIR,
  "metadata.json"
);

const ROWS = 8;
const COLS = 5;
const FEATURE_DIM = 54;
const DIR_H = 0;
const DIR_V = 1;


function fileExists(filePath) {
  try {
    return fs.existsSync(filePath);
  } catch {
    return false;
  }
}


function loadJsonIfExists(filePath) {
  if (!fileExists(filePath)) {
    return {};
  }

  const text = fs.readFileSync(filePath, "utf8");
  return JSON.parse(text);
}


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}


function normalizeCell(cell) {
  if (cell === null || cell === undefined) {
    return null;
  }

  if (cell === "" || cell === "・") {
    return null;
  }

  if (cell === "Ｆ") {
    return "F";
  }

  return String(cell);
}


function isNumberLike(value) {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}


function getCharId(cell, charToId) {
  const normalized = normalizeCell(cell);

  if (normalized === null) {
    return 0;
  }

  if (isNumberLike(cell)) {
    return Number(cell);
  }

  if (
    charToId &&
    Object.prototype.hasOwnProperty.call(charToId, normalized)
  ) {
    return Number(charToId[normalized]);
  }

  throw new Error(
    `Unknown board character: ${JSON.stringify(normalized)}`
  );
}


function encodeBoard(board, charToId) {
  if (!Array.isArray(board) || board.length !== ROWS) {
    throw new Error(
      `board must be ${ROWS} rows`
    );
  }

  const out = new Int16Array(ROWS * COLS);

  for (let r = 0; r < ROWS; r++) {
    if (!Array.isArray(board[r]) || board[r].length !== COLS) {
      throw new Error(
        `board row ${r} must have ${COLS} cols`
      );
    }

    for (let c = 0; c < COLS; c++) {
      out[r * COLS + c] = getCharId(
        board[r][c],
        charToId
      );
    }
  }

  return out;
}


function flattenEncodedBoard(encodedBoard) {
  if (encodedBoard instanceof Int16Array) {
    if (encodedBoard.length !== ROWS * COLS) {
      throw new Error(
        `encoded board length must be ${ROWS * COLS}`
      );
    }

    return encodedBoard;
  }

  if (Array.isArray(encodedBoard)) {
    // Already flattened numeric array.
    if (
      encodedBoard.length === ROWS * COLS &&
      encodedBoard.every((v) => isNumberLike(v))
    ) {
      return Int16Array.from(encodedBoard);
    }

    // 2D numeric board.
    if (
      encodedBoard.length === ROWS &&
      Array.isArray(encodedBoard[0])
    ) {
      const out = new Int16Array(ROWS * COLS);

      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          out[r * COLS + c] = Number(encodedBoard[r][c] || 0);
        }
      }

      return out;
    }
  }

  throw new Error("Unsupported encoded board format.");
}


function normalizeMove(move) {
  if (Array.isArray(move)) {
    if (move.length !== 4) {
      throw new Error(
        `encoded move array must have length 4: ${JSON.stringify(move)}`
      );
    }

    return [
      Number(move[0]),
      Number(move[1]),
      Number(move[2]),
      Number(move[3]),
    ];
  }

  if (move && typeof move === "object") {
    if (
      Object.prototype.hasOwnProperty.call(move, "word_id") ||
      Object.prototype.hasOwnProperty.call(move, "wordId")
    ) {
      return [
        Number(move.word_id ?? move.wordId),
        Number(move.row),
        Number(move.col),
        normalizeDirection(move.direction),
      ];
    }
  }

  throw new Error(
    `Unsupported move format: ${JSON.stringify(move)}`
  );
}


function normalizeDirection(direction) {
  if (direction === "H" || direction === "h") {
    return DIR_H;
  }

  if (direction === "V" || direction === "v") {
    return DIR_V;
  }

  return Number(direction);
}


function normalizeMoves(moves) {
  if (!Array.isArray(moves)) {
    throw new Error("moves must be an array");
  }

  return moves.map(normalizeMove);
}


function inferCharVocabSize(metadata, onnxMetadata) {
  const direct =
    onnxMetadata.charVocabSize ??
    onnxMetadata.char_vocab_size ??
    metadata.charVocabSize ??
    metadata.char_vocab_size;

  if (direct !== undefined) {
    return Number(direct);
  }

  const charToId =
    metadata.charToId ||
    onnxMetadata.charToId ||
    {};

  return Object.keys(charToId).length + 1;
}


function inferNumWords(metadata, onnxMetadata) {
  const direct =
    onnxMetadata.numWords ??
    onnxMetadata.num_words ??
    metadata.numWords ??
    metadata.num_words;

  if (direct !== undefined) {
    return Number(direct);
  }

  const words =
    metadata.words ||
    onnxMetadata.words ||
    [];

  return Array.isArray(words) ? words.length : 0;
}


function inferInputSize(metadata, onnxMetadata) {
  const direct =
    onnxMetadata.inputSize ??
    onnxMetadata.input_size ??
    onnxMetadata.featureDim ??
    onnxMetadata.feature_dim ??
    metadata.inputSize ??
    metadata.input_size;

  if (direct !== undefined) {
    return Number(direct);
  }

  const summary = metadata.summary || {};
  const featureFormat = summary.featureFormat || {};

  const fromSummary =
    featureFormat.inputSize ??
    featureFormat.input_size ??
    featureFormat.featureDim ??
    featureFormat.feature_dim;

  if (fromSummary !== undefined) {
    return Number(fromSummary);
  }

  return FEATURE_DIM;
}


function getWordLengths(metadata, onnxMetadata) {
  const wordLengths =
    metadata.wordLengths ||
    onnxMetadata.wordLengths ||
    metadata.word_lengths ||
    onnxMetadata.word_lengths;

  if (Array.isArray(wordLengths)) {
    return wordLengths.map((v) => Number(v));
  }

  const words =
    metadata.words ||
    onnxMetadata.words ||
    [];

  if (Array.isArray(words) && words.length > 0) {
    return words.map((word) => Array.from(String(word)).length);
  }

  return [];
}


function makeFeatureRow({
  encodedBoard,
  move,
  wordLengths,
  charVocabSize,
  numWords,
  targetF,
  depth,
  branchCount,
}) {
  const feature = new Float32Array(FEATURE_DIM);

  const denomChar = Math.max(1, charVocabSize);

  let filled = 0;

  for (let i = 0; i < ROWS * COLS; i++) {
    const value = Number(encodedBoard[i] || 0);

    if (value !== 0) {
      filled += 1;
    }

    feature[i] = value / denomChar;
  }

  const wordId = Number(move[0]);
  const row = Number(move[1]);
  const col = Number(move[2]);
  const direction = Number(move[3]);

  const wordLength = Number(
    wordLengths[wordId] !== undefined
      ? wordLengths[wordId]
      : 1
  );

  const lengthMinusOne = Math.max(0, wordLength - 1);

  const endRow = row + lengthMinusOne * direction;
  const endCol = col + lengthMinusOne * (1 - direction);

  const startIndex = row * COLS + col;

  feature[40] = wordId / Math.max(1, numWords - 1);
  feature[41] = row / Math.max(1, ROWS - 1);
  feature[42] = col / Math.max(1, COLS - 1);
  feature[43] = direction;
  feature[44] = direction === DIR_H ? 1 : 0;
  feature[45] = direction === DIR_V ? 1 : 0;
  feature[46] = wordLength / 5.0;
  feature[47] = Number(targetF || 0) / 3.0;
  feature[48] = Number(depth || 0) / 120.0;
  feature[49] = Math.log1p(Number(branchCount || 0)) / Math.log1p(256);
  feature[50] = startIndex / Math.max(1, ROWS * COLS - 1);
  feature[51] = endRow / Math.max(1, ROWS - 1);
  feature[52] = endCol / Math.max(1, COLS - 1);
  feature[53] = filled / (ROWS * COLS);

  return feature;
}


function makeFeatureMatrix({
  board,
  encodedBoard,
  moves,
  charToId,
  wordLengths,
  charVocabSize,
  numWords,
  targetF = 0,
  depth = 0,
  branchCount = null,
}) {
  const normalizedMoves = normalizeMoves(moves);

  const boardFlat = encodedBoard
    ? flattenEncodedBoard(encodedBoard)
    : encodeBoard(board, charToId);

  const n = normalizedMoves.length;
  const out = new Float32Array(n * FEATURE_DIM);

  const actualBranchCount =
    branchCount === null || branchCount === undefined
      ? n
      : Number(branchCount);

  for (let i = 0; i < n; i++) {
    const row = makeFeatureRow({
      encodedBoard: boardFlat,
      move: normalizedMoves[i],
      wordLengths,
      charVocabSize,
      numWords,
      targetF,
      depth,
      branchCount: actualBranchCount,
    });

    out.set(row, i * FEATURE_DIM);
  }

  return {
    data: out,
    shape: [n, FEATURE_DIM],
    moves: normalizedMoves,
  };
}


class OnnxPolicy {
  constructor(options = {}) {
    this.modelDir = path.resolve(
      options.modelDir || DEFAULT_MODEL_DIR
    );

    this.onnxPath = path.resolve(
      options.onnxPath || path.join(this.modelDir, "policy_model.onnx")
    );

    this.metadataPath = path.resolve(
      options.metadataPath || path.join(this.modelDir, "metadata.json")
    );

    this.onnxMetadataPath = path.resolve(
      options.onnxMetadataPath || path.join(this.modelDir, "onnx_metadata.json")
    );

    this.session = null;
    this.metadata = {};
    this.onnxMetadata = {};

    this.inputSize = FEATURE_DIM;
    this.featureDim = FEATURE_DIM;
    this.charVocabSize = 0;
    this.numWords = 0;

    this.charToId = {};
    this.idToChar = {};
    this.words = [];
    this.wordLengths = [];
    this.wildId = 0;
    this.emptyId = 0;
  }

  async load() {
    if (!fileExists(this.onnxPath)) {
      throw new Error(
        `ONNX model not found: ${this.onnxPath}`
      );
    }

    this.metadata = loadJsonIfExists(this.metadataPath);
    this.onnxMetadata = loadJsonIfExists(this.onnxMetadataPath);

    this.inputSize = inferInputSize(
      this.metadata,
      this.onnxMetadata
    );

    this.featureDim =
      Number(this.onnxMetadata.featureDim || this.inputSize);

    if (this.inputSize !== FEATURE_DIM) {
      throw new Error(
        `Unsupported input size: ${this.inputSize}, expected ${FEATURE_DIM}`
      );
    }

    this.charVocabSize = inferCharVocabSize(
      this.metadata,
      this.onnxMetadata
    );

    this.numWords = inferNumWords(
      this.metadata,
      this.onnxMetadata
    );

    this.charToId =
      this.metadata.charToId ||
      this.onnxMetadata.charToId ||
      {};

    this.idToChar =
      this.metadata.idToChar ||
      this.onnxMetadata.idToChar ||
      {};

    this.words =
      this.metadata.words ||
      this.onnxMetadata.words ||
      [];

    this.wordLengths = getWordLengths(
      this.metadata,
      this.onnxMetadata
    );

    this.wildId = Number(
      this.metadata.wildId ??
      this.onnxMetadata.wildId ??
      0
    );

    this.emptyId = Number(
      this.metadata.emptyId ??
      this.onnxMetadata.emptyId ??
      0
    );

    this.session = await ort.InferenceSession.create(
      this.onnxPath,
      {
        executionProviders: ["cpu"],
      }
    );

    return this;
  }

  ensureLoaded() {
    if (!this.session) {
      throw new Error(
        "OnnxPolicy is not loaded. Call await policy.load() first."
      );
    }
  }

  buildFeatures({
    board = null,
    encodedBoard = null,
    moves,
    targetF = 0,
    depth = 0,
    branchCount = null,
  }) {
    this.ensureLoaded();

    return makeFeatureMatrix({
      board,
      encodedBoard,
      moves,
      charToId: this.charToId,
      wordLengths: this.wordLengths,
      charVocabSize: this.charVocabSize,
      numWords: this.numWords,
      targetF,
      depth,
      branchCount,
    });
  }

  async scoreEncodedMoves({
    board = null,
    encodedBoard = null,
    moves,
    targetF = 0,
    depth = 0,
    branchCount = null,
  }) {
    this.ensureLoaded();

    const featureMatrix = this.buildFeatures({
      board,
      encodedBoard,
      moves,
      targetF,
      depth,
      branchCount,
    });

    const tensor = new ort.Tensor(
      "float32",
      featureMatrix.data,
      featureMatrix.shape
    );

    const feeds = {
      x: tensor,
    };

    const results = await this.session.run(feeds);

    const outputName = Object.keys(results)[0];
    const logits = results[outputName].data;

    const scores = Array.from(logits, (v) => sigmoid(Number(v)));

    return {
      moves: featureMatrix.moves,
      logits: Array.from(logits, Number),
      scores,
    };
  }

  async orderEncodedMoves({
    board = null,
    encodedBoard = null,
    moves,
    heuristicScores = null,
    targetF = 0,
    depth = 0,
    branchCount = null,
    policyWeight = 1.0,
    heuristicWeight = 0.0,
  }) {
    const result = await this.scoreEncodedMoves({
      board,
      encodedBoard,
      moves,
      targetF,
      depth,
      branchCount,
    });

    const hScores = Array.isArray(heuristicScores)
      ? heuristicScores.map(Number)
      : null;

    let maxH = 1;

    if (hScores && hScores.length > 0) {
      maxH = Math.max(
        1,
        ...hScores.map((v) => Math.abs(v))
      );
    }

    const rows = result.moves.map((move, i) => {
      const policyScore = result.scores[i];

      const heuristicScore =
        hScores && hScores[i] !== undefined
          ? hScores[i]
          : 0;

      const heuristicNorm = heuristicScore / maxH;

      const combinedScore =
        Number(policyWeight) * policyScore +
        Number(heuristicWeight) * heuristicNorm;

      return {
        move,
        policyScore,
        heuristicScore,
        combinedScore,
        logit: result.logits[i],
      };
    });

    rows.sort((a, b) => {
      if (b.combinedScore !== a.combinedScore) {
        return b.combinedScore - a.combinedScore;
      }

      return b.policyScore - a.policyScore;
    });

    return {
      moves: rows.map((row) => row.move),
      policyScores: rows.map((row) => row.policyScore),
      heuristicScores: rows.map((row) => row.heuristicScore),
      combinedScores: rows.map((row) => row.combinedScore),
      rows,
    };
  }

  getWordText(wordId) {
    return this.words[wordId] || String(wordId);
  }
}


async function loadDefaultPolicy(options = {}) {
  const policy = new OnnxPolicy(options);
  await policy.load();
  return policy;
}


module.exports = {
  OnnxPolicy,
  loadDefaultPolicy,

  makeFeatureMatrix,
  makeFeatureRow,

  encodeBoard,
  normalizeCell,
  normalizeMove,
  normalizeMoves,
  sigmoid,

  FEATURE_DIM,
  ROWS,
  COLS,
  DIR_H,
  DIR_V,
};


// ---------------------------------------------------------------------
// CLI smoke test
// ---------------------------------------------------------------------

if (require.main === module) {
  (async () => {
    const policy = await loadDefaultPolicy();

    console.log("=== ONNX POLICY LOADED ===");
    console.log("model:", policy.onnxPath);
    console.log("inputSize:", policy.inputSize);
    console.log("charVocabSize:", policy.charVocabSize);
    console.log("numWords:", policy.numWords);
    console.log("words:", policy.words.length);

    const encodedBoard = new Array(ROWS * COLS).fill(0);

    const moves = [
      [0, 7, 0, DIR_H],
      [1, 7, 0, DIR_H],
      [2, 3, 0, DIR_V],
    ];

    const result = await policy.scoreEncodedMoves({
      encodedBoard,
      moves,
      targetF: 0,
      depth: 0,
    });

    console.log("scores:", result.scores);
    console.log("logits:", result.logits);
  })().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}