// src/neural/inference.js
//
// Policy inference
//
// Native addon 不要版:
// - @tensorflow/tfjs を使う
// - model.json + weights.bin を自前ロード
//

const fs = require("fs");
const path = require("path");

const {
  tf,
  exampleToFeatures,
} = require("./policy-model");

const {
  WILDCARD_CHAR,
} = require("../core/puzzle-config");

let cachedModel = null;
let cachedMetadata = null;

function modelDir() {
  return path.join(
    __dirname,
    "../../data/models/policy"
  );
}

function hasPolicyModel() {
  const dir = modelDir();

  return (
    fs.existsSync(path.join(dir, "model.json")) &&
    fs.existsSync(path.join(dir, "weights.bin")) &&
    fs.existsSync(path.join(dir, "metadata.json"))
  );
}

async function loadPolicyModel() {
  if (cachedModel && cachedMetadata) {
    return {
      model: cachedModel,
      metadata: cachedMetadata,
    };
  }

  const dir = modelDir();

  const modelPath = path.join(dir, "model.json");
  const weightsPath = path.join(dir, "weights.bin");
  const metadataPath = path.join(dir, "metadata.json");

  if (!fs.existsSync(modelPath)) {
    throw new Error(
      `Policy model not found: ${modelPath}. Run node src/neural/train-policy.js first.`
    );
  }

  if (!fs.existsSync(weightsPath)) {
    throw new Error(
      `Policy weights not found: ${weightsPath}. Run training first.`
    );
  }

  if (!fs.existsSync(metadataPath)) {
    throw new Error(
      `Policy metadata not found: ${metadataPath}. Run training first.`
    );
  }

  const modelJson = JSON.parse(
    fs.readFileSync(modelPath, "utf8")
  );

  const weightData = fs.readFileSync(weightsPath);

  const weightSpecs =
    modelJson.weightsManifest?.[0]?.weights || [];

  const arrayBuffer = weightData.buffer.slice(
    weightData.byteOffset,
    weightData.byteOffset + weightData.byteLength
  );

  cachedModel = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs,
      weightData: arrayBuffer,
    })
  );

  cachedMetadata = JSON.parse(
    fs.readFileSync(metadataPath, "utf8")
  );

  return {
    model: cachedModel,
    metadata: cachedMetadata,
  };
}

function boardToTensorWithCharMap(board, charMap) {
  return board.map((row) =>
    row.map((cell) => {
      if (cell === null) {
        return 0;
      }

      return charMap[cell] || 0;
    })
  );
}

function getMoveDirection(move) {
  const path = move.path;

  if (!path || path.length < 2) {
    return "unknown";
  }

  const [r0, c0] = path[0];
  const [r1, c1] = path[1];

  if (r1 === r0 && c1 === c0 + 1) {
    return "H";
  }

  if (r1 === r0 + 1 && c1 === c0) {
    return "V";
  }

  return "invalid";
}

function readPath(board, path) {
  return path
    .map(([r, c]) => board[r][c] ?? "")
    .join("");
}

function moveToFeature(move, board) {
  const path = move.path;

  const start = path[0];
  const end = path[path.length - 1];

  const boardText = readPath(board, path);

  const wildcardCount = [...boardText].filter(
    (ch) => ch === WILDCARD_CHAR
  ).length;

  return {
    word: move.word,
    path: move.path,
    direction: getMoveDirection(move),
    start,
    end,
    length: path.length,
    boardText,
    wildcardCount,
    score: move.score ?? 0,
  };
}

function makeExampleForMove(
  board,
  move,
  metadata,
  context = {}
) {
  const boardTensor = boardToTensorWithCharMap(
    board,
    metadata.charMap || {}
  );

  return {
    boardTensor,
    move: moveToFeature(move, board),
    branchCount: context.branchCount ?? 0,
    targetWildcards: context.targetWildcards ?? 0,
  };
}

async function scoreMove(
  board,
  move,
  context = {}
) {
  const {
    model,
    metadata,
  } = await loadPolicyModel();

  const example = makeExampleForMove(
    board,
    move,
    metadata,
    context
  );

  const features = exampleToFeatures(
    example,
    metadata
  );

  const input = tf.tensor2d([features]);

  const prediction = model.predict(input);

  const value = (await prediction.data())[0];

  input.dispose();
  prediction.dispose();

  return value;
}

async function scoreMoves(
  board,
  moves,
  context = {}
) {
  if (!moves || moves.length === 0) {
    return [];
  }

  const {
    model,
    metadata,
  } = await loadPolicyModel();

  const branchCount =
    context.branchCount ?? moves.length;

  const examples = moves.map((move) =>
    makeExampleForMove(
      board,
      move,
      metadata,
      {
        ...context,
        branchCount,
      }
    )
  );

  const features = examples.map((ex) =>
    exampleToFeatures(ex, metadata)
  );

  const input = tf.tensor2d(features);

  const prediction = model.predict(input);

  const values = Array.from(
    await prediction.data()
  );

  input.dispose();
  prediction.dispose();

  return moves.map((move, index) => ({
    ...move,
    policyScore: values[index],
  }));
}

async function orderMovesByPolicy(
  board,
  moves,
  context = {}
) {
  const scored = await scoreMoves(
    board,
    moves,
    context
  );

  return scored.sort((a, b) => {
    if (b.policyScore !== a.policyScore) {
      return b.policyScore - a.policyScore;
    }

    return (b.score || 0) - (a.score || 0);
  });
}

module.exports = {
  hasPolicyModel,
  loadPolicyModel,
  boardToTensorWithCharMap,
  getMoveDirection,
  readPath,
  moveToFeature,
  makeExampleForMove,
  scoreMove,
  scoreMoves,
  orderMovesByPolicy,
};