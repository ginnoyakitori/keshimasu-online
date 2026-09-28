// src/selfplay/replay-buffer.js
//
// Replay Buffer
//
// self-play の結果を保存する
//
// 保存先:
// data/training/selfplay/f0/
// data/training/selfplay/f1/
// data/training/selfplay/f2/
// ...
//
// data/puzzles/f0/
// data/puzzles/f1/
// ...

const fs = require("fs");
const path = require("path");

function ensureDir(dir) {
  fs.mkdirSync(dir, {
    recursive: true,
  });
}

function getReplayDir(targetWildcards) {
  return path.join(
    __dirname,
    "../../data/training/selfplay",
    `f${targetWildcards}`
  );
}

function getBestDir(targetWildcards) {
  return path.join(
    __dirname,
    "../../data/puzzles",
    `f${targetWildcards}`
  );
}

function makeId(epoch) {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-");

  return `epoch-${String(epoch).padStart(6, "0")}-${timestamp}`;
}

function saveReplay({
  epoch,
  targetWildcards,
  puzzle,
  generatorResult,
  solverResult,
  score,
  words,
}) {
  const dir = getReplayDir(targetWildcards);
  ensureDir(dir);

  const id = makeId(epoch);

  const file = path.join(dir, `${id}.json`);

  const data = {
    id,
    createdAt: new Date().toISOString(),

    epoch,
    targetWildcards,
    actualWildcards:
      generatorResult.actualWildcards,

    score,

    board: puzzle,

    words,

    generator: {
      score: generatorResult.score,
      solutionMoves:
        generatorResult.solutionMoves,
      targetWildcards:
        generatorResult.targetWildcards,
      actualWildcards:
        generatorResult.actualWildcards,
    },

    solver: {
      solved: solverResult.solved,
      solvedMoves:
        solverResult.stats.solvedMoves,
      stats: solverResult.stats,
    },

    trainingTrace:
      solverResult.stats.path || [],
  };

  fs.writeFileSync(
    file,
    JSON.stringify(data, null, 2),
    "utf8"
  );

  return {
    id,
    file,
  };
}

function saveBestPuzzle({
  epoch,
  targetWildcards,
  puzzle,
  generatorResult,
  solverResult,
  score,
}) {
  const dir = getBestDir(targetWildcards);
  ensureDir(dir);

  const id = makeId(epoch);

  const file = path.join(
    dir,
    `best-${id}.json`
  );

  const data = {
    id,
    createdAt: new Date().toISOString(),

    epoch,
    targetWildcards,
    actualWildcards:
      generatorResult.actualWildcards,

    score,

    board: puzzle,

    generatorSolutionMoves:
      generatorResult.solutionMoves,

    solverSolutionMoves:
      solverResult.stats.solvedMoves,

    stats: solverResult.stats,
  };

  fs.writeFileSync(
    file,
    JSON.stringify(data, null, 2),
    "utf8"
  );

  return {
    id,
    file,
  };
}

function loadReplayFiles(targetWildcards) {
  const dir = getReplayDir(targetWildcards);

  if (!fs.existsSync(dir)) {
    return [];
  }

  return fs
    .readdirSync(dir)
    .filter((file) => file.endsWith(".json"))
    .map((file) => path.join(dir, file));
}

function loadReplayData(targetWildcards) {
  const files = loadReplayFiles(targetWildcards);

  return files.map((file) => {
    const json = fs.readFileSync(file, "utf8");
    return JSON.parse(json);
  });
}

module.exports = {
  ensureDir,
  getReplayDir,
  getBestDir,
  saveReplay,
  saveBestPuzzle,
  loadReplayFiles,
  loadReplayData,
};