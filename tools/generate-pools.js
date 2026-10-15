"use strict";

/*
 * モード別・レベル別の問題プールを生成して data/puzzles/pools/ へ保存します。
 *
 * 使い方:
 *   node tools/generate-pools.js                 全モード・全レベルを60問ずつ
 *   node tools/generate-pools.js --mode pokemon  1つのモードだけ
 *   node tools/generate-pools.js --count 100     1レベルあたりの問題数
 *   node tools/generate-pools.js --levels 8-12   レベルを絞る
 *
 * 生成した盤面は、想定解を再生して全消しできることを確認してから保存します。
 */

const fs = require("node:fs");
const path = require("node:path");

const { LEVELS, MODES } = require("../src/multiplayer/levels");
const {
  generatePuzzle,
  replaySolution,
  puzzleId,
} = require("../src/multiplayer/puzzle-generator");

const ROOT = path.resolve(__dirname, "..");
const WORDS_DIR = path.join(ROOT, "data", "words");
const POOLS_DIR = path.join(ROOT, "data", "puzzles", "pools");

function parseArguments(argv) {
  const options = { mode: "all", count: 60, levels: [1, LEVELS.length] };

  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];

    if (key === "--mode") {
      options.mode = value;
      index += 1;
    } else if (key === "--count") {
      options.count = Math.max(1, Number(value) || options.count);
      index += 1;
    } else if (key === "--levels") {
      const [from, to] = String(value).split("-").map(Number);

      options.levels = [from, Number.isFinite(to) ? to : from];
      index += 1;
    }
  }

  return options;
}

function loadWords(mode) {
  const file = path.join(WORDS_DIR, MODES[mode].wordFile);

  return JSON.parse(fs.readFileSync(file, "utf8").replace(/^\uFEFF/, ""));
}

function generateLevel({ mode, cfg, words, count }) {
  const found = new Map();
  const startedAt = Date.now();
  const timeLimitMs = 120_000;
  let failures = 0;

  while (found.size < count && Date.now() - startedAt < timeLimitMs) {
    const puzzle = generatePuzzle({ cfg, words, maxAttempts: 600 });

    if (!puzzle) {
      failures += 1;

      if (failures > count * 5) {
        break;
      }

      continue;
    }

    const id = puzzleId(`${mode}-L${cfg.level}`, puzzle.board);

    if (!found.has(id)) {
      found.set(id, { id, ...puzzle });
    }
  }

  return [...found.values()];
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const modes =
    options.mode === "all" ? Object.keys(MODES) : [options.mode];

  fs.mkdirSync(POOLS_DIR, { recursive: true });

  for (const mode of modes) {
    if (!MODES[mode]) {
      throw new Error(`Unknown mode: ${mode}`);
    }

    const words = loadWords(mode);
    const wordSet = new Set(words);
    const poolFile = path.join(POOLS_DIR, `${mode}.json`);

    // 一部のレベルだけ作り直せるよう、既存のプールを引き継ぐ
    let existing = {};

    if (fs.existsSync(poolFile)) {
      existing = JSON.parse(fs.readFileSync(poolFile, "utf8")).levels || {};
    }

    const levels = { ...existing };

    for (const cfg of LEVELS) {
      if (cfg.level < options.levels[0] || cfg.level > options.levels[1]) {
        continue;
      }

      const puzzles = generateLevel({
        mode,
        cfg,
        words,
        count: options.count,
      });

      // 保存前の最終確認
      for (const puzzle of puzzles) {
        const replay = replaySolution({
          board: puzzle.board,
          solution: puzzle.solution,
          cfg,
          isWord: (word) => wordSet.has(word),
        });

        if (!replay.ok) {
          throw new Error(`verification failed: ${puzzle.id}`);
        }
      }

      levels[String(cfg.level)] = puzzles;

      console.log(
        `[${mode}] Lv${cfg.level} ` +
          `${cfg.cols}x${cfg.rows}: ${puzzles.length}/${options.count}`
      );
    }

    fs.writeFileSync(
      poolFile,
      JSON.stringify({
        format: "keshimasu-duel-pool-v1",
        mode,
        generatedAt: new Date().toISOString(),
        levels,
      }),
      "utf8"
    );
  }
}

main();
