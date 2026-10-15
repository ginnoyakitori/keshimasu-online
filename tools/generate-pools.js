"use strict";

/*
 * モード別・レベル別の問題プールを生成して data/puzzles/pools/ へ保存します。
 *
 * 流れ:
 *   1. 各レベルの候補を、必要数の数倍つくる（逆向き構築なので必ず全消しできる）
 *   2. ソルバーで1問ずつ難易度を測る（tools/solver.js）
 *   3. レベルが上がるほど難しくなるよう、目標難易度に近い問題を選ぶ（tools/balance.js）
 *   4. 想定解を再生して全消しできることを確認してから保存する
 *
 * 使い方:
 *   node tools/generate-pools.js                 全モード・全レベルを60問ずつ
 *   node tools/generate-pools.js --mode pokemon  1つのモードだけ
 *   node tools/generate-pools.js --count 100     1レベルあたりの問題数
 *   node tools/generate-pools.js --oversample 3  候補の倍率（既定 2.5）
 *   node tools/generate-pools.js --budget 100000 1問あたりの探索の上限（既定 100000）
 *
 * 難易度の重みは src/multiplayer/levels.js の DIFFICULTY_WEIGHTS で調整できます。
 * レベルを絞る（--levels）と、難易度の並びは全レベルをそろえたときだけ保証されます。
 */

const fs = require("node:fs");
const path = require("node:path");

const { LEVELS, MODES } = require("../src/multiplayer/levels");
const {
  generatePuzzle,
  replaySolution,
  puzzleId,
} = require("../src/multiplayer/puzzle-generator");
const { Solver } = require("./solver");
const {
  measurePuzzle,
  summarize,
  selectBalanced,
  checkMonotonic,
} = require("./balance");

const ROOT = path.resolve(__dirname, "..");
const WORDS_DIR = path.join(ROOT, "data", "words");
const POOLS_DIR = path.join(ROOT, "data", "puzzles", "pools");

function parseArguments(argv) {
  const options = {
    mode: "all",
    count: 60,
    oversample: 2.5,
    budget: 100000,
    levels: [1, LEVELS.length],
  };

  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];

    if (key === "--mode") {
      options.mode = value;
      index += 1;
    } else if (key === "--count") {
      options.count = Math.max(1, Number(value) || options.count);
      index += 1;
    } else if (key === "--oversample") {
      options.oversample = Math.max(1, Number(value) || options.oversample);
      index += 1;
    } else if (key === "--budget") {
      options.budget = Math.max(1000, Number(value) || options.budget);
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

function generateCandidates({ mode, cfg, words, count }) {
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
  const modes = options.mode === "all" ? Object.keys(MODES) : [options.mode];

  fs.mkdirSync(POOLS_DIR, { recursive: true });

  for (const mode of modes) {
    if (!MODES[mode]) {
      throw new Error(`Unknown mode: ${mode}`);
    }

    const words = loadWords(mode);
    const wordSet = new Set(words);
    const solver = new Solver({ words });
    const poolFile = path.join(POOLS_DIR, `${mode}.json`);

    // 一部のレベルだけ作り直せるよう、既存のプールを引き継ぐ
    let existing = {};

    if (fs.existsSync(poolFile)) {
      existing = JSON.parse(fs.readFileSync(poolFile, "utf8")).levels || {};
    }

    const measured = {};

    for (const cfg of LEVELS) {
      if (cfg.level < options.levels[0] || cfg.level > options.levels[1]) {
        continue;
      }

      const startedAt = Date.now();
      const candidates = generateCandidates({
        mode,
        cfg,
        words,
        count: Math.ceil(options.count * options.oversample),
      });

      const list = [];
      let dropped = 0;

      for (const candidate of candidates) {
        const result = measurePuzzle({
          solver,
          puzzle: candidate,
          cfg,
          nodeBudget: options.budget,
        });

        if (result) {
          list.push(result);
        } else {
          dropped += 1;
        }
      }

      measured[cfg.level] = list;

      const exhausted = list.filter((puzzle) => puzzle.measureExhausted).length;

      console.log(
        `[${mode}] Lv${cfg.level} ${cfg.cols}x${cfg.rows}: ` +
          `候補 ${candidates.length} 測定 ${list.length}` +
          (dropped ? ` 除外 ${dropped}` : "") +
          (exhausted ? ` 探索上限 ${exhausted}` : "") +
          ` (${Math.round((Date.now() - startedAt) / 1000)}秒)`
      );
    }

    const { selected, targets } = selectBalanced(measured, options.count);
    const levels = { ...existing };
    const stats = {};

    for (const cfg of LEVELS) {
      const puzzles = selected[cfg.level];

      if (!puzzles) {
        continue;
      }

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
      stats[String(cfg.level)] = { target: targets[cfg.level], ...summarize(puzzles) };

      const s = stats[String(cfg.level)];

      console.log(
        `[${mode}] Lv${cfg.level} 選択 ${s.count}問 ` +
          `難易度 目標 ${s.target} / 最小 ${s.min} 中央 ${s.median} 最大 ${s.max}`
      );
    }

    const problems = checkMonotonic(selected);

    if (problems.length > 0) {
      console.warn(`[${mode}] 難易度が上がっていないレベルがあります:`);

      for (const problem of problems) {
        console.warn(`  - ${problem}`);
      }
    } else if (Object.keys(selected).length > 1) {
      console.log(`[${mode}] 平均難易度は Lv が上がるたびに上がっています`);
    }

    fs.writeFileSync(
      poolFile,
      JSON.stringify({
        format: "keshimasu-duel-pool-v2",
        mode,
        generatedAt: new Date().toISOString(),
        difficultyWeights: require("../src/multiplayer/levels").DIFFICULTY_WEIGHTS,
        stats,
        levels,
      }),
      "utf8"
    );
  }
}

main();