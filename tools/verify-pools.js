"use strict";

/*
 * 保存済みの問題プールを、すべて想定解で再生して確認します。
 *
 *   npm run verify
 *
 * 確認すること:
 *   - 盤面の大きさがレベル設定と一致している
 *   - 想定解の単語がすべて辞書にあり、同じ単語を重複して使っていない
 *   - 想定解どおりに消すと、盤面がすべて空になる
 *   - 縦に消す手だけでは全消しできない（縦だけで解ける盤面が混ざっていない）
 */

const fs = require("node:fs");
const path = require("node:path");

const { LEVELS, MODES } = require("../src/multiplayer/levels");
const {
  replaySolution,
  canClearWithVerticalOnly,
  groupByLength,
  normalizeWords,
} = require("../src/multiplayer/puzzle-generator");

const ROOT = path.resolve(__dirname, "..");

let failed = 0;
let total = 0;

for (const [mode, info] of Object.entries(MODES)) {
  const poolFile = path.join(ROOT, "data", "puzzles", "pools", `${mode}.json`);

  if (!fs.existsSync(poolFile)) {
    console.log(`[${mode}] pool file not found (skipped)`);
    continue;
  }

  const words = new Set(
    JSON.parse(
      fs.readFileSync(path.join(ROOT, "data", "words", info.wordFile), "utf8")
    )
  );

  const { levels } = JSON.parse(fs.readFileSync(poolFile, "utf8"));

  for (const cfg of LEVELS) {
    const puzzles = levels[String(cfg.level)] || [];

    if (puzzles.length === 0) {
      console.log(`[${mode}] Lv${cfg.level}: NO PUZZLES`);
      failed += 1;
      continue;
    }

    for (const puzzle of puzzles) {
      total += 1;

      const sizeOk =
        puzzle.board.length === cfg.rows &&
        puzzle.board.every((row) => row.length === cfg.cols);

      const replay = sizeOk
        ? replaySolution({
            board: puzzle.board,
            solution: puzzle.solution,
            cfg,
            isWord: (word) => words.has(word),
          })
        : { ok: false, reason: "size mismatch" };

      if (!replay.ok) {
        failed += 1;
        console.log(`[${mode}] Lv${cfg.level} ${puzzle.id}: ${replay.reason || "not cleared"}`);
        continue;
      }

      const byLength = groupByLength(normalizeWords([...words], 2, cfg.maxWordLength));

      if (
        canClearWithVerticalOnly({
          board: puzzle.board,
          byLength,
          cfg,
          maxNodes: 30000,
        })
      ) {
        failed += 1;
        console.log(`[${mode}] Lv${cfg.level} ${puzzle.id}: vertical-only solvable`);
      }
    }
  }
}

console.log(`checked=${total} failed=${failed}`);
process.exit(failed === 0 ? 0 : 1);
