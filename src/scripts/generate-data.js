// src/scripts/generate-data.js
//
// Self-play データ大量生成スクリプト
//
// 目的:
// - F数ごとに replay data を増やす
// - data/training/selfplay/f0/
// - data/training/selfplay/f1/
// - data/training/selfplay/f2/
// - data/training/selfplay/f3/
// に保存する
//
// 実行:
//   node src/scripts/generate-data.js
//   node src/scripts/generate-data.js --mode focus
//   node src/scripts/generate-data.js --mode f3
//   node src/scripts/generate-data.js --mode f1
//   node src/scripts/generate-data.js --mode f2
//   node src/scripts/generate-data.js --mode quick
//

const {
  selfPlay,
} = require("../selfplay");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    mode: "focus",
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--mode" && args[i + 1]) {
      out.mode = String(args[i + 1]);
      i++;
      continue;
    }

    if (arg.startsWith("--mode=")) {
      out.mode = arg.slice("--mode=".length);
      continue;
    }
  }

  return out;
}

async function runLeague({
  targetWildcards,
  epochs,
  generatorIterations,
  timeLimitMs,
  solverMaxNodes,
  solverMaxDepth,
  solverMaxPathsPerWord,
}) {
  console.log("\n====================================");
  console.log(`START LEAGUE F=${targetWildcards}`);
  console.log("====================================\n");

  console.log("Config:", {
    targetWildcards,
    epochs,
    generatorIterations,
    timeLimitMs,
    solverMaxNodes,
    solverMaxDepth,
    solverMaxPathsPerWord,
  });

  const result = await selfPlay(epochs, {
    targetWildcards,
    generatorIterations,
    timeLimitMs,
    solverMaxNodes,
    solverMaxDepth,
    solverMaxPathsPerWord,
  });

  console.log("\n====================================");
  console.log(`DONE LEAGUE F=${targetWildcards}`);
  console.log("====================================\n");

  console.log("Best score:", result.bestScore);

  if (result.best) {
    console.log("Best F:", result.best.targetWildcards);
  }

  return result;
}

function getJobs(mode) {
  //
  // 現在のクリーンデータ状況:
  //   f0: 多め
  //   f1: 不足
  //   f2: 中程度
  //   f3: かなり不足
  //
  // そのため default は focus にして、
  // f1 / f2 / f3 を重点的に増やす。
  //

  if (mode === "quick") {
    return [
      {
        targetWildcards: 1,
        epochs: 5,
        generatorIterations: 30,
        timeLimitMs: 8000,
        solverMaxNodes: 50000,
        solverMaxDepth: 90,
        solverMaxPathsPerWord: 40,
      },
      {
        targetWildcards: 2,
        epochs: 5,
        generatorIterations: 40,
        timeLimitMs: 10000,
        solverMaxNodes: 80000,
        solverMaxDepth: 100,
        solverMaxPathsPerWord: 50,
      },
      {
        targetWildcards: 3,
        epochs: 3,
        generatorIterations: 40,
        timeLimitMs: 12000,
        solverMaxNodes: 120000,
        solverMaxDepth: 110,
        solverMaxPathsPerWord: 60,
      },
    ];
  }

  if (mode === "f1") {
    return [
      {
        targetWildcards: 1,
        epochs: 50,
        generatorIterations: 50,
        timeLimitMs: 14000,
        solverMaxNodes: 70000,
        solverMaxDepth: 100,
        solverMaxPathsPerWord: 50,
      },
    ];
  }

  if (mode === "f2") {
    return [
      {
        targetWildcards: 2,
        epochs: 80,
        generatorIterations: 60,
        timeLimitMs: 16000,
        solverMaxNodes: 100000,
        solverMaxDepth: 110,
        solverMaxPathsPerWord: 60,
      },
    ];
  }

  if (mode === "f3") {
    return [
      {
        targetWildcards: 3,
        epochs: 40,
        generatorIterations: 70,
        timeLimitMs: 22000,
        solverMaxNodes: 160000,
        solverMaxDepth: 120,
        solverMaxPathsPerWord: 70,
      },
    ];
  }

  if (mode === "balanced") {
    return [
      {
        targetWildcards: 0,
        epochs: 20,
        generatorIterations: 30,
        timeLimitMs: 10000,
        solverMaxNodes: 30000,
        solverMaxDepth: 80,
        solverMaxPathsPerWord: 30,
      },
      {
        targetWildcards: 1,
        epochs: 50,
        generatorIterations: 45,
        timeLimitMs: 13000,
        solverMaxNodes: 60000,
        solverMaxDepth: 95,
        solverMaxPathsPerWord: 45,
      },
      {
        targetWildcards: 2,
        epochs: 60,
        generatorIterations: 55,
        timeLimitMs: 16000,
        solverMaxNodes: 90000,
        solverMaxDepth: 105,
        solverMaxPathsPerWord: 55,
      },
      {
        targetWildcards: 3,
        epochs: 30,
        generatorIterations: 65,
        timeLimitMs: 22000,
        solverMaxNodes: 150000,
        solverMaxDepth: 120,
        solverMaxPathsPerWord: 70,
      },
    ];
  }

  //
  // default / focus:
  // f1, f2, f3 を重点的に増やす。
  //
  return [
    {
      targetWildcards: 1,
      epochs: 60,
      generatorIterations: 50,
      timeLimitMs: 14000,
      solverMaxNodes: 70000,
      solverMaxDepth: 100,
      solverMaxPathsPerWord: 50,
    },
    {
      targetWildcards: 2,
      epochs: 80,
      generatorIterations: 60,
      timeLimitMs: 16000,
      solverMaxNodes: 100000,
      solverMaxDepth: 110,
      solverMaxPathsPerWord: 60,
    },
    {
      targetWildcards: 3,
      epochs: 40,
      generatorIterations: 70,
      timeLimitMs: 22000,
      solverMaxNodes: 160000,
      solverMaxDepth: 120,
      solverMaxPathsPerWord: 70,
    },
  ];
}

async function main() {
  const args = parseArgs();

  console.log("\n=== DATA GENERATION START ===\n");
  console.log("Mode:", args.mode);

  const jobs = getJobs(args.mode);

  const summaries = [];

  for (const job of jobs) {
    const result = await runLeague(job);

    summaries.push({
      targetWildcards: job.targetWildcards,
      epochs: job.epochs,
      bestScore: result.bestScore,
      leagueSummary: result.leagueSummary,
    });
  }

  console.log("\n=== DATA GENERATION DONE ===\n");

  console.log(
    JSON.stringify(
      summaries,
      null,
      2
    )
  );

  console.log("\nNext step:");
  console.log("python python/apps/clean_duplicate_word_replays.py --archive _archive/duplicate_word_replays");
  console.log("python python/apps/build_numpy_dataset.py");
  console.log("python python/apps/train_policy.py --epochs 20 --early-stop 6");
  console.log("python python/apps/solver_policy.py --benchmark --league f2 --limit 50 --policy-min-branch 20");
}

main().catch((error) => {
  console.error("\nDATA GENERATION FAILED");
  console.error(error);
  process.exit(1);
});