// src/neural/build-policy-dataset.js
//
// Replay Buffer 縺九ｉ Policy 蟄ｦ鄙堤畑繝・・繧ｿ繧ｻ繝・ヨ繧剃ｽ懊ｋ螳溯｡後ヵ繧｡繧､繝ｫ
//
// 螳溯｡・
//   node src/neural/build-policy-dataset.js
//
// F謨ｰ繧堤ｵ槭ｋ蝣ｴ蜷・
//   node src/neural/build-policy-dataset.js 0
//   node src/neural/build-policy-dataset.js 3
//

const {
  buildDataset,
} = require("./dataset");

function parseTargetWildcards() {
  const arg = process.argv[2];

  if (arg === undefined) {
    return undefined;
  }

  const n = Number(arg);

  if (Number.isNaN(n)) {
    return undefined;
  }

  return n;
}

function main() {
  const targetWildcards =
    parseTargetWildcards();

  console.log(
    "Building policy dataset..."
  );

  if (targetWildcards !== undefined) {
    console.log(
      "Target wildcard league:",
      `f${targetWildcards}`
    );
  } else {
    console.log(
      "Target wildcard league: all"
    );
  }

  const result = buildDataset({
    targetWildcards,
    maxPathsPerWord: 80,
  });

  console.log("\n=== DATASET CREATED ===\n");

  console.log(
    "Policy examples:",
    result.summary.policyExamples
  );

  console.log(
    "Move examples:",
    result.summary.moveExamples
  );

  console.log(
    "Replay files:",
    result.summary.inputReplayFiles
  );

  console.log(
    "Skipped files:",
    result.summary.skippedFiles
  );

  console.log("\nOutput files:");
  console.log(" ", result.policyFile);
  console.log(" ", result.moveFile);
  console.log(" ", result.summaryFile);

  console.log("\nBy F:");
  console.log(
    JSON.stringify(
      result.summary.byF,
      null,
      2
    )
  );

  if (result.summary.skipped.length > 0) {
    console.log("\nSkipped:");
    console.log(
      JSON.stringify(
        result.summary.skipped.slice(0, 10),
        null,
        2
      )
    );
  }
}

main();
