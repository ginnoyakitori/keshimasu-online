// src/scripts/benchmark-onnx-policy.js
//
// Benchmark ONNX Policy Solver vs Normal Solver.
//
// Usage:
//   node src/scripts/benchmark-onnx-policy.js
//   node src/scripts/benchmark-onnx-policy.js --league f3 --limit 25
//   node src/scripts/benchmark-onnx-policy.js --league f2 --limit 50 --policy-min-branch 20
//   node src/scripts/benchmark-onnx-policy.js --league all --limit 100
//   node src/scripts/benchmark-onnx-policy.js --league f3 --policy-weight 0.25 --heuristic-weight 0.75 --policy-top-k 8
//

const fs = require("node:fs");
const path = require("node:path");

const {
  solve,
} = require("../solver/solver");

const {
  solvePolicy,
  resetDefaultPolicyCache,
} = require("../solver/solver-policy");

const {
  getPlayableWords,
} = require("../generator/words");


const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

const DEFAULT_REPLAY_ROOT = path.join(
  PROJECT_ROOT,
  "data",
  "training",
  "selfplay"
);


function parseArgs(argv) {
  const args = {
    league: "f3",
    limit: 25,
    root: DEFAULT_REPLAY_ROOT,

    maxNodes: 500000,
    maxDepth: 120,
    maxPathsPerWord: 80,

    policyMinBranch: 20,
    policyTopK: 8,
    policyWeight: 0.25,
    heuristicWeight: 0.75,

    noPolicy: false,
    verbose: false,
    json: false,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];

    function nextValue() {
      if (i + 1 >= argv.length) {
        throw new Error(`Missing value for ${arg}`);
      }

      i += 1;
      return argv[i];
    }

    if (arg === "--league") {
      args.league = nextValue();
    } else if (arg.startsWith("--league=")) {
      args.league = arg.slice("--league=".length);
    } else if (arg === "--limit") {
      args.limit = Number(nextValue());
    } else if (arg.startsWith("--limit=")) {
      args.limit = Number(arg.slice("--limit=".length));
    } else if (arg === "--root") {
      args.root = path.resolve(nextValue());
    } else if (arg.startsWith("--root=")) {
      args.root = path.resolve(arg.slice("--root=".length));
    } else if (arg === "--max-nodes") {
      args.maxNodes = Number(nextValue());
    } else if (arg.startsWith("--max-nodes=")) {
      args.maxNodes = Number(arg.slice("--max-nodes=".length));
    } else if (arg === "--max-depth") {
      args.maxDepth = Number(nextValue());
    } else if (arg.startsWith("--max-depth=")) {
      args.maxDepth = Number(arg.slice("--max-depth=".length));
    } else if (arg === "--max-paths-per-word") {
      args.maxPathsPerWord = Number(nextValue());
    } else if (arg.startsWith("--max-paths-per-word=")) {
      args.maxPathsPerWord = Number(arg.slice("--max-paths-per-word=".length));
    } else if (arg === "--policy-min-branch") {
      args.policyMinBranch = Number(nextValue());
    } else if (arg.startsWith("--policy-min-branch=")) {
      args.policyMinBranch = Number(arg.slice("--policy-min-branch=".length));
    } else if (arg === "--policy-top-k") {
      args.policyTopK = Number(nextValue());
    } else if (arg.startsWith("--policy-top-k=")) {
      args.policyTopK = Number(arg.slice("--policy-top-k=".length));
    } else if (arg === "--policy-weight") {
      args.policyWeight = Number(nextValue());
    } else if (arg.startsWith("--policy-weight=")) {
      args.policyWeight = Number(arg.slice("--policy-weight=".length));
    } else if (arg === "--heuristic-weight") {
      args.heuristicWeight = Number(nextValue());
    } else if (arg.startsWith("--heuristic-weight=")) {
      args.heuristicWeight = Number(arg.slice("--heuristic-weight=".length));
    } else if (arg === "--no-policy") {
      args.noPolicy = true;
    } else if (arg === "--verbose") {
      args.verbose = true;
    } else if (arg === "--json") {
      args.json = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelpAndExit();
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.limit) || args.limit < 0) {
    args.limit = 0;
  }

  return args;
}


function printHelpAndExit() {
  console.log(`
Benchmark ONNX Policy Solver.

Usage:
  node src/scripts/benchmark-onnx-policy.js
  node src/scripts/benchmark-onnx-policy.js --league f3 --limit 25
  node src/scripts/benchmark-onnx-policy.js --league f2 --limit 50 --policy-min-branch 20
  node src/scripts/benchmark-onnx-policy.js --league all --limit 100

Options:
  --league f0|f1|f2|f3|all
  --limit N
  --root PATH
  --max-nodes N
  --max-depth N
  --max-paths-per-word N
  --policy-min-branch N
  --policy-top-k N
  --policy-weight FLOAT
  --heuristic-weight FLOAT
  --no-policy
  --verbose
  --json
`);
  process.exit(0);
}


function readJson(file) {
  return JSON.parse(
    fs.readFileSync(file, "utf8")
  );
}


function isJsonFile(file) {
  return file.toLowerCase().endsWith(".json");
}


function listJsonFiles(dir) {
  if (!fs.existsSync(dir)) {
    return [];
  }

  const entries = fs.readdirSync(dir, {
    withFileTypes: true,
  });

  const files = [];

  for (const entry of entries) {
    const full = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...listJsonFiles(full));
    } else if (entry.isFile() && isJsonFile(entry.name)) {
      files.push(full);
    }
  }

  return files;
}


function listReplayFiles({ root, league, limit }) {
  let files = [];

  if (league === "all") {
    for (const f of ["f0", "f1", "f2", "f3"]) {
      const dir = path.join(root, f);
      files.push(...listJsonFiles(dir));
    }
  } else {
    const dir = path.join(root, league);
    files = listJsonFiles(dir);
  }

  files = files
    .filter((file) => {
      const base = path.basename(file);
      return base.startsWith("epoch-") && base.endsWith(".json");
    })
    .sort();

  if (limit && limit > 0) {
    files = files.slice(0, limit);
  }

  return files;
}


function replayTargetF(replay, file) {
  const candidates = [
    replay.targetWildcards,
    replay.targetF,
    replay.actualWildcards,
    replay.puzzle && replay.puzzle.targetWildcards,
    replay.puzzle && replay.puzzle.targetF,
    replay.config && replay.config.targetWildcards,
    replay.league && String(replay.league).replace(/^f/, ""),
  ];

  for (const value of candidates) {
    if (value !== undefined && value !== null && value !== "") {
      const n = Number(value);

      if (Number.isFinite(n)) {
        return n;
      }
    }
  }

  const parent = path.basename(path.dirname(file));
  const match = parent.match(/^f(\d+)$/);

  if (match) {
    return Number(match[1]);
  }

  return 0;
}


function replayBoard(replay) {
  const candidates = [
    replay.board,
    replay.initialBoard,
    replay.puzzle && replay.puzzle.board,
    replay.game && replay.game.board,
    replay.result && replay.result.board,
  ];

  for (const board of candidates) {
    if (
      Array.isArray(board) &&
      board.length > 0 &&
      Array.isArray(board[0])
    ) {
      return board;
    }
  }

  return null;
}


function replaySolvedMoves(replay) {
  const candidates = [
    replay.solvedMoves,
    replay.solutionMoves,
    replay.solverSolutionMoves,
    replay.solver && replay.solver.solvedMoves,
    replay.stats && replay.stats.solvedMoves,
    replay.result && replay.result.solvedMoves,
  ];

  for (const moves of candidates) {
    if (Array.isArray(moves) && moves.length > 0) {
      return moves;
    }
  }

  return [];
}


function makeWords(replay) {
  if (Array.isArray(replay.words) && replay.words.length > 0) {
    return replay.words.map(String);
  }

  if (
    replay.puzzle &&
    Array.isArray(replay.puzzle.words) &&
    replay.puzzle.words.length > 0
  ) {
    return replay.puzzle.words.map(String);
  }

  return getPlayableWords({
    minLen: 2,
    maxLen: 5,
  });
}


function makeSolveOptions({
  replay,
  targetF,
  args,
}) {
  return {
    words: makeWords(replay),

    maxNodes: args.maxNodes,
    maxDepth: args.maxDepth,
    maxPathsPerWord: args.maxPathsPerWord,

    targetWildcards: targetF,

    useWordOnce: true,
    useVisited: true,

    usePolicy: !args.noPolicy,
    fallbackToHeuristic: true,

    policyMinBranch: args.policyMinBranch,
    policyTopK: args.policyTopK,
    policyWeight: args.policyWeight,
    heuristicWeight: args.heuristicWeight,

    verbose: args.verbose,
  };
}


function statValue(result, name, fallback = 0) {
  if (!result || !result.stats) {
    return fallback;
  }

  const value = result.stats[name];

  return Number.isFinite(Number(value))
    ? Number(value)
    : fallback;
}


function solvedValue(result) {
  return Boolean(result && result.solved);
}


function elapsedValue(result) {
  return statValue(result, "elapsedMs", 0);
}


function makeRow({
  file,
  targetF,
  normal,
  policy,
}) {
  const row = {
    file,
    league: `f${targetF}`,
    targetF,

    normalSolved: solvedValue(normal),
    policySolved: solvedValue(policy),

    normalNodes: statValue(normal, "exploredNodes", 0),
    policyNodes: statValue(policy, "exploredNodes", 0),

    normalBacktracks: statValue(normal, "backtracks", 0),
    policyBacktracks: statValue(policy, "backtracks", 0),

    normalDeadEnds: statValue(normal, "deadEnds", 0),
    policyDeadEnds: statValue(policy, "deadEnds", 0),

    normalElapsedMs: elapsedValue(normal),
    policyElapsedMs: elapsedValue(policy),
  };

  row.nodesImprovement = row.normalNodes - row.policyNodes;
  row.backtracksImprovement = row.normalBacktracks - row.policyBacktracks;
  row.deadEndsImprovement = row.normalDeadEnds - row.policyDeadEnds;
  row.elapsedMsImprovement = row.normalElapsedMs - row.policyElapsedMs;

  return row;
}


function addRowToTotals(totals, row) {
  totals.files += 1;

  totals.normalSolved += row.normalSolved ? 1 : 0;
  totals.policySolved += row.policySolved ? 1 : 0;

  totals.normalNodes += row.normalNodes;
  totals.policyNodes += row.policyNodes;

  totals.normalBacktracks += row.normalBacktracks;
  totals.policyBacktracks += row.policyBacktracks;

  totals.normalDeadEnds += row.normalDeadEnds;
  totals.policyDeadEnds += row.policyDeadEnds;

  totals.normalElapsedMs += row.normalElapsedMs;
  totals.policyElapsedMs += row.policyElapsedMs;
}


function createTotals() {
  return {
    files: 0,

    normalSolved: 0,
    policySolved: 0,

    normalNodes: 0,
    policyNodes: 0,

    normalBacktracks: 0,
    policyBacktracks: 0,

    normalDeadEnds: 0,
    policyDeadEnds: 0,

    normalElapsedMs: 0,
    policyElapsedMs: 0,
  };
}


function summarizeTotals(totals) {
  const n = Math.max(1, totals.files);

  return {
    files: totals.files,

    normalSolved: totals.normalSolved,
    normalSolveRate: totals.normalSolved / n,

    policySolved: totals.policySolved,
    policySolveRate: totals.policySolved / n,

    avgNormalNodes: totals.normalNodes / n,
    avgPolicyNodes: totals.policyNodes / n,
    avgNodesImprovement: (
      totals.normalNodes - totals.policyNodes
    ) / n,

    avgNormalBacktracks: totals.normalBacktracks / n,
    avgPolicyBacktracks: totals.policyBacktracks / n,
    avgBacktracksImprovement: (
      totals.normalBacktracks - totals.policyBacktracks
    ) / n,

    avgNormalDeadEnds: totals.normalDeadEnds / n,
    avgPolicyDeadEnds: totals.policyDeadEnds / n,
    avgDeadEndsImprovement: (
      totals.normalDeadEnds - totals.policyDeadEnds
    ) / n,

    avgNormalElapsedMs: totals.normalElapsedMs / n,
    avgPolicyElapsedMs: totals.policyElapsedMs / n,
    avgElapsedMsImprovement: (
      totals.normalElapsedMs - totals.policyElapsedMs
    ) / n,
  };
}


function printRow(index, total, file, row) {
  const rel = path.relative(
    DEFAULT_REPLAY_ROOT,
    file
  );

  console.log(
    `[${index}/${total}] ` +
      `${rel} ` +
      `F=${row.targetF} ` +
      `N solved=${row.normalSolved} ` +
      `P solved=${row.policySolved} ` +
      `N nodes=${row.normalNodes} ` +
      `P nodes=${row.policyNodes} ` +
      `diff=${row.nodesImprovement} ` +
      `N ms=${row.normalElapsedMs.toFixed(3)} ` +
      `P ms=${row.policyElapsedMs.toFixed(3)}`
  );
}


async function benchmark(args) {
  resetDefaultPolicyCache();

  const files = listReplayFiles({
    root: args.root,
    league: args.league,
    limit: args.limit,
  });

  console.log("=== BENCHMARK ONNX POLICY SOLVER ===");
  console.log("Files:", files.length);
  console.log("League:", args.league);
  console.log("Replay root:", args.root);
  console.log("Rule: same word cannot be used more than once");
  console.log("Policy:", args.noPolicy ? "disabled" : "enabled");
  console.log("policyMinBranch:", args.policyMinBranch);
  console.log("policyTopK:", args.policyTopK);
  console.log("policyWeight:", args.policyWeight);
  console.log("heuristicWeight:", args.heuristicWeight);

  if (files.length === 0) {
    console.log("No replay files found.");
    return {
      rows: [],
      summary: summarizeTotals(createTotals()),
    };
  }

  const rows = [];
  const totals = createTotals();
  const byF = new Map();

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    let replay;
    let board;

    try {
      replay = readJson(file);
      board = replayBoard(replay);

      if (!board) {
        if (args.verbose) {
          console.log("SKIP no board:", file);
        }

        continue;
      }

      const solvedMoves = replaySolvedMoves(replay);

      if (!solvedMoves.length) {
        if (args.verbose) {
          console.log("SKIP no solvedMoves:", file);
        }

        continue;
      }

      const targetF = replayTargetF(replay, file);

      const options = makeSolveOptions({
        replay,
        targetF,
        args,
      });

      const normal = solve(board, {
        ...options,
        usePolicy: false,
      });

      const policy = await solvePolicy(board, {
        ...options,
        usePolicy: !args.noPolicy,
      });

      const row = makeRow({
        file,
        targetF,
        normal,
        policy,
      });

      rows.push(row);
      addRowToTotals(totals, row);

      const key = `f${targetF}`;

      if (!byF.has(key)) {
        byF.set(key, createTotals());
      }

      addRowToTotals(byF.get(key), row);

      printRow(
        rows.length,
        files.length,
        file,
        row
      );
    } catch (error) {
      console.log(
        `[ERROR] ${file}:`,
        error && error.stack ? error.stack : error
      );

      if (!args.verbose) {
        console.log("Use --verbose for more details.");
      }
    }
  }

  const summaryByF = {};

  for (const [key, value] of Array.from(byF.entries()).sort()) {
    summaryByF[key] = summarizeTotals(value);
  }

  const summary = {
    ...summarizeTotals(totals),
    byF: summaryByF,
  };

  return {
    rows,
    summary,
  };
}


async function main() {
  const args = parseArgs(process.argv);

  const result = await benchmark(args);

  console.log("\n=== SUMMARY ===");
  console.log(
    JSON.stringify(
      result.summary,
      null,
      2
    )
  );

  if (args.json) {
    const outPath = path.join(
      PROJECT_ROOT,
      "data",
      "cleanup",
      "benchmark_onnx_policy_result.json"
    );

    fs.mkdirSync(path.dirname(outPath), {
      recursive: true,
    });

    fs.writeFileSync(
      outPath,
      JSON.stringify(result, null, 2),
      "utf8"
    );

    console.log("\nSaved:", outPath);
  }
}


main().catch((error) => {
  console.error("\nBENCHMARK FAILED");
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});