// src/scripts/export-puzzles-only.js
//
// replay から「作られたパズルだけ」を抽出して保存するスクリプト。
//
// 入力:
//   data/training/selfplay/f0/*.json
//   data/training/selfplay/f1/*.json
//   data/training/selfplay/f2/*.json
//   data/training/selfplay/f3/*.json
//
// 出力:
//   data/puzzles_only/f0/*.json
//   data/puzzles_only/f1/*.json
//   data/puzzles_only/f2/*.json
//   data/puzzles_only/f3/*.json
//   data/puzzles_only/index.json
//
// 実行:
//   node src/scripts/export-puzzles-only.js
//   node src/scripts/export-puzzles-only.js --league f3
//   node src/scripts/export-puzzles-only.js --league f3 --limit 25
//   node src/scripts/export-puzzles-only.js --include-unsolved
//

const fs = require("node:fs");
const path = require("node:path");


const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

const DEFAULT_INPUT_ROOT = path.join(
  PROJECT_ROOT,
  "data",
  "training",
  "selfplay"
);

const DEFAULT_OUTPUT_ROOT = path.join(
  PROJECT_ROOT,
  "data",
  "puzzles_only"
);


function parseArgs(argv) {
  const args = {
    inputRoot: DEFAULT_INPUT_ROOT,
    outputRoot: DEFAULT_OUTPUT_ROOT,
    league: "all",
    limit: 0,
    includeUnsolved: false,
    overwrite: true,
    pretty: true,
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

    if (arg === "--input-root") {
      args.inputRoot = path.resolve(nextValue());
    } else if (arg.startsWith("--input-root=")) {
      args.inputRoot = path.resolve(arg.slice("--input-root=".length));
    } else if (arg === "--output-root") {
      args.outputRoot = path.resolve(nextValue());
    } else if (arg.startsWith("--output-root=")) {
      args.outputRoot = path.resolve(arg.slice("--output-root=".length));
    } else if (arg === "--league") {
      args.league = nextValue();
    } else if (arg.startsWith("--league=")) {
      args.league = arg.slice("--league=".length);
    } else if (arg === "--limit") {
      args.limit = Number(nextValue());
    } else if (arg.startsWith("--limit=")) {
      args.limit = Number(arg.slice("--limit=".length));
    } else if (arg === "--include-unsolved") {
      args.includeUnsolved = true;
    } else if (arg === "--no-overwrite") {
      args.overwrite = false;
    } else if (arg === "--compact") {
      args.pretty = false;
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
Export puzzles only from selfplay replays.

Usage:
  node src/scripts/export-puzzles-only.js
  node src/scripts/export-puzzles-only.js --league f3
  node src/scripts/export-puzzles-only.js --league f3 --limit 25
  node src/scripts/export-puzzles-only.js --output-root data/public_puzzles
  node src/scripts/export-puzzles-only.js --include-unsolved

Options:
  --input-root PATH       Default: data/training/selfplay
  --output-root PATH      Default: data/puzzles_only
  --league f0|f1|f2|f3|all
  --limit N
  --include-unsolved
  --no-overwrite
  --compact
`);
  process.exit(0);
}


function ensureDir(dir) {
  fs.mkdirSync(dir, {
    recursive: true,
  });
}


function readJson(file) {
  return JSON.parse(
    fs.readFileSync(file, "utf8")
  );
}


function writeJson(file, data, pretty = true) {
  ensureDir(path.dirname(file));

  fs.writeFileSync(
    file,
    JSON.stringify(data, null, pretty ? 2 : 0),
    "utf8"
  );
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
    } else if (
      entry.isFile() &&
      entry.name.toLowerCase().endsWith(".json")
    ) {
      files.push(full);
    }
  }

  return files;
}


function listReplayFiles(inputRoot, league, limit) {
  let files = [];

  if (league === "all") {
    for (const f of ["f0", "f1", "f2", "f3"]) {
      files.push(...listJsonFiles(path.join(inputRoot, f)));
    }
  } else {
    files = listJsonFiles(path.join(inputRoot, league));
  }

  files = files
    .filter((file) => {
      const name = path.basename(file);
      return name.startsWith("epoch-") && name.endsWith(".json");
    })
    .sort();

  if (limit && limit > 0) {
    files = files.slice(0, limit);
  }

  return files;
}


function getReplayId(replay, file) {
  return (
    replay.id ||
    replay.replayId ||
    replay.puzzleId ||
    path.basename(file, ".json")
  );
}


function getBoard(replay) {
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


function getWords(replay) {
  const candidates = [
    replay.words,
    replay.puzzle && replay.puzzle.words,
    replay.config && replay.config.words,
  ];

  for (const words of candidates) {
    if (Array.isArray(words) && words.length > 0) {
      return words.map(String);
    }
  }

  return [];
}


function getSolvedMoves(replay) {
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


function getTargetWildcards(replay, file) {
  const candidates = [
    replay.targetWildcards,
    replay.targetF,
    replay.actualWildcards,
    replay.puzzle && replay.puzzle.targetWildcards,
    replay.puzzle && replay.puzzle.targetF,
    replay.config && replay.config.targetWildcards,
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


function getScore(replay) {
  const candidates = [
    replay.score,
    replay.bestScore,
    replay.puzzle && replay.puzzle.score,
    replay.result && replay.result.score,
  ];

  for (const value of candidates) {
    if (value !== undefined && value !== null) {
      const n = Number(value);

      if (Number.isFinite(n)) {
        return n;
      }
    }
  }

  return null;
}


function getSolverStats(replay) {
  const candidates = [
    replay.solverStats,
    replay.solver && replay.solver.stats,
    replay.stats,
    replay.result && replay.result.stats,
  ];

  for (const stats of candidates) {
    if (stats && typeof stats === "object") {
      return {
        solved: Boolean(
          stats.solved ??
          replay.solved ??
          replay.solver?.solved ??
          true
        ),
        exploredNodes: numberOrNull(
          stats.exploredNodes ??
          stats.nodes
        ),
        backtracks: numberOrNull(stats.backtracks),
        deadEnds: numberOrNull(stats.deadEnds),
        forcedMoves: numberOrNull(stats.forcedMoves),
        maxDepth: numberOrNull(stats.maxDepth),
        elapsedMs: numberOrNull(
          stats.elapsedMs ??
          stats.ms
        ),
      };
    }
  }

  return {
    solved: Boolean(
      replay.solved ??
      replay.solver?.solved ??
      true
    ),
    exploredNodes: null,
    backtracks: null,
    deadEnds: null,
    forcedMoves: null,
    maxDepth: null,
    elapsedMs: null,
  };
}


function numberOrNull(value) {
  const n = Number(value);

  return Number.isFinite(n) ? n : null;
}


function normalizeMove(move) {
  if (!move || typeof move !== "object") {
    return move;
  }

  const out = {
    word: move.word,
    direction: move.direction,
    row: numberOrNull(move.row),
    col: numberOrNull(move.col),
    path: Array.isArray(move.path)
      ? move.path.map((p) => [Number(p[0]), Number(p[1])])
      : undefined,
  };

  if (out.direction === undefined && Array.isArray(out.path) && out.path.length >= 2) {
    const [r0, c0] = out.path[0];
    const [r1, c1] = out.path[1];

    if (r1 === r0 && c1 === c0 + 1) {
      out.direction = "H";
    } else if (r1 === r0 + 1 && c1 === c0) {
      out.direction = "V";
    }
  }

  if ((out.row === null || out.col === null) && Array.isArray(out.path) && out.path.length > 0) {
    out.row = Number(out.path[0][0]);
    out.col = Number(out.path[0][1]);
  }

  if (move.score !== undefined) {
    out.score = numberOrNull(move.score);
  }

  if (move.policyScore !== undefined) {
    out.policyScore = numberOrNull(move.policyScore);
  }

  return out;
}


function hasDuplicateWords(moves) {
  const seen = new Set();

  for (const move of moves) {
    if (!move || !move.word) {
      continue;
    }

    const word = String(move.word);

    if (seen.has(word)) {
      return true;
    }

    seen.add(word);
  }

  return false;
}


function buildPuzzleOnly(replay, file) {
  const id = getReplayId(replay, file);
  const board = getBoard(replay);
  const words = getWords(replay);
  const solvedMoves = getSolvedMoves(replay).map(normalizeMove);
  const targetWildcards = getTargetWildcards(replay, file);
  const score = getScore(replay);
  const solverStats = getSolverStats(replay);

  if (!board) {
    return {
      ok: false,
      reason: "missing board",
      puzzle: null,
    };
  }

  if (!solvedMoves.length) {
    return {
      ok: false,
      reason: "missing solvedMoves",
      puzzle: null,
    };
  }

  if (hasDuplicateWords(solvedMoves)) {
    return {
      ok: false,
      reason: "duplicate word in solvedMoves",
      puzzle: null,
    };
  }

  const puzzle = {
    id,
    format: "keshimasu-puzzle-only-v1",

    board,
    rows: board.length,
    cols: Array.isArray(board[0]) ? board[0].length : 0,

    targetWildcards,
    league: `f${targetWildcards}`,

    words,

    solution: {
      moves: solvedMoves,
      moveCount: solvedMoves.length,
      wordReuseRule: "same word cannot be used more than once",
    },

    score,

    solverStats,

    source: {
      replayFile: path.relative(PROJECT_ROOT, file).replace(/\\/g, "/"),
      exportedAt: new Date().toISOString(),
    },
  };

  return {
    ok: true,
    reason: "",
    puzzle,
  };
}


function makeOutputFile(outputRoot, puzzle, sourceFile) {
  const league = puzzle.league || "unknown";
  const sourceBase = path.basename(sourceFile, ".json");

  return path.join(
    outputRoot,
    league,
    `${sourceBase}.puzzle.json`
  );
}


function main() {
  const args = parseArgs(process.argv);

  const files = listReplayFiles(
    args.inputRoot,
    args.league,
    args.limit
  );

  console.log("=== EXPORT PUZZLES ONLY ===");
  console.log("Input root :", args.inputRoot);
  console.log("Output root:", args.outputRoot);
  console.log("League     :", args.league);
  console.log("Files      :", files.length);

  ensureDir(args.outputRoot);

  const index = {
    format: "keshimasu-puzzles-only-index-v1",
    exportedAt: new Date().toISOString(),
    inputRoot: path.relative(PROJECT_ROOT, args.inputRoot).replace(/\\/g, "/"),
    outputRoot: path.relative(PROJECT_ROOT, args.outputRoot).replace(/\\/g, "/"),
    league: args.league,
    count: 0,
    skipped: 0,
    byF: {},
    puzzles: [],
    skippedFiles: [],
  };

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    try {
      const replay = readJson(file);
      const result = buildPuzzleOnly(replay, file);

      if (!result.ok) {
        if (!args.includeUnsolved) {
          index.skipped += 1;
          index.skippedFiles.push({
            file: path.relative(PROJECT_ROOT, file).replace(/\\/g, "/"),
            reason: result.reason,
          });

          console.log(
            `[${i + 1}/${files.length}] SKIP ${path.basename(file)} reason=${result.reason}`
          );

          continue;
        }
      }

      const puzzle = result.puzzle;

      if (!puzzle) {
        index.skipped += 1;
        index.skippedFiles.push({
          file: path.relative(PROJECT_ROOT, file).replace(/\\/g, "/"),
          reason: result.reason || "unknown",
        });

        console.log(
          `[${i + 1}/${files.length}] SKIP ${path.basename(file)} reason=${result.reason}`
        );

        continue;
      }

      const outFile = makeOutputFile(
        args.outputRoot,
        puzzle,
        file
      );

      if (!args.overwrite && fs.existsSync(outFile)) {
        console.log(
          `[${i + 1}/${files.length}] EXISTS ${path.relative(PROJECT_ROOT, outFile)}`
        );
      } else {
        writeJson(
          outFile,
          puzzle,
          args.pretty
        );

        console.log(
          `[${i + 1}/${files.length}] SAVED ${path.relative(PROJECT_ROOT, outFile)}`
        );
      }

      const relOut = path.relative(PROJECT_ROOT, outFile).replace(/\\/g, "/");
      const league = puzzle.league || "unknown";

      index.count += 1;

      if (!index.byF[league]) {
        index.byF[league] = {
          count: 0,
        };
      }

      index.byF[league].count += 1;

      index.puzzles.push({
        id: puzzle.id,
        league,
        targetWildcards: puzzle.targetWildcards,
        board: relOut,
        moveCount: puzzle.solution.moveCount,
        score: puzzle.score,
        sourceReplay: puzzle.source.replayFile,
      });
    } catch (error) {
      index.skipped += 1;
      index.skippedFiles.push({
        file: path.relative(PROJECT_ROOT, file).replace(/\\/g, "/"),
        reason: error && error.message ? error.message : String(error),
      });

      console.log(
        `[${i + 1}/${files.length}] ERROR ${path.basename(file)} ${error.message}`
      );
    }
  }

  const indexFile = path.join(
    args.outputRoot,
    "index.json"
  );

  writeJson(
    indexFile,
    index,
    args.pretty
  );

  console.log("\n=== SUMMARY ===");
  console.log("Exported:", index.count);
  console.log("Skipped :", index.skipped);
  console.log("Index   :", path.relative(PROJECT_ROOT, indexFile));

  console.log(
    JSON.stringify(
      {
        count: index.count,
        skipped: index.skipped,
        byF: index.byF,
      },
      null,
      2
    )
  );
}


main();