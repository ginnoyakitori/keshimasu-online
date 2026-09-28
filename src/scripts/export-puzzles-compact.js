// src/scripts/export-puzzles-compact.js
//
// Self-play replay から、1パズル=1行のコンパクトCSVを出力する。
//
// 出力例:
//   data/puzzles_compact/puzzles_compact.csv
//
// 実行:
//   node src/scripts/export-puzzles-compact.js
//   node src/scripts/export-puzzles-compact.js --league f3
//   node src/scripts/export-puzzles-compact.js --league f3 --limit 50
//   node src/scripts/export-puzzles-compact.js --output data/puzzles_compact/f3.csv
//

const fs = require("node:fs");
const path = require("node:path");

const {
  findAllMoves,
} = require("../solver/word-search");

const {
  getPlayableWords,
} = require("../generator/words");


const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

const DEFAULT_INPUT_ROOT = path.join(
  PROJECT_ROOT,
  "data",
  "training",
  "selfplay"
);

const DEFAULT_OUTPUT = path.join(
  PROJECT_ROOT,
  "data",
  "puzzles_compact",
  "puzzles_compact.csv"
);

const ROWS = 8;
const COLS = 5;
const WILDCARD = "F";


const COLUMNS = [
  "id",
  "mode",
  "hardness_score",
  "hardness_label",
  "elapsed_ms",
  "nodes",
  "visited_states",
  "max_depth",
  "timeout",
  "node_limit_reached",

  "solution_steps",
  "solution_horizontal_moves",
  "solution_vertical_moves",
  "solution_average_word_length",
  "solution_min_word_length",
  "solution_max_word_length",

  "expected_steps",
  "expected_horizontal_moves",
  "expected_vertical_moves",
  "expected_average_word_length",
  "expected_min_word_length",
  "expected_max_word_length",

  "board_char_count",
  "board_empty_count",
  "board_f_count",
  "board_f_rate",
  "board_unique_chars",
  "board_char_entropy",

  "visible_f_count",
  "visible_unique_chars",
  "board_column_heights",

  "initial_candidate_count",
  "initial_horizontal_candidate_count",
  "initial_vertical_candidate_count",
  "initial_f_candidate_count",
  "initial_avg_candidate_length",
  "initial_max_candidate_length",
  "initial_min_candidate_length",
  "initial_unique_word_count",
  "initial_candidate_cells_coverage_mean",
  "initial_horizontal_candidate_rate",
  "initial_vertical_candidate_rate",
  "initial_f_candidate_rate",

  "initial_len2_candidate_count",
  "initial_len3_candidate_count",
  "initial_len4_candidate_count",
  "initial_len5_candidate_count",

  "max_f_in_column",
  "max_consecutive_f_in_column",

  "initial_legal_candidate_count",
  "initial_no_f_long_candidate_count",
  "initial_no_f_vertical_long_candidate_count",

  "initial_visible_solution_count",
  "initial_visible_solution_no_f_count",
  "initial_visible_solution_long_no_f_count",
  "initial_visible_solution_vertical_long_no_f_count",

  "independent_visible_solution_columns",
  "solution_steps_not_initially_visible",
  "solution_f_move_count",
  "initial_ambiguous_f_rate",

  "falling_no_f_long_candidate_max",

  "initial_trap_move_count",
  "initial_trap_move_rate",
  "low_f_trap_move_count",
  "low_f_trap_move_rate",
  "trap_average_f_usage_rate",
  "trap_low_f_bonus",

  "generation_attempt",
  "board_key",
];


function parseArgs(argv) {
  const args = {
    inputRoot: DEFAULT_INPUT_ROOT,
    output: DEFAULT_OUTPUT,
    league: "all",
    limit: 0,
    mode: "country",
    maxPathsPerWord: 80,
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
    } else if (arg === "--output") {
      args.output = path.resolve(nextValue());
    } else if (arg.startsWith("--output=")) {
      args.output = path.resolve(arg.slice("--output=".length));
    } else if (arg === "--league") {
      args.league = nextValue();
    } else if (arg.startsWith("--league=")) {
      args.league = arg.slice("--league=".length);
    } else if (arg === "--limit") {
      args.limit = Number(nextValue());
    } else if (arg.startsWith("--limit=")) {
      args.limit = Number(arg.slice("--limit=".length));
    } else if (arg === "--mode") {
      args.mode = nextValue();
    } else if (arg.startsWith("--mode=")) {
      args.mode = arg.slice("--mode=".length);
    } else if (arg === "--max-paths-per-word") {
      args.maxPathsPerWord = Number(nextValue());
    } else if (arg.startsWith("--max-paths-per-word=")) {
      args.maxPathsPerWord = Number(arg.slice("--max-paths-per-word=".length));
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
Export compact puzzle CSV.

Usage:
  node src/scripts/export-puzzles-compact.js
  node src/scripts/export-puzzles-compact.js --league f3
  node src/scripts/export-puzzles-compact.js --league f3 --limit 50
  node src/scripts/export-puzzles-compact.js --output data/puzzles_compact/f3.csv

Options:
  --input-root PATH
  --output PATH
  --league f0|f1|f2|f3|all
  --limit N
  --mode country
  --max-paths-per-word N
`);
  process.exit(0);
}


function ensureDir(dir) {
  fs.mkdirSync(dir, {
    recursive: true,
  });
}


function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
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


function normalizeCell(cell) {
  if (cell === null || cell === undefined) {
    return null;
  }

  if (cell === "" || cell === "・") {
    return null;
  }

  if (cell === "Ｆ") {
    return WILDCARD;
  }

  return String(cell);
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
      return board.map((row) => row.map(normalizeCell));
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

  return getPlayableWords({
    minLen: 2,
    maxLen: 5,
  });
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
      return moves.map(normalizeMove);
    }
  }

  return [];
}


function getStats(replay) {
  const candidates = [
    replay.solverStats,
    replay.solver && replay.solver.stats,
    replay.stats,
    replay.result && replay.result.stats,
  ];

  for (const stats of candidates) {
    if (stats && typeof stats === "object") {
      return stats;
    }
  }

  return {};
}


function getScore(replay) {
  const candidates = [
    replay.score,
    replay.bestScore,
    replay.puzzle && replay.puzzle.score,
    replay.result && replay.result.score,
  ];

  for (const value of candidates) {
    const n = Number(value);

    if (Number.isFinite(n)) {
      return n;
    }
  }

  return null;
}


function normalizeMove(move) {
  if (!move || typeof move !== "object") {
    return move;
  }

  const out = {
    word: move.word ? String(move.word) : "",
    path: Array.isArray(move.path)
      ? move.path.map((p) => [Number(p[0]), Number(p[1])])
      : [],
    row: numberOrNull(move.row),
    col: numberOrNull(move.col),
    direction: move.direction,
  };

  if (out.path.length > 0) {
    if (out.row === null) {
      out.row = Number(out.path[0][0]);
    }

    if (out.col === null) {
      out.col = Number(out.path[0][1]);
    }
  }

  if (!out.direction && out.path.length >= 2) {
    const [r0, c0] = out.path[0];
    const [r1, c1] = out.path[1];

    if (r1 === r0 && c1 === c0 + 1) {
      out.direction = "H";
    } else if (r1 === r0 + 1 && c1 === c0) {
      out.direction = "V";
    }
  }

  if (out.direction === 0) {
    out.direction = "H";
  }

  if (out.direction === 1) {
    out.direction = "V";
  }

  return out;
}


function numberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}


function wordLength(word) {
  return Array.from(String(word || "")).length;
}


function moveLength(move) {
  if (move.word) {
    return wordLength(move.word);
  }

  if (Array.isArray(move.path)) {
    return move.path.length;
  }

  return 0;
}


function pathKey(path) {
  return (path || [])
    .map(([r, c]) => `${r},${c}`)
    .join(";");
}


function moveKey(move) {
  return `${move.word}:${pathKey(move.path)}`;
}


function boardKey(board) {
  return board
    .map((row) =>
      row.map((cell) => cell || "").join("")
    )
    .join("/");
}


function countBoardChars(board) {
  const chars = [];

  for (const row of board) {
    for (const cell of row) {
      if (cell !== null) {
        chars.push(cell);
      }
    }
  }

  return chars;
}


function entropy(values) {
  if (!values.length) {
    return 0;
  }

  const counts = new Map();

  for (const v of values) {
    counts.set(v, (counts.get(v) || 0) + 1);
  }

  let h = 0;
  const n = values.length;

  for (const count of counts.values()) {
    const p = count / n;
    h -= p * Math.log2(p);
  }

  return h;
}


function visibleCells(board) {
  // このプロジェクトでは下5段が playable / visible 扱い
  return board.slice(3).flat().filter((v) => v !== null);
}


function columnHeights(board) {
  const heights = [];

  for (let c = 0; c < COLS; c++) {
    let h = 0;

    for (let r = 0; r < ROWS; r++) {
      if (board[r][c] !== null) {
        h += 1;
      }
    }

    heights.push(h);
  }

  return heights;
}


function countFInPath(board, path) {
  let count = 0;

  for (const [r, c] of path || []) {
    if (board[r] && board[r][c] === WILDCARD) {
      count += 1;
    }
  }

  return count;
}


function moveUsesF(board, move) {
  return countFInPath(board, move.path) > 0;
}


function directionOfMove(move) {
  if (move.direction === "H" || move.direction === 0) {
    return "H";
  }

  if (move.direction === "V" || move.direction === 1) {
    return "V";
  }

  if (move.path && move.path.length >= 2) {
    const [r0, c0] = move.path[0];
    const [r1, c1] = move.path[1];

    if (r1 === r0 && c1 === c0 + 1) {
      return "H";
    }

    if (r1 === r0 + 1 && c1 === c0) {
      return "V";
    }
  }

  return "H";
}


function summarizeMoves(moves) {
  const lengths = moves.map(moveLength).filter((v) => v > 0);
  const steps = moves.length;

  const horizontal = moves.filter((m) => directionOfMove(m) === "H").length;
  const vertical = moves.filter((m) => directionOfMove(m) === "V").length;

  return {
    steps,
    horizontal,
    vertical,
    avgLen: avg(lengths),
    minLen: lengths.length ? Math.min(...lengths) : 0,
    maxLen: lengths.length ? Math.max(...lengths) : 0,
  };
}


function avg(values) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((a, b) => a + b, 0) / values.length;
}


function round3(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return 0;
  }

  return Math.round(n * 1000) / 1000;
}


function maxFInColumn(board) {
  let max = 0;

  for (let c = 0; c < COLS; c++) {
    let count = 0;

    for (let r = 0; r < ROWS; r++) {
      if (board[r][c] === WILDCARD) {
        count += 1;
      }
    }

    max = Math.max(max, count);
  }

  return max;
}


function maxConsecutiveFInColumn(board) {
  let best = 0;

  for (let c = 0; c < COLS; c++) {
    let cur = 0;

    for (let r = 0; r < ROWS; r++) {
      if (board[r][c] === WILDCARD) {
        cur += 1;
        best = Math.max(best, cur);
      } else {
        cur = 0;
      }
    }
  }

  return best;
}


function candidateCoverageMean(board, candidates) {
  const counts = new Map();

  for (const move of candidates) {
    for (const [r, c] of move.path || []) {
      const key = `${r},${c}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }

  const filled = countBoardChars(board).length;

  if (filled === 0) {
    return 0;
  }

  let total = 0;

  for (const value of counts.values()) {
    total += value;
  }

  return total / filled;
}


function getInitialCandidates(board, words, maxPathsPerWord) {
  try {
    return findAllMoves(board, {
      words,
      maxPathsPerWord,
      playableStartRow: 3,
      useWordOnce: true,
      usedWords: new Set(),
    });
  } catch {
    return [];
  }
}


function summarizeInitialCandidates(board, candidates) {
  const count = candidates.length;

  const horizontal = candidates.filter((m) => directionOfMove(m) === "H").length;
  const vertical = candidates.filter((m) => directionOfMove(m) === "V").length;
  const fCandidates = candidates.filter((m) => moveUsesF(board, m)).length;

  const lengths = candidates.map(moveLength).filter((v) => v > 0);

  const uniqueWords = new Set(candidates.map((m) => m.word)).size;

  const len2 = candidates.filter((m) => moveLength(m) === 2).length;
  const len3 = candidates.filter((m) => moveLength(m) === 3).length;
  const len4 = candidates.filter((m) => moveLength(m) === 4).length;
  const len5 = candidates.filter((m) => moveLength(m) === 5).length;

  const noFLong = candidates.filter(
    (m) => !moveUsesF(board, m) && moveLength(m) >= 4
  );

  const noFVerticalLong = candidates.filter(
    (m) =>
      !moveUsesF(board, m) &&
      moveLength(m) >= 4 &&
      directionOfMove(m) === "V"
  );

  return {
    count,
    horizontal,
    vertical,
    fCandidates,
    avgLen: avg(lengths),
    maxLen: lengths.length ? Math.max(...lengths) : 0,
    minLen: lengths.length ? Math.min(...lengths) : 0,
    uniqueWords,
    coverageMean: candidateCoverageMean(board, candidates),
    horizontalRate: count ? horizontal / count : 0,
    verticalRate: count ? vertical / count : 0,
    fRate: count ? fCandidates / count : 0,
    len2,
    len3,
    len4,
    len5,
    noFLongCount: noFLong.length,
    noFVerticalLongCount: noFVerticalLong.length,
  };
}


function initialVisibleSolutionMetrics(board, solutionMoves, candidates) {
  const candidateKeys = new Set(candidates.map(moveKey));

  const visible = solutionMoves.filter((move) => candidateKeys.has(moveKey(move)));

  const visibleNoF = visible.filter((m) => !moveUsesF(board, m));
  const visibleLongNoF = visible.filter(
    (m) => !moveUsesF(board, m) && moveLength(m) >= 4
  );
  const visibleVerticalLongNoF = visible.filter(
    (m) =>
      !moveUsesF(board, m) &&
      moveLength(m) >= 4 &&
      directionOfMove(m) === "V"
  );

  const solutionFMoveCount = solutionMoves.filter((m) => moveUsesF(board, m)).length;

  const verticalColumns = new Set();

  for (const move of visible) {
    if (directionOfMove(move) === "V" && move.path && move.path.length > 0) {
      verticalColumns.add(Number(move.path[0][1]));
    }
  }

  return {
    visibleCount: visible.length,
    visibleNoFCount: visibleNoF.length,
    visibleLongNoFCount: visibleLongNoF.length,
    visibleVerticalLongNoFCount: visibleVerticalLongNoF.length,
    independentVisibleSolutionColumns: verticalColumns.size,
    stepsNotInitiallyVisible: Math.max(0, solutionMoves.length - visible.length),
    solutionFMoveCount,
  };
}


function trapMetrics(board, candidates, solutionMoves) {
  const solutionKeys = new Set(solutionMoves.map(moveKey));

  const traps = candidates.filter((move) => !solutionKeys.has(moveKey(move)));

  const lowFTraps = traps.filter((move) => {
    const len = Math.max(1, moveLength(move));
    const fUsage = countFInPath(board, move.path) / len;
    return fUsage <= 0.25;
  });

  const fUsageRates = traps.map((move) => {
    const len = Math.max(1, moveLength(move));
    return countFInPath(board, move.path) / len;
  });

  const trapAverageFUsageRate = avg(fUsageRates);

  return {
    trapCount: traps.length,
    trapRate: candidates.length ? traps.length / candidates.length : 0,
    lowFTrapCount: lowFTraps.length,
    lowFTrapRate: traps.length ? lowFTraps.length / traps.length : 0,
    trapAverageFUsageRate,
    trapLowFBonus: lowFTraps.length,
  };
}


function generationAttemptFromFile(file) {
  const name = path.basename(file);
  const m = name.match(/epoch-(\d+)/);

  if (!m) {
    return 0;
  }

  return Number(m[1]);
}


function hardnessLabel(score) {
  if (score >= 500) {
    return "extreme";
  }

  if (score >= 250) {
    return "hard";
  }

  if (score >= 100) {
    return "normal";
  }

  return "easy";
}


function fallbackHardnessScore(stats, solution, candidates, board) {
  const nodes = Number(stats.exploredNodes ?? stats.nodes ?? 0);
  const deadEnds = Number(stats.deadEnds ?? 0);
  const backtracks = Number(stats.backtracks ?? 0);
  const fCount = countBoardChars(board).filter((c) => c === WILDCARD).length;

  return Math.round(
    nodes * 0.5 +
      backtracks * 2 +
      deadEnds * 3 +
      solution.length * 20 +
      candidates.length * 5 +
      fCount * 15
  );
}


function csvEscape(value) {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  const s = String(value);

  if (
    s.includes(",") ||
    s.includes("\n") ||
    s.includes("\r") ||
    s.includes('"')
  ) {
    return `"${s.replace(/"/g, '""')}"`;
  }

  return s;
}


function makeCompactRow({
  id,
  mode,
  replay,
  file,
  maxPathsPerWord,
}) {
  const board = getBoard(replay);
  const words = getWords(replay);
  const solutionMoves = getSolvedMoves(replay);
  const stats = getStats(replay);

  if (!board) {
    throw new Error("missing board");
  }

  if (!solutionMoves.length) {
    throw new Error("missing solvedMoves");
  }

  const candidates = getInitialCandidates(
    board,
    words,
    maxPathsPerWord
  );

  const boardChars = countBoardChars(board);
  const boardCharCount = boardChars.length;
  const boardEmptyCount = ROWS * COLS - boardCharCount;
  const boardFCount = boardChars.filter((c) => c === WILDCARD).length;

  const visible = visibleCells(board);
  const visibleFCount = visible.filter((c) => c === WILDCARD).length;

  const sol = summarizeMoves(solutionMoves);
  const expected = sol;

  const initial = summarizeInitialCandidates(board, candidates);
  const visibleSol = initialVisibleSolutionMetrics(
    board,
    solutionMoves,
    candidates
  );
  const traps = trapMetrics(board, candidates, solutionMoves);

  const rawScore = getScore(replay);
  const hardnessScore =
    rawScore !== null
      ? Math.round(rawScore)
      : fallbackHardnessScore(stats, solutionMoves, candidates, board);

  const elapsedMs = numberOrNull(stats.elapsedMs ?? stats.ms) ?? 0;
  const nodes = numberOrNull(stats.exploredNodes ?? stats.nodes) ?? 0;
  const visitedStates = numberOrNull(stats.visitedStates) ?? Math.max(0, nodes - 1);
  const maxDepth = numberOrNull(stats.maxDepth) ?? sol.steps;

  const timeout = Boolean(stats.timeout ?? replay.timeout ?? false);
  const nodeLimitReached = Boolean(
    stats.nodeLimitReached ??
      stats.maxNodesReached ??
      false
  );

  const row = {
    id,
    mode,

    hardness_score: hardnessScore,
    hardness_label: hardnessLabel(hardnessScore),

    elapsed_ms: Math.round(elapsedMs),
    nodes: Math.round(nodes),
    visited_states: Math.round(visitedStates),
    max_depth: Math.round(maxDepth),
    timeout,
    node_limit_reached: nodeLimitReached,

    solution_steps: sol.steps,
    solution_horizontal_moves: sol.horizontal,
    solution_vertical_moves: sol.vertical,
    solution_average_word_length: round3(sol.avgLen),
    solution_min_word_length: sol.minLen,
    solution_max_word_length: sol.maxLen,

    expected_steps: expected.steps,
    expected_horizontal_moves: expected.horizontal,
    expected_vertical_moves: expected.vertical,
    expected_average_word_length: round3(expected.avgLen),
    expected_min_word_length: expected.minLen,
    expected_max_word_length: expected.maxLen,

    board_char_count: boardCharCount,
    board_empty_count: boardEmptyCount,
    board_f_count: boardFCount,
    board_f_rate: round3(boardCharCount ? boardFCount / boardCharCount : 0),
    board_unique_chars: new Set(boardChars).size,
    board_char_entropy: round3(entropy(boardChars)),

    visible_f_count: visibleFCount,
    visible_unique_chars: new Set(visible).size,
    board_column_heights: columnHeights(board).join("|"),

    initial_candidate_count: initial.count,
    initial_horizontal_candidate_count: initial.horizontal,
    initial_vertical_candidate_count: initial.vertical,
    initial_f_candidate_count: initial.fCandidates,
    initial_avg_candidate_length: round3(initial.avgLen),
    initial_max_candidate_length: initial.maxLen,
    initial_min_candidate_length: initial.minLen,
    initial_unique_word_count: initial.uniqueWords,
    initial_candidate_cells_coverage_mean: round3(initial.coverageMean),
    initial_horizontal_candidate_rate: round3(initial.horizontalRate),
    initial_vertical_candidate_rate: round3(initial.verticalRate),
    initial_f_candidate_rate: round3(initial.fRate),

    initial_len2_candidate_count: initial.len2,
    initial_len3_candidate_count: initial.len3,
    initial_len4_candidate_count: initial.len4,
    initial_len5_candidate_count: initial.len5,

    max_f_in_column: maxFInColumn(board),
    max_consecutive_f_in_column: maxConsecutiveFInColumn(board),

    initial_legal_candidate_count: initial.count,
    initial_no_f_long_candidate_count: initial.noFLongCount,
    initial_no_f_vertical_long_candidate_count: initial.noFVerticalLongCount,

    initial_visible_solution_count: visibleSol.visibleCount,
    initial_visible_solution_no_f_count: visibleSol.visibleNoFCount,
    initial_visible_solution_long_no_f_count: visibleSol.visibleLongNoFCount,
    initial_visible_solution_vertical_long_no_f_count:
      visibleSol.visibleVerticalLongNoFCount,

    independent_visible_solution_columns:
      visibleSol.independentVisibleSolutionColumns,
    solution_steps_not_initially_visible:
      visibleSol.stepsNotInitiallyVisible,
    solution_f_move_count: visibleSol.solutionFMoveCount,
    initial_ambiguous_f_rate: round3(initial.fRate),

    falling_no_f_long_candidate_max: 0,

    initial_trap_move_count: traps.trapCount,
    initial_trap_move_rate: round3(traps.trapRate),
    low_f_trap_move_count: traps.lowFTrapCount,
    low_f_trap_move_rate: round3(traps.lowFTrapRate),
    trap_average_f_usage_rate: round3(traps.trapAverageFUsageRate),
    trap_low_f_bonus: round3(traps.trapLowFBonus),

    generation_attempt: generationAttemptFromFile(file),
    board_key: boardKey(board),
  };

  return row;
}


function main() {
  const args = parseArgs(process.argv);

  const files = listReplayFiles(
    args.inputRoot,
    args.league,
    args.limit
  );

  console.log("=== EXPORT COMPACT PUZZLE CSV ===");
  console.log("Input root:", args.inputRoot);
  console.log("Output    :", args.output);
  console.log("League    :", args.league);
  console.log("Files     :", files.length);

  ensureDir(path.dirname(args.output));

  const lines = [];
  lines.push(COLUMNS.join(","));

  let exported = 0;
  let skipped = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    try {
      const replay = readJson(file);

      const row = makeCompactRow({
        id: exported + 1,
        mode: args.mode,
        replay,
        file,
        maxPathsPerWord: args.maxPathsPerWord,
      });

      lines.push(
        COLUMNS.map((col) => csvEscape(row[col])).join(",")
      );

      exported += 1;

      console.log(
        `[${i + 1}/${files.length}] OK ${path.relative(PROJECT_ROOT, file)}`
      );
    } catch (error) {
      skipped += 1;

      console.log(
        `[${i + 1}/${files.length}] SKIP ${path.relative(PROJECT_ROOT, file)} reason=${error.message}`
      );
    }
  }

  fs.writeFileSync(
    args.output,
    lines.join("\n") + "\n",
    "utf8"
  );

  console.log("\n=== SUMMARY ===");
  console.log("Exported:", exported);
  console.log("Skipped :", skipped);
  console.log("Output  :", path.relative(PROJECT_ROOT, args.output));
}


main();
