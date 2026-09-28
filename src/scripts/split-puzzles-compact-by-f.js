// src/scripts/split-puzzles-compact-by-f.js
//
// data/puzzles_compact/puzzles_compact.csv を F数別に分割する。
// 分割キー: board_f_count
//
// Usage:
//   node src/scripts/split-puzzles-compact-by-f.js
//   node src/scripts/split-puzzles-compact-by-f.js --input data/puzzles_compact/puzzles_compact.csv
//   node src/scripts/split-puzzles-compact-by-f.js --output-dir data/puzzles_compact/by_f
//

const fs = require("node:fs");
const path = require("node:path");


const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

const DEFAULT_INPUT = path.join(
  PROJECT_ROOT,
  "data",
  "puzzles_compact",
  "puzzles_compact.csv"
);

const DEFAULT_OUTPUT_DIR = path.join(
  PROJECT_ROOT,
  "data",
  "puzzles_compact",
  "by_f"
);


function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    outputDir: DEFAULT_OUTPUT_DIR,
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

    if (arg === "--input") {
      args.input = path.resolve(nextValue());
    } else if (arg.startsWith("--input=")) {
      args.input = path.resolve(arg.slice("--input=".length));
    } else if (arg === "--output-dir") {
      args.outputDir = path.resolve(nextValue());
    } else if (arg.startsWith("--output-dir=")) {
      args.outputDir = path.resolve(arg.slice("--output-dir=".length));
    } else if (arg === "--help" || arg === "-h") {
      printHelpAndExit();
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}


function printHelpAndExit() {
  console.log(`
Split compact puzzle CSV by F count.

Usage:
  node src/scripts/split-puzzles-compact-by-f.js
  node src/scripts/split-puzzles-compact-by-f.js --input data/puzzles_compact/puzzles_compact.csv
  node src/scripts/split-puzzles-compact-by-f.js --output-dir data/puzzles_compact/by_f
`);
  process.exit(0);
}


function ensureDir(dir) {
  fs.mkdirSync(dir, {
    recursive: true,
  });
}


function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += ch;
    }
  }

  values.push(current);

  return values;
}


function splitCsvRows(text) {
  const rows = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
        current += ch;
      }
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && text[i + 1] === "\n") {
        i += 1;
      }

      if (current.length > 0) {
        rows.push(current);
      }

      current = "";
    } else {
      current += ch;
    }
  }

  if (current.length > 0) {
    rows.push(current);
  }

  return rows;
}


function normalizeFCount(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return 0;
  }

  return Math.max(0, Math.trunc(n));
}


function main() {
  const args = parseArgs(process.argv);

  if (!fs.existsSync(args.input)) {
    throw new Error(`Input CSV not found: ${args.input}`);
  }

  ensureDir(args.outputDir);

  const text = fs.readFileSync(args.input, "utf8");
  const rows = splitCsvRows(text);

  if (rows.length === 0) {
    throw new Error("CSV is empty.");
  }

  const header = rows[0];
  const columns = parseCsvLine(header);

  const fIndex = columns.indexOf("board_f_count");

  if (fIndex < 0) {
    throw new Error(
      "Column board_f_count not found. " +
      "Please export compact CSV with board_f_count column."
    );
  }

  const groups = new Map();

  for (let i = 1; i < rows.length; i++) {
    const line = rows[i];

    if (!line.trim()) {
      continue;
    }

    const values = parseCsvLine(line);
    const fCount = normalizeFCount(values[fIndex]);
    const key = `f${fCount}`;

    if (!groups.has(key)) {
      groups.set(key, []);
    }

    groups.get(key).push(line);
  }

  const index = {
    format: "keshimasu-puzzles-compact-by-f-index-v1",
    input: path.relative(PROJECT_ROOT, args.input).replace(/\\/g, "/"),
    outputDir: path.relative(PROJECT_ROOT, args.outputDir).replace(/\\/g, "/"),
    totalRows: rows.length - 1,
    byF: {},
    files: [],
  };

  for (const [key, lines] of Array.from(groups.entries()).sort()) {
    const outFile = path.join(args.outputDir, `${key}.csv`);

    fs.writeFileSync(
      outFile,
      [header, ...lines].join("\n") + "\n",
      "utf8"
    );

    const rel = path.relative(PROJECT_ROOT, outFile).replace(/\\/g, "/");

    index.byF[key] = {
      rows: lines.length,
      file: rel,
    };

    index.files.push(rel);

    console.log(
      `SAVED ${key}: ${lines.length} rows -> ${rel}`
    );
  }

  const indexFile = path.join(args.outputDir, "index.json");

  fs.writeFileSync(
    indexFile,
    JSON.stringify(index, null, 2),
    "utf8"
  );

  console.log("\n=== SUMMARY ===");
  console.log("Input     :", path.relative(PROJECT_ROOT, args.input));
  console.log("Output dir:", path.relative(PROJECT_ROOT, args.outputDir));
  console.log("Total rows:", index.totalRows);
  console.log("Groups    :", Object.keys(index.byF).join(", "));
  console.log("Index     :", path.relative(PROJECT_ROOT, indexFile));
}


main();