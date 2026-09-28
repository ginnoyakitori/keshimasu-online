"use strict";
const fs = require("node:fs");
const path = require("node:path");

function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(p));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) out.push(p);
  }
  return out;
}
function readWordList(file) {
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  if (Array.isArray(data)) return data;
  for (const key of ["words", "countries", "data"]) if (Array.isArray(data[key])) return data[key];
  throw new Error(`Unsupported country file: ${file}`);
}
function cloneBoard(board) { return board.map(row => row.map(v => String(v))); }
function isPuzzle(p) {
  const board = p && p.board;
  if (!Array.isArray(board) || board.length !== 8 || !board.every(r => Array.isArray(r) && r.length === 5)) return false;
  if (p.solved !== true) return false;
  if (Number(p.filledCount ?? 40) !== 40 || Number(p.emptyCount ?? 0) !== 0) return false;
  const mode = p.config?.wordMode;
  return !mode || mode === "country" || mode === "custom";
}
class PuzzleLoader {
  constructor({ puzzleDir, countryFile }) {
    this.puzzleDir = puzzleDir;
    this.countryFile = countryFile;
    this.puzzles = [];
    this.countries = new Set();
  }
  load() {
    this.countries = new Set(readWordList(this.countryFile).map(String).filter(w => [...w].length >= 2 && [...w].length <= 5));
    this.puzzles = [];
    for (const file of walk(this.puzzleDir)) {
      try {
        const p = JSON.parse(fs.readFileSync(file, "utf8"));
        if (!isPuzzle(p)) continue;
        this.puzzles.push({
          id: path.basename(file, ".json"),
          file,
          board: cloneBoard(p.board),
          targetWildcards: Number(p.targetWildcards ?? p.actualWildcards ?? 0),
          difficulty: Number(p.solverStats?.exploredNodes ?? 0),
          versions: p.versions ?? null
        });
      } catch (error) {
        console.warn(`Skip puzzle ${file}: ${error.message}`);
      }
    }
    if (!this.puzzles.length) throw new Error(`No playable country puzzles in ${this.puzzleDir}`);
  }
  get count() { return this.puzzles.length; }
  get countryCount() { return this.countries.size; }
  random(excludeId = null) {
    const choices = this.puzzles.filter(p => p.id !== excludeId);
    const list = choices.length ? choices : this.puzzles;
    const p = list[Math.floor(Math.random() * list.length)];
    return { ...p, board: cloneBoard(p.board) };
  }
  isCountry(word) { return this.countries.has(String(word)); }
}
module.exports = { PuzzleLoader, cloneBoard };
