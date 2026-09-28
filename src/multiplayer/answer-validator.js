"use strict";
const ROWS = 8;
const COLS = 5;
const PLAYABLE_START_ROW = 3;
const EMPTY = "・";
const WILDS = new Set(["Ｆ", "F"]);

function chars(word) { return [...String(word || "")]; }
function normalizePath(path) {
  if (!Array.isArray(path)) return null;
  const out = path.map(p => Array.isArray(p) && p.length === 2 ? [Number(p[0]), Number(p[1])] : null);
  if (out.some(p => !p || !Number.isInteger(p[0]) || !Number.isInteger(p[1]))) return null;
  return out;
}
function validatePath(path) {
  if (!path || path.length < 2 || path.length > 5) return "2〜5マスを選択してください";
  if (path.some(([r,c]) => r < 0 || r >= ROWS || c < 0 || c >= COLS)) return "盤面外です";
  if (path[0][0] < PLAYABLE_START_ROW) return "下5段から始めてください";
  const dr = path[1][0] - path[0][0];
  const dc = path[1][1] - path[0][1];
  if (!((Math.abs(dr) === 1 && dc === 0) || (Math.abs(dc) === 1 && dr === 0))) return "縦または横に連続して選んでください";
  for (let i = 1; i < path.length; i++) {
    if (path[i][0] - path[i-1][0] !== dr || path[i][1] - path[i-1][1] !== dc) return "選択マスが連続していません";
  }
  return null;
}
function applyGravity(board) {
  for (let c = 0; c < COLS; c++) {
    const kept = [];
    for (let r = ROWS - 1; r >= 0; r--) if (board[r][c] !== EMPTY) kept.push(board[r][c]);
    for (let r = ROWS - 1, i = 0; r >= 0; r--, i++) board[r][c] = i < kept.length ? kept[i] : EMPTY;
  }
}
function remainingCells(board) {
  let n = 0;
  for (const row of board) for (const cell of row) if (cell !== EMPTY) n++;
  return n;
}
function validateAndApply({ board, word, path, usedWords, isCountry }) {
  const w = chars(word);
  const normalized = normalizePath(path);
  const pathError = validatePath(normalized);
  if (pathError) return { ok: false, error: pathError };
  if (w.length !== normalized.length) return { ok: false, error: "単語と選択マスの長さが違います" };
  if (!isCountry(word)) return { ok: false, error: "国名リストにありません" };
  if (usedWords.has(word)) return { ok: false, error: "この問題ですでに使った国名です" };
  for (let i = 0; i < normalized.length; i++) {
    const [r,c] = normalized[i];
    const cell = board[r][c];
    if (cell === EMPTY) return { ok: false, error: "空マスは選択できません" };
    if (!WILDS.has(cell) && cell !== w[i]) return { ok: false, error: `${i+1}文字目が一致しません` };
  }
  for (const [r,c] of normalized) board[r][c] = EMPTY;
  applyGravity(board);
  usedWords.add(word);
  const remaining = remainingCells(board);
  return { ok: true, board, remaining, cleared: remaining === 0 };
}
module.exports = { validateAndApply, remainingCells };
