"use strict";

/* ==================================================
   ケシマス対戦クライアント（3分スコアバトル）
================================================== */

const MAX_SELECTION = 5;
const SESSION_KEY = "keshimasu-duel-session";
const NAME_KEY = "keshimasu-duel-name";
const MODE_KEY = "keshimasu-duel-mode";
const MUTE_KEY = "keshimasu_muted_v1";

const KANA_PALETTE_ROWS = [
  ["ア", "イ", "ウ", "エ", "オ"],
  ["カ", "キ", "ク", "ケ", "コ"],
  ["サ", "シ", "ス", "セ", "ソ"],
  ["タ", "チ", "ツ", "テ", "ト"],
  ["ナ", "ニ", "ヌ", "ネ", "ノ"],
  ["ハ", "ヒ", "フ", "ヘ", "ホ"],
  ["マ", "ミ", "ム", "メ", "モ"],
  ["ヤ", "", "ユ", "", "ヨ"],
  ["ラ", "リ", "ル", "レ", "ロ"],
  ["ワ", "ヲ", "ン", "ー", "ッ"],
  ["ガ", "ギ", "グ", "ゲ", "ゴ"],
  ["ザ", "ジ", "ズ", "ゼ", "ゾ"],
  ["ダ", "ヂ", "ヅ", "デ", "ド"],
  ["バ", "ビ", "ブ", "ベ", "ボ"],
  ["パ", "ピ", "プ", "ペ", "ポ"],
  ["ャ", "ュ", "ョ", "ヴ", ""],
  ["ァ", "ィ", "ゥ", "ェ", "ォ"],
];
const POKEMON_PALETTE_ROW = ["♂", "♀", "Z", "2", "・"];

const socket = io({ transports: ["polling", "websocket"] });

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function prefersReducedMotion() {
  return Boolean(
    window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}


/* ==================================================
   効果音・振動（音声ファイルは使わず、その場で合成）
================================================== */

const FX = (() => {
  let ctx = null;
  let muted = false;

  try {
    muted = localStorage.getItem(MUTE_KEY) === "1";
  } catch {
    muted = false;
  }

  function getContext() {
    if (muted) {
      return null;
    }

    if (!ctx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;

      if (!AudioContextClass) {
        return null;
      }

      try {
        ctx = new AudioContextClass();
      } catch {
        return null;
      }
    }

    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    return ctx;
  }

  function tone({ freq, to, type = "sine", start = 0, dur = 0.12, gain = 0.12 }) {
    const audio = getContext();

    if (!audio) {
      return;
    }

    const t0 = audio.currentTime + start;
    const osc = audio.createOscillator();
    const amp = audio.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);

    if (to) {
      osc.frequency.exponentialRampToValueAtTime(to, t0 + dur);
    }

    amp.gain.setValueAtTime(0.0001, t0);
    amp.gain.exponentialRampToValueAtTime(gain, t0 + 0.01);
    amp.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

    osc.connect(amp).connect(audio.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  function noise({ start = 0, dur = 0.12, gain = 0.1, highpass = 1800 }) {
    const audio = getContext();

    if (!audio) {
      return;
    }

    const t0 = audio.currentTime + start;
    const length = Math.max(1, Math.floor(audio.sampleRate * dur));
    const buffer = audio.createBuffer(1, length, audio.sampleRate);
    const data = buffer.getChannelData(0);

    for (let index = 0; index < length; index += 1) {
      data[index] = Math.random() * 2 - 1;
    }

    const source = audio.createBufferSource();
    const filter = audio.createBiquadFilter();
    const amp = audio.createGain();

    source.buffer = buffer;
    filter.type = "highpass";
    filter.frequency.value = highpass;

    amp.gain.setValueAtTime(gain, t0);
    amp.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

    source.connect(filter).connect(amp).connect(audio.destination);
    source.start(t0);
  }

  function vibrate(pattern) {
    if (muted) {
      return;
    }

    try {
      if (navigator.vibrate) {
        navigator.vibrate(pattern);
      }
    } catch {
      // 振動に対応していない端末では何もしない
    }
  }

  return {
    unlock() {
      getContext();
    },

    isMuted() {
      return muted;
    },

    setMuted(value) {
      muted = Boolean(value);

      try {
        localStorage.setItem(MUTE_KEY, muted ? "1" : "0");
      } catch {
        // 保存できなくても、このページでは有効
      }
    },

    // 文字を選んだ「ポッ」（長くなるほど少しずつ高くなる）
    tick(count = 1) {
      const freq = 480 + Math.min(count, 10) * 45;

      tone({ freq, to: freq * 1.12, type: "triangle", dur: 0.07, gain: 0.07 });
      vibrate(8);
    },

    // ワードが成立した「ピロリン」
    valid() {
      tone({ freq: 784, type: "sine", dur: 0.1, gain: 0.1 });
      tone({ freq: 1175, type: "sine", start: 0.08, dur: 0.16, gain: 0.1 });
      vibrate(14);
    },

    // 消去の「パァン」
    erase(count = 2) {
      const rise = 600 + Math.min(count, 8) * 60;

      noise({ dur: 0.13, gain: 0.09 });
      tone({ freq: 420, to: rise * 2, type: "triangle", dur: 0.16, gain: 0.14 });
      tone({ freq: rise, to: rise * 1.5, type: "sine", start: 0.05, dur: 0.14, gain: 0.08 });
      vibrate(22);
    },

    // 不正解の低い音
    error() {
      tone({ freq: 190, to: 110, type: "sawtooth", dur: 0.22, gain: 0.08 });
      tone({ freq: 150, to: 90, type: "square", start: 0.09, dur: 0.2, gain: 0.05 });
      vibrate([30, 40, 30]);
    },

    // リセットの小さな音
    undo() {
      tone({ freq: 620, to: 380, type: "triangle", dur: 0.12, gain: 0.07 });
      vibrate(10);
    },

    // レベルクリア・勝利のファンファーレ
    clear() {
      [523, 659, 784, 1047].forEach((freq, index) => {
        tone({
          freq,
          type: "triangle",
          start: index * 0.11,
          dur: index === 3 ? 0.5 : 0.14,
          gain: 0.12,
        });
      });

      tone({ freq: 1568, type: "sine", start: 0.44, dur: 0.45, gain: 0.06 });
      vibrate([40, 50, 40, 50, 120]);
    },
  };
})();

document.addEventListener("pointerdown", () => FX.unlock(), { once: true });

// 画面右上の「音のオン／オフ」ボタン
function createSoundToggle() {
  const button = document.createElement("button");

  button.type = "button";
  button.id = "soundToggle";

  const render = () => {
    const muted = FX.isMuted();

    button.textContent = muted ? "🔇" : "🔊";
    button.setAttribute("aria-pressed", muted ? "true" : "false");
    button.setAttribute(
      "aria-label",
      muted ? "効果音と振動: オフ（押すとオン）" : "効果音と振動: オン（押すとオフ）"
    );
    button.title = muted ? "効果音・振動: オフ" : "効果音・振動: オン";
  };

  button.addEventListener("click", () => {
    FX.setMuted(!FX.isMuted());
    render();

    if (!FX.isMuted()) {
      FX.tick(3);
    }
  });

  render();
  document.body.appendChild(button);
}

createSoundToggle();


/* ==================================================
   紙吹雪
================================================== */

function launchConfetti(durationMs = 2800) {
  if (prefersReducedMotion()) {
    return;
  }

  const canvas = document.createElement("canvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = window.innerWidth;
  const height = window.innerHeight;

  canvas.className = "confetti-canvas";
  canvas.setAttribute("aria-hidden", "true");
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const context = canvas.getContext("2d");

  if (!context) {
    return;
  }

  context.scale(dpr, dpr);
  document.body.appendChild(canvas);

  const colors = ["#FFD700", "#ff6b6b", "#6ddf6d", "#4dabf7", "#ffffff", "#f783ac"];
  const particles = [];

  // 画面の中ほどから打ち上がる紙吹雪
  for (let index = 0; index < 90; index += 1) {
    const angle = -Math.PI * (0.1 + Math.random() * 0.8);
    const speed = 7 + Math.random() * 9;

    particles.push({
      x: width / 2,
      y: height * 0.45,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      gravity: 0.32,
      drag: 0.985,
    });
  }

  // 上からひらひら降ってくる紙吹雪
  for (let index = 0; index < 70; index += 1) {
    particles.push({
      x: Math.random() * width,
      y: -20 - Math.random() * height * 0.5,
      vx: (Math.random() - 0.5) * 2,
      vy: 2 + Math.random() * 3,
      gravity: 0.03,
      drag: 0.995,
    });
  }

  for (const particle of particles) {
    particle.w = 6 + Math.random() * 5;
    particle.h = 9 + Math.random() * 6;
    particle.angle = Math.random() * Math.PI * 2;
    particle.spin = (Math.random() - 0.5) * 0.35;
    particle.sway = Math.random() * Math.PI * 2;
    particle.color = colors[Math.floor(Math.random() * colors.length)];
  }

  const startTime = performance.now();

  function frame(now) {
    const elapsed = now - startTime;
    const fade =
      elapsed > durationMs - 600 ? Math.max(0, (durationMs - elapsed) / 600) : 1;

    context.clearRect(0, 0, width, height);

    for (const particle of particles) {
      particle.vx *= particle.drag;
      particle.vy = particle.vy * particle.drag + particle.gravity;
      particle.x += particle.vx + Math.sin(particle.sway + elapsed / 250) * 0.6;
      particle.y += particle.vy;
      particle.angle += particle.spin;

      context.save();
      context.globalAlpha = fade;
      context.translate(particle.x, particle.y);
      context.rotate(particle.angle);
      context.fillStyle = particle.color;
      context.fillRect(-particle.w / 2, -particle.h / 2, particle.w, particle.h);
      context.restore();
    }

    if (elapsed < durationMs) {
      requestAnimationFrame(frame);
    } else {
      canvas.remove();
    }
  }

  requestAnimationFrame(frame);
}


/* ==================================================
   要素
================================================== */

const el = {
  lobby: $("lobby"),
  game: $("game"),
  name: $("name"),
  mode: $("mode"),
  matchButton: $("matchButton"),
  cancelMatch: $("cancelMatch"),
  matchingIndicator: $("matchingIndicator"),
  message: $("message"),
  yourName: $("yourName"),
  opponentName: $("opponentName"),
  timer: $("timer"),
  levelNumber: $("levelNumber"),
  yourScore: $("yourScore"),
  opponentScore: $("opponentScore"),
  opponentLevel: $("opponentLevel"),
  opponentRemaining: $("opponentRemaining"),
  opponentProgress: $("opponentProgress"),
  countdown: $("countdown"),
  boardWrap: $("boardWrap"),
  board: $("board"),
  fxLayer: $("fxLayer"),
  levelBanner: $("levelBanner"),
  selectedText: $("selectedText"),
  submitMove: $("submitMove"),
  resetBoard: $("resetBoard"),
  gameMessage: $("gameMessage"),
  paletteModal: $("paletteModal"),
  paletteTitle: $("paletteTitle"),
  paletteWord: $("paletteWord"),
  paletteGrid: $("paletteGrid"),
  paletteCancel: $("paletteCancel"),
  result: $("result"),
  resultTitle: $("resultTitle"),
  resultScores: $("resultScores"),
  resultNote: $("resultNote"),
  rematch: $("rematch"),
  newMatch: $("newMatch"),
};


/* ==================================================
   状態
================================================== */

let status = "lobby"; // lobby | matching | countdown | playing | finished
let roomId = null;
let playerToken = null;
let mode = "country";

// 現在の問題
let puzzle = null; // { cols, rows, visibleRows, visibleStartRow, level, totalCells }
let board = []; // 見えている行だけ（visibleRows × cols）
let selected = []; // [[絶対行, 列], ...]
let usedWords = new Set(); // この盤面で使ったワード（画面表示用）

// 辞書（選択中のワードが成立するかを示すために使う）
let dictionary = null;
let dictionaryMode = null;
let dictionaryChars = null;

let startAt = 0;
let endAt = 0;
let clockOffset = 0; // サーバー時刻 - 端末時刻

let timerFrame = null;
let countdownTimer = null;
let sending = false;
let lastScore = 0;
let lastOpponentLevel = 1;
let lastCountdownShown = 0;
let lastSecondShown = null;
let opponentLeft = false;

let lastFeedbackCount = 0;
let lastFeedbackStatus = "";
let ignoreClickUntil = 0;

const drag = {
  active: false,
  pointerId: null,
  startCell: null,
  startElement: null,
  axis: null,
  moved: false,
  previousSelection: [],
};

const serverNow = () => Date.now() + clockOffset;

function syncClock(serverTime) {
  if (Number.isFinite(serverTime)) {
    clockOffset = serverTime - Date.now();
  }
}


/* ==================================================
   画面表示の補助
================================================== */

function show(screen) {
  el.lobby.classList.toggle("hidden", screen !== "lobby");
  el.game.classList.toggle("hidden", screen !== "game");
}

function setMessage(text) {
  el.message.textContent = text || "";
}

function setGameMessage(text) {
  el.gameMessage.textContent = text || "";
}

function saveSession() {
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ roomId, playerToken }));
  } catch {
    // 保存できなくても対戦は続けられる
  }
}

function loadSession() {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
  } catch {
    return null;
  }
}

function clearSession() {
  try {
    sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // 無視
  }
}

function formatScore(value) {
  return Number(value || 0).toLocaleString("ja-JP");
}

function bump(element) {
  const item = element.parentElement;

  item.classList.remove("bump");
  void item.offsetWidth;
  item.classList.add("bump");
}


/* ==================================================
   辞書
================================================== */

async function loadDictionary(targetMode) {
  if (dictionaryMode === targetMode && dictionary) {
    return;
  }

  dictionary = null;
  dictionaryChars = null;
  dictionaryMode = targetMode;

  try {
    const response = await fetch(`/api/words/${encodeURIComponent(targetMode)}`);
    const data = await response.json();

    if (!data.ok || dictionaryMode !== targetMode) {
      return;
    }

    dictionary = new Set(data.words);
    dictionaryChars = new Set(data.words.join("").split(""));
  } catch {
    // 取得できなくても、判定はサーバーが行うので遊べる
    dictionary = null;
    dictionaryChars = null;
  }
}


/* ==================================================
   ロビー
================================================== */

function resetLobbyUi() {
  el.matchButton.disabled = false;
  el.mode.disabled = false;
  el.name.disabled = false;
  el.cancelMatch.classList.add("hidden");
  el.matchingIndicator.classList.add("hidden");
}

function setMatchingUi(active) {
  el.matchButton.disabled = active;
  el.mode.disabled = active;
  el.name.disabled = active;
  el.cancelMatch.classList.toggle("hidden", !active);
  el.matchingIndicator.classList.toggle("hidden", !active);
}

async function loadModes() {
  try {
    const response = await fetch("/api/modes");
    const data = await response.json();

    if (!data.ok) {
      return;
    }

    const available = new Set(data.modes.map((entry) => entry.id));

    for (const option of el.mode.options) {
      option.disabled = !available.has(option.value);
    }

    if (el.mode.selectedOptions[0]?.disabled) {
      const first = [...el.mode.options].find((option) => !option.disabled);

      if (first) {
        el.mode.value = first.value;
      }
    }
  } catch {
    // 取得できなくても、固定の選択肢で続けられる
  }
}

try {
  el.name.value = localStorage.getItem(NAME_KEY) || el.name.value;

  const savedMode = localStorage.getItem(MODE_KEY);

  if (savedMode) {
    el.mode.value = savedMode;
  }
} catch {
  // 無視
}

el.matchButton.addEventListener("click", () => {
  const name = el.name.value.trim() || "プレイヤー";

  mode = el.mode.value;

  try {
    localStorage.setItem(NAME_KEY, name);
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    // 無視
  }

  setMatchingUi(true);
  status = "matching";
  setMessage("対戦相手を探しています…");

  // 待ち時間のうちに辞書を取っておく
  loadDictionary(mode);

  socket.emit("matchmaking:join", { name, mode }, (response) => {
    if (!response?.ok) {
      status = "lobby";
      resetLobbyUi();
      setMessage(response?.error || "マッチングに失敗しました");
    }
  });
});

el.cancelMatch.addEventListener("click", () => {
  socket.emit("matchmaking:cancel", {}, () => {
    status = "lobby";
    resetLobbyUi();
    setMessage("マッチングをやめました");
  });
});

socket.on("matchmaking:waiting", (data) => {
  setMessage(`${data.modeLabel}モードの対戦相手を探しています…`);
});


/* ==================================================
   盤面の描画
================================================== */

function applyPuzzle(data, { animate = false } = {}) {
  puzzle = {
    id: data.id,
    level: data.level,
    cols: data.cols,
    rows: data.rows,
    visibleRows: data.visibleRows,
    visibleStartRow: data.visibleStartRow,
    totalCells: data.totalCells,
  };

  board = data.board.map((row) => row.slice());
  usedWords = new Set();

  clearSelection();
  el.levelNumber.textContent = String(puzzle.level);

  renderBoard();

  if (animate && !prefersReducedMotion()) {
    el.board.classList.remove("level-in");
    void el.board.offsetWidth;
    el.board.classList.add("level-in");
  }
}

function cellAt(row, col) {
  return board[row - puzzle.visibleStartRow]?.[col] ?? "";
}

function isFilledCell(row, col) {
  return (
    Boolean(puzzle) &&
    row >= puzzle.visibleStartRow &&
    row < puzzle.rows &&
    col >= 0 &&
    col < puzzle.cols &&
    cellAt(row, col) !== ""
  );
}

/*
 * 盤面を描き直します。
 * fallMap があれば、そのマスに落下アニメーションを付けます。
 */
function renderBoard(fallMap = null) {
  if (!puzzle) {
    return;
  }

  el.board.style.setProperty("--cols", String(puzzle.cols));
  el.board.replaceChildren();

  board.forEach((cells, visibleRow) => {
    const row = visibleRow + puzzle.visibleStartRow;

    cells.forEach((character, col) => {
      const cell = document.createElement("button");

      cell.type = "button";
      cell.className = "cell";
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);

      if (col === puzzle.cols - 1) {
        cell.classList.add("last-col");
      }

      if (visibleRow === board.length - 1) {
        cell.classList.add("last-row");
      }

      if (character === "") {
        cell.classList.add("empty");
        cell.disabled = true;
        cell.tabIndex = -1;
        cell.setAttribute("aria-hidden", "true");
      } else {
        cell.textContent = character;
        cell.setAttribute(
          "aria-label",
          `${character}、${visibleRow + 1}行目${col + 1}列目`
        );
        cell.setAttribute("aria-pressed", "false");

        const fall = fallMap?.get(`${row},${col}`);

        if (fall) {
          cell.classList.add("falling");
          cell.style.setProperty("--fall", String(fall));
          cell.style.setProperty("--fall-dur", `${fallDurationMs(fall)}ms`);
        }
      }

      el.board.appendChild(cell);
    });
  });

  refreshSelectionView();
}

// 対戦は時間との勝負なので、落下は短めにして操作を待たせない
function fallDurationMs(rows) {
  return Math.min(300, 110 + 45 * rows);
}

/*
 * 消したあとに、各文字が何段落ちるかを求めます。
 * 戻り値のキーは「落ちたあとの位置」です。
 */
function computeFallMap(oldBoard, erasedCells, newBoard) {
  const map = new Map();
  const start = puzzle.visibleStartRow;

  for (let col = 0; col < puzzle.cols; col += 1) {
    const erasedRows = new Set(
      erasedCells.filter(([, c]) => c === col).map(([r]) => r)
    );

    if (erasedRows.size === 0) {
      continue;
    }

    // 消えずに残った文字の、元の行（下から順）
    const remainingRows = [];

    for (let row = puzzle.rows - 1; row >= start; row -= 1) {
      if (oldBoard[row - start][col] !== "" && !erasedRows.has(row)) {
        remainingRows.push(row);
      }
    }

    for (let index = 0; index < puzzle.visibleRows; index += 1) {
      const newRow = puzzle.rows - 1 - index;

      if (newBoard[newRow - start]?.[col] === "") {
        break;
      }

      const fall =
        index < remainingRows.length
          ? newRow - remainingRows[index]
          : erasedRows.size; // 見えていなかった上の行から落ちてきた文字

      if (fall > 0) {
        map.set(`${newRow},${col}`, fall);
      }
    }
  }

  return map;
}

// 消した文字を、盤面の上に重ねて光らせて縮ませる（操作は止めない）
function spawnGhosts(cells, word) {
  if (prefersReducedMotion()) {
    return;
  }

  const wrapRect = el.boardWrap.getBoundingClientRect();

  cells.forEach(([row, col], index) => {
    const node = el.board.querySelector(
      `.cell[data-row="${row}"][data-col="${col}"]`
    );

    if (!node) {
      return;
    }

    const rect = node.getBoundingClientRect();
    const ghost = document.createElement("div");

    ghost.className = "ghost-cell";
    ghost.textContent = [...word][index] || node.textContent;
    ghost.style.left = `${rect.left - wrapRect.left}px`;
    ghost.style.top = `${rect.top - wrapRect.top}px`;
    ghost.style.width = `${rect.width}px`;
    ghost.style.height = `${rect.height}px`;
    ghost.style.fontSize = getComputedStyle(node).fontSize;
    ghost.style.setProperty("--delay", `${Math.min(index * 25, 100)}ms`);

    el.fxLayer.appendChild(ghost);
    setTimeout(() => ghost.remove(), 600);
  });
}

function showLevelBanner(level, gained) {
  el.levelBanner.replaceChildren();

  const title = document.createElement("span");

  title.textContent = `LEVEL ${level}`;

  const sub = document.createElement("small");

  sub.textContent = `+${formatScore(gained)}点`;

  el.levelBanner.append(title, sub);
  el.levelBanner.classList.remove("hidden");

  // アニメーションをやり直す
  el.levelBanner.style.animation = "none";
  void el.levelBanner.offsetWidth;
  el.levelBanner.style.animation = "";

  setTimeout(() => el.levelBanner.classList.add("hidden"), 1000);
}

function shakeSelectedCells() {
  const nodes = [...el.board.querySelectorAll(".cell.selected")];

  for (const node of nodes) {
    node.classList.remove("shake");
    void node.offsetWidth;
    node.classList.add("shake");
  }

  setTimeout(() => {
    for (const node of nodes) {
      node.classList.remove("shake");
    }
  }, 420);
}


/* ==================================================
   選択（タップ・ドラッグ・キーボード）
================================================== */

function sortReadingOrder(cells) {
  const sorted = cells.map(([row, col]) => [row, col]);

  if (sorted.length < 2) {
    return sorted;
  }

  if (sorted.every(([row]) => row === sorted[0][0])) {
    sorted.sort((left, right) => left[1] - right[1]);
  } else if (sorted.every(([, col]) => col === sorted[0][1])) {
    sorted.sort((left, right) => left[0] - right[0]);
  }

  return sorted;
}

// 横は左から右、縦は上から下に読んだ文字（判定もこの順で行われます）
function readingChars() {
  return sortReadingOrder(selected).map(([row, col]) => cellAt(row, col));
}

function clearSelection() {
  selected = [];
  lastFeedbackCount = 0;
  lastFeedbackStatus = "";

  refreshSelectionView();
}

/*
 * 選んだ文字が成立しているか。
 *  'valid' … 消せる / 'used' … 使用済み / '' … 成立していない・判定できない
 */
function selectionStatus(chars) {
  if (chars.length < 2 || !dictionary) {
    return "";
  }

  if (!chars.includes("F")) {
    const word = chars.join("");

    if (!dictionary.has(word)) {
      return "";
    }

    return usedWords.has(word) ? "used" : "valid";
  }

  let foundUsed = false;

  for (const candidate of dictionary) {
    if ([...candidate].length !== chars.length) {
      continue;
    }

    const candidateChars = [...candidate];

    if (!chars.every((ch, index) => ch === "F" || ch === candidateChars[index])) {
      continue;
    }

    if (!usedWords.has(candidate)) {
      return "valid";
    }

    foundUsed = true;
  }

  return foundUsed ? "used" : "";
}

// 盤面を作り直さずに、選択の見た目だけを更新する
function refreshSelectionView() {
  const chars = readingChars();
  const selectionStatusValue = selectionStatus(chars);

  for (const node of el.board.querySelectorAll(".cell")) {
    const row = Number(node.dataset.row);
    const col = Number(node.dataset.col);
    const order = selected.findIndex(([r, c]) => r === row && c === col);

    node.classList.toggle("selected", order > -1);
    node.classList.toggle("valid", order > -1 && selectionStatusValue === "valid");
    node.classList.toggle("used", order > -1 && selectionStatusValue === "used");

    if (node.tagName === "BUTTON" && !node.disabled) {
      node.setAttribute("aria-pressed", order > -1 ? "true" : "false");
    }

    if (order > -1) {
      node.dataset.order = String(order + 1);
    } else {
      delete node.dataset.order;
    }
  }

  el.selectedText.textContent = chars.length > 0 ? chars.join("") : "未選択";

  const canSubmit = status === "playing" && !sending && selected.length >= 2;

  el.submitMove.disabled = !canSubmit;
  el.submitMove.classList.toggle("ready", canSubmit && selectionStatusValue === "valid");
  el.resetBoard.disabled = status !== "playing" || sending;

  if (selectionStatusValue === "used" && chars.length >= 2) {
    el.selectedText.textContent = `${chars.join("")}（使用済み）`;
  }

  // 選んだときの「ポッ」「ピロリン」（同じ状態での再描画では鳴らさない）
  if (selectionStatusValue === "valid" && lastFeedbackStatus !== "valid") {
    FX.valid();
  } else if (selected.length > lastFeedbackCount) {
    FX.tick(selected.length);
  }

  lastFeedbackCount = selected.length;
  lastFeedbackStatus = selectionStatusValue;
}

function canOperateBoard() {
  return status === "playing" && !sending && Boolean(puzzle);
}

// 1マスをタップしたとき
function applyTap(row, col) {
  if (!isFilledCell(row, col)) {
    return;
  }

  setGameMessage("");

  if (selected.length === 0) {
    selected = [[row, col]];
  } else {
    const [lastRow, lastCol] = selected[selected.length - 1];
    const horizontal = row === lastRow && Math.abs(col - lastCol) === 1;
    const vertical = col === lastCol && Math.abs(row - lastRow) === 1;
    const index = selected.findIndex(([r, c]) => r === row && c === col);

    if (index > -1) {
      if (index === selected.length - 1) {
        // 最後に選んだマスをもう一度タップ → 1つ取り消す
        selected.pop();
      } else {
        selected = selected.slice(0, index + 1);
      }
    } else if (horizontal || vertical) {
      let shouldAdd = false;

      if (selected.length === 1) {
        shouldAdd = true;
      } else {
        const [firstRow, firstCol] = selected[0];
        const isHorizontalLine = selected.every(([r]) => r === firstRow);
        const isVerticalLine = selected.every(([, c]) => c === firstCol);

        if (isHorizontalLine) {
          shouldAdd = row === firstRow && horizontal;
        } else if (isVerticalLine) {
          shouldAdd = col === firstCol && vertical;
        }
      }

      if (shouldAdd && selected.length < MAX_SELECTION) {
        selected.push([row, col]);
      } else {
        selected = [[row, col]];
      }
    } else {
      selected = [[row, col]];
    }
  }

  refreshSelectionView();
}

// なぞった範囲を、押したマスからの一直線として求める
function computeDragSelection(anchor, target, previousAxis) {
  const [anchorRow, anchorCol] = anchor;
  const deltaRow = target[0] - anchorRow;
  const deltaCol = target[1] - anchorCol;

  if (deltaRow === 0 && deltaCol === 0) {
    return { cells: [[anchorRow, anchorCol]], axis: null };
  }

  let axis = previousAxis;

  // 逆方向に大きくずれたときだけ、縦横を切り替える
  if (axis === "h" && Math.abs(deltaRow) > Math.abs(deltaCol) + 1) {
    axis = "v";
  } else if (axis === "v" && Math.abs(deltaCol) > Math.abs(deltaRow) + 1) {
    axis = "h";
  }

  // 向きに沿った移動量がなくなったら、向きを決め直す
  if (axis === "h" && deltaCol === 0) {
    axis = null;
  }

  if (axis === "v" && deltaRow === 0) {
    axis = null;
  }

  if (axis === null) {
    if (Math.abs(deltaCol) > Math.abs(deltaRow)) {
      axis = "h";
    } else if (Math.abs(deltaRow) > Math.abs(deltaCol)) {
      axis = "v";
    } else {
      return { cells: [[anchorRow, anchorCol]], axis: null };
    }
  }

  const stepRow = axis === "v" ? Math.sign(deltaRow) : 0;
  const stepCol = axis === "h" ? Math.sign(deltaCol) : 0;
  const distance = axis === "v" ? Math.abs(deltaRow) : Math.abs(deltaCol);

  const cells = [[anchorRow, anchorCol]];
  let row = anchorRow;
  let col = anchorCol;

  for (let index = 0; index < distance && cells.length < MAX_SELECTION; index += 1) {
    row += stepRow;
    col += stepCol;

    // 空きマス・盤面の外・見えていない段で止める
    if (!isFilledCell(row, col)) {
      break;
    }

    cells.push([row, col]);
  }

  return { cells, axis };
}

function cellAtPoint(x, y) {
  const node = document.elementFromPoint(x, y);
  const cell = node instanceof Element ? node.closest("button.cell") : null;

  if (!cell || !el.board.contains(cell) || cell.disabled) {
    return null;
  }

  return [Number(cell.dataset.row), Number(cell.dataset.col)];
}

function clearPressingMark() {
  drag.startElement?.classList.remove("pressing");
}

function resetDragState() {
  clearPressingMark();

  drag.active = false;
  drag.pointerId = null;
  drag.startCell = null;
  drag.startElement = null;
  drag.axis = null;
  drag.moved = false;
  drag.previousSelection = [];
}

el.board.addEventListener("pointerdown", (event) => {
  if (!canOperateBoard() || drag.active || !event.isPrimary) {
    return;
  }

  if (event.pointerType === "mouse" && event.button !== 0) {
    return;
  }

  const node =
    event.target instanceof Element ? event.target.closest("button.cell") : null;

  // 空きマスや枠線から始めたときは何もしない
  if (!node || node.disabled || !el.board.contains(node)) {
    return;
  }

  drag.active = true;
  drag.pointerId = event.pointerId;
  drag.startElement = node;
  drag.startCell = [Number(node.dataset.row), Number(node.dataset.col)];
  drag.axis = null;
  drag.moved = false;
  drag.previousSelection = selected.map((cell) => [...cell]);

  try {
    el.board.setPointerCapture(event.pointerId);
  } catch {
    // 取得できなくても、タップとしては動作する
  }

  // 押した瞬間にも反応が分かるように、押したマスを色づける
  node.classList.add("pressing");
});

el.board.addEventListener("pointermove", (event) => {
  if (!drag.active || event.pointerId !== drag.pointerId) {
    return;
  }

  const target = cellAtPoint(event.clientX, event.clientY);

  if (!target) {
    return;
  }

  const [startRow, startCol] = drag.startCell;

  // まだ押したマスの中にいる間は、タップかもしれないので何もしない
  if (!drag.moved && target[0] === startRow && target[1] === startCol) {
    return;
  }

  if (!drag.moved) {
    drag.moved = true;
    clearPressingMark();
  }

  const result = computeDragSelection(drag.startCell, target, drag.axis);

  drag.axis = result.axis;

  const changed =
    result.cells.length !== selected.length ||
    result.cells.some(
      (cell, index) =>
        cell[0] !== selected[index]?.[0] || cell[1] !== selected[index]?.[1]
    );

  if (changed) {
    selected = result.cells;
    setGameMessage("");
    refreshSelectionView();
  }
});

el.board.addEventListener("pointerup", (event) => {
  if (!drag.active || event.pointerId !== drag.pointerId) {
    return;
  }

  const wasMoved = drag.moved;
  const [row, col] = drag.startCell;

  try {
    el.board.releasePointerCapture(event.pointerId);
  } catch {
    // すでに解放されていれば何もしない
  }

  resetDragState();

  // この後に届く click は、ここで処理済みなので無視する
  ignoreClickUntil = Date.now() + 400;

  if (!wasMoved) {
    applyTap(row, col);
  }
});

el.board.addEventListener("pointercancel", (event) => {
  if (!drag.active || event.pointerId !== drag.pointerId) {
    return;
  }

  // 画面のスクロールなどで中断されたときは、なぞる前の選択に戻す
  const restore = drag.previousSelection;
  const wasMoved = drag.moved;

  resetDragState();

  if (wasMoved) {
    selected = restore;
    refreshSelectionView();
  }
});

// 長押しのメニューが、なぞり操作の邪魔をしないようにする
el.board.addEventListener("contextmenu", (event) => event.preventDefault());

// キーボード（Enter / Space）やスクリーンリーダーからの選択
el.board.addEventListener("click", (event) => {
  // マウス・タッチ由来の click は detail が 1 以上（上で処理済み）
  if (event.detail > 0 && Date.now() < ignoreClickUntil) {
    return;
  }

  if (!canOperateBoard()) {
    return;
  }

  const node =
    event.target instanceof Element ? event.target.closest("button.cell") : null;

  if (node && !node.disabled) {
    applyTap(Number(node.dataset.row), Number(node.dataset.col));
  }
});

// 矢印キーでマス間を移動、Escで選択解除、Enterで消去
el.board.addEventListener("keydown", (event) => {
  const target = event.target;

  if (!(target instanceof HTMLElement) || target.dataset.row === undefined) {
    return;
  }

  if (event.key === "Escape") {
    clearSelection();

    return;
  }

  const delta = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  }[event.key];

  if (!delta) {
    return;
  }

  event.preventDefault();

  let row = Number(target.dataset.row);
  let col = Number(target.dataset.col);

  // 空きマスは飛ばして、その方向の次の文字のマスへ移動する
  for (;;) {
    row += delta[0];
    col += delta[1];

    if (
      row < puzzle.visibleStartRow ||
      row >= puzzle.rows ||
      col < 0 ||
      col >= puzzle.cols
    ) {
      return;
    }

    const next = el.board.querySelector(
      `button.cell[data-row="${row}"][data-col="${col}"]`
    );

    if (next && !next.disabled) {
      next.focus();

      return;
    }
  }
});

document.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    status === "playing" &&
    !el.submitMove.disabled &&
    el.paletteModal.classList.contains("hidden") &&
    el.result.classList.contains("hidden") &&
    !(event.target instanceof HTMLButtonElement)
  ) {
    event.preventDefault();
    submitSelection();
  }
});


/* ==================================================
   Fに入る文字を選ぶパレット
================================================== */

let paletteResolve = null;

function closePalette(value) {
  el.paletteModal.classList.add("hidden");

  const resolve = paletteResolve;

  paletteResolve = null;

  if (resolve) {
    resolve(value);
  }
}

// chars: 選んだ文字（Fを含む）、targetIndex: 今決めるFの位置。キャンセルなら null
function askWildcard(chars, targetIndex) {
  return new Promise((resolve) => {
    if (paletteResolve) {
      closePalette(null);
    }

    paletteResolve = resolve;

    el.paletteTitle.textContent = `${targetIndex + 1}文字目（F）を何にしますか？`;

    el.paletteWord.replaceChildren(
      ...chars.map((character, index) => {
        const span = document.createElement("span");

        span.textContent = index === targetIndex ? "？" : character;

        if (index === targetIndex) {
          span.className = "wildcard-target";
        }

        return span;
      })
    );

    const rows = [...KANA_PALETTE_ROWS];

    if (mode === "pokemon") {
      rows.push(POKEMON_PALETTE_ROW);
    }

    const keys = [];

    for (const character of rows.flat()) {
      if (!character) {
        const spacer = document.createElement("span");

        spacer.className = "palette-spacer";
        keys.push(spacer);

        continue;
      }

      // このモードの辞書に出てこない文字は出さない
      if (dictionaryChars && !dictionaryChars.has(character)) {
        continue;
      }

      const key = document.createElement("button");

      key.type = "button";
      key.className = "palette-key";
      key.textContent = character;
      key.setAttribute("aria-label", character);
      key.addEventListener("click", () => {
        FX.tick(2);
        closePalette(character);
      });

      keys.push(key);
    }

    el.paletteGrid.replaceChildren(...keys);
    el.paletteModal.classList.remove("hidden");
    el.paletteGrid.querySelector(".palette-key")?.focus({ preventScroll: true });
  });
}

el.paletteCancel.addEventListener("click", () => closePalette(null));

el.paletteModal.addEventListener("click", (event) => {
  if (event.target === el.paletteModal) {
    closePalette(null);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el.paletteModal.classList.contains("hidden")) {
    closePalette(null);
  }
});


/* ==================================================
   消去（回答の送信）
================================================== */

async function submitSelection() {
  if (!canOperateBoard() || selected.length < 2) {
    return;
  }

  const ordered = sortReadingOrder(selected);
  const chars = ordered.map(([row, col]) => cellAt(row, col));
  const wildcardIndexes = chars
    .map((character, index) => (character === "F" ? index : -1))
    .filter((index) => index >= 0);

  // Fは、読む順（左→右・上→下）に1文字ずつ選ぶ
  let fText = "";

  if (wildcardIndexes.length > 0) {
    const working = [...chars];

    for (const index of wildcardIndexes) {
      const picked = await askWildcard(working, index);

      if (picked === null || status !== "playing") {
        return;
      }

      working[index] = picked;
      fText += picked;
    }

    chars.splice(0, chars.length, ...working);
  }

  const word = chars.join("");

  // 辞書がある場合は、明らかな間違いをサーバーへ送る前に知らせる
  if (dictionary && !dictionary.has(word)) {
    rejectLocally(`「${word}」はリストにありません`);

    return;
  }

  if (usedWords.has(word)) {
    rejectLocally(`「${word}」はこの問題ですでに使用しています`);

    return;
  }

  sending = true;
  refreshSelectionView();

  const snapshotBoard = board.map((row) => row.slice());
  const snapshotCells = ordered;

  socket.emit("move:submit", { path: ordered, fText }, (response) => {
    sending = false;

    if (!response?.ok) {
      FX.error();
      shakeSelectedCells();
      setGameMessage(response?.error || "判定に失敗しました");
      refreshSelectionView();

      return;
    }

    handleMoveSuccess(response, snapshotBoard, snapshotCells);
  });
}

function rejectLocally(message) {
  FX.error();
  shakeSelectedCells();
  setGameMessage(message);
}

function handleMoveSuccess(response, oldBoard, erasedCells) {
  syncClock(response.serverTime);

  usedWords.add(response.word);

  FX.erase(erasedCells.length);
  spawnGhosts(erasedCells, response.word);
  applyProgress(response.progress, { mine: true });

  if (response.levelCleared) {
    FX.clear();
    launchConfetti(1600);
    applyPuzzle(response.puzzle, { animate: true });
    showLevelBanner(response.puzzle.level, response.gained);

    setGameMessage(
      `「${response.word}」正解！ Lv${response.clearedLevel}クリア！ ` +
        `+${formatScore(response.gained)}点`
    );

    return;
  }

  const newBoard = response.board.map((row) => row.slice());
  const fallMap = prefersReducedMotion()
    ? null
    : computeFallMap(oldBoard, erasedCells, newBoard);

  board = newBoard;
  selected = [];
  lastFeedbackCount = 0;
  lastFeedbackStatus = "";

  renderBoard(fallMap);
  setGameMessage(`「${response.word}」正解！`);
}

el.submitMove.addEventListener("click", submitSelection);

el.resetBoard.addEventListener("click", () => {
  if (!canOperateBoard()) {
    return;
  }

  if (!window.confirm("この盤面を最初の状態に戻しますか？（時間は戻りません）")) {
    return;
  }

  socket.emit("match:reset", {}, (response) => {
    if (!response?.ok) {
      setGameMessage(response?.error || "リセットできませんでした");

      return;
    }

    FX.undo();

    board = response.board.map((row) => row.slice());
    usedWords = new Set();
    selected = [];
    lastFeedbackCount = 0;
    lastFeedbackStatus = "";

    applyProgress(response.progress, { mine: true });
    renderBoard();
    setGameMessage("盤面を最初の状態に戻しました");
  });
});


/* ==================================================
   得点・進捗
================================================== */

function applyProgress(progress, { mine }) {
  if (!progress) {
    return;
  }

  if (mine) {
    el.yourScore.textContent = formatScore(progress.score);

    if (progress.score > lastScore) {
      bump(el.yourScore);
    }

    lastScore = progress.score;
    el.levelNumber.textContent = String(progress.level);

    return;
  }

  el.opponentScore.textContent = formatScore(progress.score);
  el.opponentLevel.textContent = String(progress.level);
  el.opponentRemaining.textContent = String(progress.remaining);

  // 相手がレベルを上げたら、得点を光らせて知らせる
  if (progress.level > lastOpponentLevel) {
    bump(el.opponentScore);
  }

  lastOpponentLevel = progress.level;

  const total = progress.totalCells || 1;
  const percent = Math.round(((total - progress.remaining) / total) * 100);

  el.opponentProgress.style.width = `${percent}%`;
  el.opponentProgress.parentElement.setAttribute("aria-valuenow", String(percent));
}

socket.on("opponent:progress", (progress) => {
  applyProgress(progress, { mine: false });
});

function applyState(state) {
  if (!state) {
    return;
  }

  const you = state.players.find((player) => player.isYou);
  const rival = state.players.find((player) => !player.isYou);

  if (you) {
    el.yourName.textContent = you.name;
  }

  opponentLeft = !rival;

  if (rival) {
    el.opponentName.textContent = rival.name;
    applyProgress(rival, { mine: false });
  }

  if (status === "finished") {
    el.rematch.disabled = opponentLeft;

    if (opponentLeft) {
      el.resultNote.textContent = "相手が退出しました。";
    }
  }
}

socket.on("room:state", applyState);


/* ==================================================
   タイマーとカウントダウン
================================================== */

function formatTime(milliseconds) {
  const totalSeconds = Math.max(0, Math.ceil(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function stopTimer() {
  if (timerFrame) {
    cancelAnimationFrame(timerFrame);
    timerFrame = null;
  }

  lastSecondShown = null;
}

function tickTimer() {
  const remaining = endAt - serverNow();
  const seconds = Math.max(0, Math.ceil(remaining / 1000));

  el.timer.textContent = formatTime(remaining);
  el.timer.parentElement.classList.toggle("danger", remaining <= 10_000);

  // 残り3秒からは、1秒ごとに音で知らせる
  if (seconds !== lastSecondShown) {
    if (seconds > 0 && seconds <= 3 && lastSecondShown !== null) {
      FX.tick(seconds);
    }

    lastSecondShown = seconds;
  }

  if (remaining <= 0 && status === "playing") {
    setGameMessage("時間切れ！結果を待っています…");
  }

  if (status === "playing" && remaining > -3000) {
    timerFrame = requestAnimationFrame(tickTimer);
  } else {
    timerFrame = null;
  }
}

function startTimer() {
  stopTimer();
  tickTimer();
}

function stopCountdown() {
  if (countdownTimer) {
    clearInterval(countdownTimer);
    countdownTimer = null;
  }

  lastCountdownShown = 0;
  el.countdown.textContent = "";
}

function runCountdown() {
  stopCountdown();

  const update = () => {
    const remaining = startAt - serverNow();

    if (remaining <= 0) {
      stopCountdown();

      return;
    }

    const shown = Math.ceil(remaining / 1000);

    el.countdown.textContent = String(shown);

    if (shown !== lastCountdownShown) {
      FX.tick(shown + 2);
      lastCountdownShown = shown;
    }
  };

  update();
  countdownTimer = setInterval(update, 100);
}


/* ==================================================
   マッチ成立 → カウントダウン → 開始 → 終了
================================================== */

socket.on("matchmaking:matched", (data) => {
  roomId = data.roomId;
  playerToken = data.playerToken;
  mode = data.mode;

  saveSession();
  loadDictionary(mode);

  el.opponentName.textContent = data.opponentName;
  applyState(data.state);
});

socket.on("match:countdown", (data) => {
  syncClock(data.serverTime);

  closePalette(null);

  status = "countdown";
  startAt = data.startAt;
  endAt = data.endAt;
  mode = data.mode;
  lastScore = 0;
  lastOpponentLevel = 1;
  sending = false;

  el.result.classList.add("hidden");
  el.yourScore.textContent = "0";
  el.opponentScore.textContent = "0";
  el.opponentLevel.textContent = "1";
  el.opponentProgress.style.width = "0%";
  el.timer.textContent = formatTime(data.durationMs);
  el.timer.parentElement.classList.remove("danger");

  applyPuzzle(data.puzzle, { animate: true });
  el.opponentRemaining.textContent = String(data.puzzle.totalCells);

  setMessage("");
  setGameMessage(`${data.modeLabel}モード・制限時間3分`);
  show("game");
  refreshSelectionView();
  runCountdown();
});

socket.on("match:started", (data) => {
  syncClock(data.serverTime);

  status = "playing";
  startAt = data.startAt;
  endAt = data.endAt;

  stopCountdown();
  startTimer();
  FX.valid();
  setGameMessage("スタート！");
  refreshSelectionView();
});

socket.on("match:error", (data) => {
  setGameMessage(data?.error || "エラーが発生しました");
});

function showResult(result) {
  status = "finished";
  sending = false;

  closePalette(null);
  stopTimer();
  stopCountdown();
  clearSelection();

  el.timer.textContent = "0:00";

  const you = result.players.find((player) => player.token === playerToken);
  const won = !result.draw && result.winnerToken === playerToken;
  const lost = !result.draw && !won;

  el.resultTitle.textContent = result.draw ? "引き分け" : won ? "勝利！" : "敗北…";
  el.resultTitle.classList.toggle("winner", won);
  el.resultTitle.classList.toggle("loser", lost);

  el.resultScores.replaceChildren();

  for (const player of result.players) {
    const item = document.createElement("li");

    if (player.token === playerToken) {
      item.classList.add("you");
    }

    const label = document.createElement("span");

    label.textContent =
      `${player.name}${player.token === playerToken ? "（あなた）" : ""}` +
      ` ・ Lv${player.level}`;

    const score = document.createElement("span");

    score.className = "score";
    score.textContent = `${formatScore(player.score)}点`;

    item.append(label, score);
    el.resultScores.appendChild(item);
  }

  el.resultNote.textContent =
    you && lost ? "次は勝てるかも。もう一度挑戦しよう。" : "";

  el.rematch.disabled = opponentLeft;
  el.rematch.textContent = "同じ相手と再戦（同じモード）";
  el.result.classList.remove("hidden");
  refreshSelectionView();

  if (won) {
    FX.clear();
    launchConfetti();
  } else if (lost) {
    FX.undo();
  }
}

socket.on("match:finished", showResult);

el.rematch.addEventListener("click", () => {
  socket.emit("match:rematch", {}, (response) => {
    if (!response?.ok) {
      el.resultNote.textContent = response?.error || "再戦できません";

      return;
    }

    el.rematch.disabled = true;
    el.rematch.textContent = "相手を待っています…";
  });
});

function backToLobby() {
  stopTimer();
  stopCountdown();
  closePalette(null);
  clearSession();

  roomId = null;
  playerToken = null;
  puzzle = null;
  status = "lobby";

  el.result.classList.add("hidden");
  resetLobbyUi();
  setMessage("");
  show("lobby");
}

el.newMatch.addEventListener("click", () => {
  socket.emit("match:leave", {}, () => {
    backToLobby();
    el.matchButton.click();
  });
});


/* ==================================================
   再接続
================================================== */

socket.on("connect", () => {
  const session = loadSession();

  if (!session?.roomId || !session?.playerToken) {
    return;
  }

  socket.emit("room:resume", session, (response) => {
    if (!response?.ok) {
      clearSession();

      if (status !== "lobby") {
        backToLobby();
        setMessage("対戦への復帰に失敗しました。もう一度マッチングしてください。");
      }

      return;
    }

    roomId = session.roomId;
    playerToken = session.playerToken;
    mode = response.mode;

    loadDictionary(mode);
    syncClock(response.serverTime);

    startAt = response.startAt;
    endAt = response.endAt;

    if (response.puzzle) {
      applyPuzzle(response.puzzle);
    }

    lastScore = response.progress?.score ?? 0;
    applyProgress(response.progress, { mine: true });
    applyState(response.state);
    show("game");

    const roomStatus = response.state.status;

    if (roomStatus === "playing") {
      status = "playing";
      startTimer();
      setGameMessage("対戦に復帰しました");
    } else if (roomStatus === "countdown") {
      status = "countdown";
      runCountdown();
    } else if (roomStatus === "finished" && response.result) {
      showResult(response.result);
    }

    refreshSelectionView();
  });
});

socket.on("disconnect", () => {
  if (status === "matching") {
    setMessage("サーバーへ再接続しています…");
  } else if (status === "playing" || status === "countdown") {
    setGameMessage("サーバーへ再接続しています…");
  }
});


/* ==================================================
   初期表示
================================================== */

resetLobbyUi();
show("lobby");
loadModes();
