"use strict";

/*
 * 制限時間つきスコア対戦の管理。
 *
 *  - マッチング時にモード（国名・首都名・ポケモン）を選び、同じモードの2人を組む
 *  - 制限時間は3分。同じ問題列（レベル1、2、3…）を2人が同時に解く
 *  - 盤面を全部消すと次のレベルへ進む。スキップはない
 *  - 得点 = 全消ししたレベルの点数 + 今の盤面で消したマス数 × 2
 *  - 時間切れで得点が高い方の勝ち。同点なら、その得点に先に届いた方の勝ち
 */

const fs = require("node:fs");
const path = require("node:path");

const { roomCode, playerToken } = require("./room-code");
const { cloneBoard } = require("./puzzle-loader");
const { validateAndApply } = require("./answer-validator");
const {
  DEFAULT_MODE,
  MODES,
  LEVELS,
  COUNTDOWN_MS,
  MATCH_DURATION_MS,
  PARTIAL_POINTS_PER_CELL,
  clearPoints,
} = require("./levels");

// 全員が切断した部屋を片付けるまでの時間
const ABANDONED_ROOM_MS = 3 * 60 * 1000;
const SWEEP_INTERVAL_MS = 60 * 1000;


/* ==================================================
   共通関数
================================================== */

function cleanName(value) {
  return String(value || "プレイヤー").trim().slice(0, 20) || "プレイヤー";
}

function ackSafe(ack, response) {
  if (typeof ack === "function") {
    ack(response);
  }
}

/*
 * サーバーは盤面全体を保持しますが、
 * ブラウザへ送るのは見えている下側の行だけです。
 */
function visibleBoard(board, visibleRows) {
  if (!Array.isArray(board)) {
    return null;
  }

  return board.slice(board.length - visibleRows).map((row) => row.slice());
}


/* ==================================================
   MatchManager
================================================== */

class MatchManager {
  constructor({ io, puzzleLoader, matchDir }) {
    this.io = io;
    this.puzzleLoader = puzzleLoader;
    this.matchDir = matchDir;

    // roomId → room
    this.rooms = new Map();

    // socketId → { roomId, token }
    this.socketToPlayer = new Map();

    // 自動マッチング待機列（モードごとに組み合わせる）
    this.waiting = [];

    this.sweepTimer = setInterval(() => this.sweepRooms(), SWEEP_INTERVAL_MS);
    this.sweepTimer.unref();
  }

  get roomCount() {
    return this.rooms.size;
  }

  get waitingCount() {
    return this.waiting.length;
  }


  /* ================================================
     プレイヤー
  ================================================ */

  newPlayer(socket, name, mode) {
    return {
      token: playerToken(),
      socketId: socket.id,
      name: cleanName(name),
      mode,
      connected: true,
      disconnectedAt: null,

      levelIndex: 0,
      puzzle: null,
      board: null,
      usedWords: new Set(),
      totalCells: 0,
      remaining: 0,

      baseScore: 0,
      clearedCount: 0,
      moveCount: 0,
      scoreReachedAt: null,

      rematch: false,
    };
  }

  scoreOf(player) {
    if (!player.puzzle) {
      return player.baseScore;
    }

    const clearedCells = player.totalCells - player.remaining;

    return player.baseScore + clearedCells * PARTIAL_POINTS_PER_CELL;
  }

  progressOf(player) {
    return {
      level: player.levelIndex + 1,
      score: this.scoreOf(player),
      remaining: player.remaining,
      totalCells: player.totalCells,
      moveCount: player.moveCount,
      clearedCount: player.clearedCount,
    };
  }

  // 問題を、そのプレイヤーの現在のレベルにする
  setPlayerLevel(player, room, levelIndex) {
    const puzzle = this.getPuzzle(room, levelIndex);

    player.levelIndex = levelIndex;
    player.puzzle = puzzle;
    player.board = cloneBoard(puzzle.board);
    player.usedWords = new Set();
    player.totalCells = puzzle.rows * puzzle.cols;
    player.remaining = player.totalCells;
  }

  publicPuzzle(player) {
    const puzzle = player.puzzle;

    return {
      id: puzzle.id,
      level: player.levelIndex + 1,
      cols: puzzle.cols,
      rows: puzzle.rows,
      visibleRows: puzzle.visibleRows,
      visibleStartRow: puzzle.rows - puzzle.visibleRows,
      board: visibleBoard(player.board, puzzle.visibleRows),
      totalCells: player.totalCells,
      remaining: player.remaining,
      clearPoints: clearPoints(player.levelIndex + 1),
      targetWildcards: puzzle.targetWildcards,
    };
  }


  /* ================================================
     部屋の公開情報
  ================================================ */

  publicState(room, viewerToken) {
    return {
      roomId: room.id,
      mode: room.mode,
      modeLabel: MODES[room.mode].label,
      status: room.status,
      startAt: room.startAt,
      endAt: room.endAt,
      serverTime: Date.now(),
      winnerToken: room.winnerToken,
      players: [...room.players.values()].map((player) => ({
        token: player.token,
        name: player.name,
        connected: player.connected,
        isYou: player.token === viewerToken,
        ...this.progressOf(player),
      })),
    };
  }

  emitState(room) {
    for (const player of room.players.values()) {
      if (player.socketId) {
        this.io
          .to(player.socketId)
          .emit("room:state", this.publicState(room, player.token));
      }
    }
  }

  getPlayerContext(socket) {
    const reference = this.socketToPlayer.get(socket.id);

    if (!reference) {
      return { reference: null, room: null, player: null };
    }

    const room = this.rooms.get(reference.roomId);
    const player = room?.players.get(reference.token) || null;

    return { reference, room: room || null, player };
  }


  /* ================================================
     マッチング
  ================================================ */

  removeWaitingSocket(socketId) {
    const index = this.waiting.findIndex((entry) => entry.socketId === socketId);

    if (index < 0) {
      return null;
    }

    return this.waiting.splice(index, 1)[0];
  }

  joinMatchmaking(socket, payload = {}, ack) {
    if (this.socketToPlayer.has(socket.id)) {
      return ackSafe(ack, { ok: false, error: "すでに対戦へ参加しています" });
    }

    const mode = String(payload.mode || DEFAULT_MODE);

    if (!this.puzzleLoader.hasMode(mode)) {
      return ackSafe(ack, {
        ok: false,
        error: "選択したモードは現在遊べません",
      });
    }

    const waiting = this.waiting.find((entry) => entry.socketId === socket.id);

    if (waiting) {
      return ackSafe(ack, {
        ok: true,
        waiting: true,
        playerToken: waiting.token,
        mode: waiting.mode,
      });
    }

    const player = this.newPlayer(socket, payload.name, mode);

    this.waiting.push(player);

    ackSafe(ack, { ok: true, waiting: true, playerToken: player.token, mode });

    socket.emit("matchmaking:waiting", {
      mode,
      modeLabel: MODES[mode].label,
      waitingCount: this.waiting.filter((entry) => entry.mode === mode).length,
    });

    this.matchNextPair();
  }

  cancelMatchmaking(socket, ack) {
    const removed = this.removeWaitingSocket(socket.id);

    ackSafe(ack, { ok: true, cancelled: Boolean(removed) });
    socket.emit("matchmaking:cancelled");
  }

  // 同じモードで待っている2人を組みます。
  matchNextPair() {
    this.waiting = this.waiting.filter(
      (player) =>
        player.connected &&
        player.socketId &&
        this.io.sockets.sockets.has(player.socketId)
    );

    const remaining = [];
    const byMode = new Map();

    for (const player of this.waiting) {
      if (!byMode.has(player.mode)) {
        byMode.set(player.mode, []);
      }

      const list = byMode.get(player.mode);

      list.push(player);

      if (list.length === 2) {
        byMode.set(player.mode, []);
        this.createMatchedRoom(list[0], list[1]);
      }
    }

    for (const list of byMode.values()) {
      remaining.push(...list);
    }

    this.waiting = remaining;
  }

  createMatchedRoom(first, second) {
    let id;

    do {
      id = roomCode();
    } while (this.rooms.has(id));

    const room = {
      id,
      mode: first.mode,
      status: "matched",
      players: new Map([
        [first.token, first],
        [second.token, second],
      ]),
      sequence: [],
      previousIds: new Set(),
      startAt: null,
      endAt: null,
      winnerToken: null,
      result: null,
      countdownTimer: null,
      endTimer: null,
      createdAt: Date.now(),
    };

    this.rooms.set(id, room);

    for (const player of room.players.values()) {
      this.socketToPlayer.set(player.socketId, { roomId: id, token: player.token });

      this.io.sockets.sockets.get(player.socketId)?.join(id);
    }

    for (const player of room.players.values()) {
      const opponent = [...room.players.values()].find(
        (candidate) => candidate.token !== player.token
      );

      this.io.to(player.socketId).emit("matchmaking:matched", {
        roomId: id,
        mode: room.mode,
        modeLabel: MODES[room.mode].label,
        playerToken: player.token,
        opponentName: opponent?.name || "対戦相手",
        state: this.publicState(room, player.token),
      });
    }

    this.startCountdown(room);
  }


  /* ================================================
     問題列
  ================================================ */

  // 2人が同じ順番で同じ問題を解けるよう、部屋ごとに問題列を持つ
  getPuzzle(room, levelIndex) {
    while (room.sequence.length <= levelIndex) {
      const level = room.sequence.length + 1;
      const exclude = new Set([
        ...room.previousIds,
        ...room.sequence.map((puzzle) => puzzle.id),
      ]);

      room.sequence.push(this.puzzleLoader.pick(room.mode, level, exclude));
    }

    return room.sequence[levelIndex];
  }

  /* ================================================
     3秒カウントダウン → 開始 → 制限時間
  ================================================ */

  startCountdown(room) {
    if (!room) {
      return;
    }

    this.clearRoomTimers(room);

    // 前回の問題を避けて、今回の問題列を作る
    room.previousIds = new Set(room.sequence.map((puzzle) => puzzle.id));
    room.sequence = [];

    room.status = "preparing";
    room.winnerToken = null;
    room.result = null;

    for (let index = 0; index < LEVELS.length; index += 1) {
      this.getPuzzle(room, index);
    }

    room.startAt = Date.now() + COUNTDOWN_MS;
    room.endAt = room.startAt + MATCH_DURATION_MS;
    room.status = "countdown";

    for (const player of room.players.values()) {
      player.baseScore = 0;
      player.clearedCount = 0;
      player.moveCount = 0;
      player.scoreReachedAt = null;
      player.rematch = false;

      this.setPlayerLevel(player, room, 0);

      if (player.connected && player.socketId) {
        this.io.to(player.socketId).emit("match:countdown", {
          roomId: room.id,
          mode: room.mode,
          modeLabel: MODES[room.mode].label,
          serverTime: Date.now(),
          startAt: room.startAt,
          endAt: room.endAt,
          durationMs: MATCH_DURATION_MS,
          countdownSeconds: COUNTDOWN_MS / 1000,
          puzzle: this.publicPuzzle(player),
          progress: this.progressOf(player),
        });
      }
    }

    this.emitState(room);

    room.countdownTimer = setTimeout(
      () => this.startPlaying(room),
      Math.max(0, room.startAt - Date.now())
    );
  }

  startPlaying(room) {
    room.countdownTimer = null;

    if (room.status !== "countdown") {
      return;
    }

    const connected = [...room.players.values()].filter(
      (player) => player.connected && player.socketId
    );

    if (connected.length < 2) {
      room.status = "waiting";

      this.io.to(room.id).emit("match:error", {
        error: "対戦相手との接続が切れたため、開始できませんでした。",
      });

      this.emitState(room);

      return;
    }

    room.status = "playing";

    this.io.to(room.id).emit("match:started", {
      serverTime: Date.now(),
      startAt: room.startAt,
      endAt: room.endAt,
    });

    this.emitState(room);

    room.endTimer = setTimeout(
      () => this.finishByTime(room),
      Math.max(0, room.endAt - Date.now())
    );

    console.log(
      `[match] started room=${room.id} mode=${room.mode} endAt=${room.endAt}`
    );
  }

  clearRoomTimers(room) {
    if (room.countdownTimer) {
      clearTimeout(room.countdownTimer);
      room.countdownTimer = null;
    }

    if (room.endTimer) {
      clearTimeout(room.endTimer);
      room.endTimer = null;
    }
  }


  /* ================================================
     再接続
  ================================================ */

  resumeRoom(socket, payload = {}, ack) {
    const id = String(payload.roomId || "").toUpperCase();
    const room = this.rooms.get(id);
    const player = room?.players.get(String(payload.playerToken || ""));

    if (!room || !player) {
      return ackSafe(ack, { ok: false, error: "復帰情報が無効です" });
    }

    if (player.socketId) {
      this.socketToPlayer.delete(player.socketId);
    }

    player.socketId = socket.id;
    player.connected = true;
    player.disconnectedAt = null;

    this.socketToPlayer.set(socket.id, { roomId: room.id, token: player.token });

    socket.join(room.id);

    ackSafe(ack, {
      ok: true,
      mode: room.mode,
      modeLabel: MODES[room.mode].label,
      state: this.publicState(room, player.token),
      puzzle: player.puzzle ? this.publicPuzzle(player) : null,
      progress: this.progressOf(player),
      startAt: room.startAt,
      endAt: room.endAt,
      serverTime: Date.now(),
      result: room.result,
    });

    this.emitState(room);
  }


  /* ================================================
     盤面リセット（今のレベルを最初からやり直す）
  ================================================ */

  resetBoard(socket, ack) {
    const { room, player } = this.getPlayerContext(socket);

    if (!room || !player) {
      return ackSafe(ack, { ok: false, error: "対戦に参加していません" });
    }

    if (room.status !== "playing" || Date.now() >= room.endAt) {
      return ackSafe(ack, {
        ok: false,
        error: "対戦中のみ盤面をリセットできます",
      });
    }

    // レベルと得点（全消し済みのレベル分）はそのまま、今の盤面だけ戻す
    player.board = cloneBoard(player.puzzle.board);
    player.usedWords = new Set();
    player.remaining = player.totalCells;

    ackSafe(ack, {
      ok: true,
      board: visibleBoard(player.board, player.puzzle.visibleRows),
      progress: this.progressOf(player),
    });

    socket.to(room.id).emit("opponent:progress", this.progressOf(player));

    this.emitState(room);
  }


  /* ================================================
     回答
  ================================================ */

  submitMove(socket, payload = {}, ack) {
    const { room, player } = this.getPlayerContext(socket);

    if (!room || !player) {
      return ackSafe(ack, { ok: false, error: "対戦に参加していません" });
    }

    if (room.status !== "playing" || !room.startAt || Date.now() < room.startAt) {
      return ackSafe(ack, { ok: false, error: "まだ開始していません" });
    }

    if (Date.now() >= room.endAt) {
      return ackSafe(ack, { ok: false, error: "時間切れです" });
    }

    const puzzle = player.puzzle;

    const result = validateAndApply({
      board: player.board,
      path: payload.path,
      fText: String(payload.fText || ""),
      usedWords: player.usedWords,
      isWord: (word) => this.puzzleLoader.isWord(room.mode, word),
      visibleRows: puzzle.visibleRows,
    });

    if (!result.ok) {
      return ackSafe(ack, result);
    }

    player.moveCount += 1;
    player.remaining = result.remaining;

    const clearedLevel = player.levelIndex + 1;
    let gained = null;

    if (result.cleared) {
      gained = clearPoints(clearedLevel);

      player.baseScore += gained;
      player.clearedCount += 1;

      // 全消し: すぐに次のレベルへ
      this.setPlayerLevel(player, room, player.levelIndex + 1);
    }

    // 点数が増えたので、この時刻を同点のときの判定に使う
    player.scoreReachedAt = Date.now();

    ackSafe(ack, {
      ok: true,
      word: result.word,
      levelCleared: result.cleared,
      clearedLevel: result.cleared ? clearedLevel : null,
      gained,
      board: visibleBoard(player.board, player.puzzle.visibleRows),
      puzzle: result.cleared ? this.publicPuzzle(player) : null,
      progress: this.progressOf(player),
      serverTime: Date.now(),
    });

    socket.to(room.id).emit("opponent:progress", {
      ...this.progressOf(player),
      levelCleared: result.cleared,
    });
  }


  /* ================================================
     対戦終了（時間切れ）
  ================================================ */

  decideWinner(players) {
    const ranked = players
      .map((player) => ({
        player,
        score: this.scoreOf(player),
        reachedAt: player.scoreReachedAt ?? Infinity,
      }))
      .sort((left, right) => {
        if (right.score !== left.score) {
          return right.score - left.score;
        }

        return left.reachedAt - right.reachedAt;
      });

    if (ranked.length === 0) {
      return null;
    }

    if (ranked.length === 1) {
      return ranked[0].player;
    }

    const [first, second] = ranked;

    // 完全に同じ（0点どうしを含む）なら引き分け
    if (first.score === second.score && first.reachedAt === second.reachedAt) {
      return null;
    }

    return first.player;
  }

  finishByTime(room) {
    room.endTimer = null;

    if (room.status !== "playing") {
      return;
    }

    room.status = "finished";

    const players = [...room.players.values()];
    const winner = this.decideWinner(players);

    room.winnerToken = winner ? winner.token : null;

    room.result = {
      reason: "time",
      draw: !winner,
      winnerToken: room.winnerToken,
      winnerName: winner ? winner.name : null,
      players: players
        .map((player) => ({
          token: player.token,
          name: player.name,
          score: this.scoreOf(player),
          level: player.levelIndex + 1,
          clearedCount: player.clearedCount,
          moveCount: player.moveCount,
        }))
        .sort((left, right) => right.score - left.score),
    };

    this.io.to(room.id).emit("match:finished", room.result);

    this.save(room);
    this.emitState(room);
  }


  /* ================================================
     再戦
  ================================================ */

  requestRematch(socket, ack) {
    const { room, player } = this.getPlayerContext(socket);

    if (!room || !player || room.status !== "finished") {
      return ackSafe(ack, { ok: false, error: "再戦できません" });
    }

    player.rematch = true;

    ackSafe(ack, { ok: true });

    socket.emit("match:rematch-waiting");

    const everyone = [...room.players.values()];

    if (
      everyone.length >= 2 &&
      everyone.every((entry) => entry.rematch && entry.connected)
    ) {
      // 同じモードのまま、新しい問題列で3秒後に開始する
      this.startCountdown(room);
    }
  }


  /* ================================================
     退出・切断
  ================================================ */

  leaveRoom(socket, ack) {
    this.removeSocket(socket.id, true);

    ackSafe(ack, { ok: true });
  }

  disconnect(socket) {
    this.removeWaitingSocket(socket.id);
    this.removeSocket(socket.id, false);
  }

  removeSocket(socketId, permanent) {
    const reference = this.socketToPlayer.get(socketId);

    if (!reference) {
      return;
    }

    this.socketToPlayer.delete(socketId);

    const room = this.rooms.get(reference.roomId);
    const player = room?.players.get(reference.token);

    if (!room || !player) {
      return;
    }

    player.connected = false;
    player.socketId = null;
    player.disconnectedAt = Date.now();

    if (permanent) {
      room.players.delete(player.token);
    }

    if (room.players.size === 0) {
      this.deleteRoom(room);

      return;
    }

    this.emitState(room);
  }

  deleteRoom(room) {
    this.clearRoomTimers(room);
    this.rooms.delete(room.id);
  }

  // 全員が切断したまま放置された部屋を片付けます。
  sweepRooms() {
    const now = Date.now();

    for (const room of [...this.rooms.values()]) {
      const players = [...room.players.values()];

      const abandoned =
        players.length > 0 &&
        players.every(
          (player) =>
            !player.connected &&
            player.disconnectedAt &&
            now - player.disconnectedAt > ABANDONED_ROOM_MS
        );

      if (abandoned) {
        this.deleteRoom(room);
      }
    }
  }


  /* ================================================
     対戦結果の保存
  ================================================ */

  save(room) {
    try {
      fs.mkdirSync(this.matchDir, { recursive: true });

      const output = {
        format: "keshimasu-duel-result-v7-score-attack",
        roomId: room.id,
        mode: room.mode,
        puzzleIds: room.sequence.map((puzzle) => puzzle.id),
        startedAt: new Date(room.startAt).toISOString(),
        finishedAt: new Date().toISOString(),
        winnerToken: room.winnerToken,
        draw: room.result?.draw ?? false,
        players: room.result?.players ?? [],
      };

      fs.writeFileSync(
        path.join(this.matchDir, `${Date.now()}-${room.id}.json`),
        JSON.stringify(output, null, 2),
        "utf8"
      );
    } catch (error) {
      console.warn("[match save] failed:", error.message);
    }
  }
}

module.exports = {
  MatchManager,
};
