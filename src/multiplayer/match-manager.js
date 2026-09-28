"use strict";

const fs = require("node:fs");
const path = require("node:path");

const {
  roomCode,
  playerToken,
} = require("./room-code");

const {
  cloneBoard,
} = require("./puzzle-loader");

const {
  validateAndApply,
} = require("./answer-validator");


const ROWS = 8;
const COLS = 5;
const PLAYABLE_START_ROW = 3;


function cleanName(value) {
  return (
    String(value || "プレイヤー")
      .trim()
      .slice(0, 20) ||
    "プレイヤー"
  );
}


function ackSafe(ack, data) {
  if (typeof ack === "function") {
    ack(data);
  }
}


function visibleBoard(board) {
  if (!Array.isArray(board)) {
    return null;
  }

  return board
    .slice(PLAYABLE_START_ROW, ROWS)
    .map((row) => row.slice(0, COLS));
}


class MatchManager {
  constructor({
    io,
    puzzleLoader,
    matchDir,
  }) {
    this.io = io;
    this.puzzleLoader = puzzleLoader;
    this.matchDir = matchDir;

    this.rooms = new Map();
    this.socketToPlayer = new Map();
    this.waiting = [];
  }


  get roomCount() {
    return this.rooms.size;
  }


  get waitingCount() {
    return this.waiting.length;
  }


  newPlayer(socket, name) {
    return {
      token: playerToken(),
      socketId: socket.id,
      name: cleanName(name),

      connected: true,

      board: null,
      usedWords: new Set(),

      remaining: 40,
      moveCount: 0,

      finishedAt: null,
      elapsedMs: null,

      rematch: false,
    };
  }


  publicState(room, viewerToken) {
    return {
      roomId: room.id,
      status: room.status,
      puzzleId: room.puzzle?.id ?? null,
      startAt: room.startAt,
      winnerToken: room.winnerToken,

      players: [...room.players.values()].map(
        (player) => ({
          token: player.token,
          name: player.name,
          connected: player.connected,
          remaining: player.remaining,
          moveCount: player.moveCount,
          finishedAt: player.finishedAt,
          isYou: player.token === viewerToken,
        })
      ),
    };
  }


  emitState(room) {
    for (const player of room.players.values()) {
      if (!player.socketId) {
        continue;
      }

      this.io
        .to(player.socketId)
        .emit(
          "room:state",
          this.publicState(room, player.token)
        );
    }
  }


  removeWaitingSocket(socketId) {
    const index = this.waiting.findIndex(
      (entry) => entry.socketId === socketId
    );

    if (index < 0) {
      return null;
    }

    return this.waiting.splice(index, 1)[0];
  }


  joinMatchmaking(socket, payload = {}, ack) {
    if (this.socketToPlayer.has(socket.id)) {
      return ackSafe(ack, {
        ok: false,
        error: "すでに対戦へ参加しています",
      });
    }

    if (
      this.waiting.some(
        (entry) => entry.socketId === socket.id
      )
    ) {
      return ackSafe(ack, {
        ok: true,
        waiting: true,
      });
    }

    const player = this.newPlayer(
      socket,
      payload.name
    );

    this.waiting.push(player);

    ackSafe(ack, {
      ok: true,
      waiting: true,
      playerToken: player.token,
    });

    socket.emit("matchmaking:waiting", {
      waitingCount: this.waiting.length,
    });

    this.matchNextPair();
  }


  cancelMatchmaking(socket, ack) {
    const removed =
      this.removeWaitingSocket(socket.id);

    ackSafe(ack, {
      ok: true,
      cancelled: Boolean(removed),
    });

    socket.emit("matchmaking:cancelled");
  }


  matchNextPair() {
    this.waiting = this.waiting.filter(
      (player) =>
        player.connected &&
        this.io.sockets.sockets.has(
          player.socketId
        )
    );

    while (this.waiting.length >= 2) {
      const first = this.waiting.shift();
      const second = this.waiting.shift();

      if (!first || !second) {
        return;
      }

      this.createMatchedRoom(first, second);
    }
  }


  createMatchedRoom(first, second) {
    let id;

    do {
      id = roomCode();
    } while (this.rooms.has(id));

    const room = {
      id,
      status: "matched",

      players: new Map([
        [first.token, first],
        [second.token, second],
      ]),

      puzzle: null,
      startAt: null,
      winnerToken: null,
      previousPuzzleId: null,

      createdAt: Date.now(),
    };

    this.rooms.set(id, room);

    for (const player of room.players.values()) {
      this.socketToPlayer.set(
        player.socketId,
        {
          roomId: id,
          token: player.token,
        }
      );

      const socket =
        this.io.sockets.sockets.get(
          player.socketId
        );

      if (socket) {
        socket.join(id);
      }
    }

    for (const player of room.players.values()) {
      const opponent =
        [...room.players.values()].find(
          (candidate) =>
            candidate.token !== player.token
        );

      this.io
        .to(player.socketId)
        .emit("matchmaking:matched", {
          roomId: id,
          playerToken: player.token,
          opponentName:
            opponent?.name || "対戦相手",

          state: this.publicState(
            room,
            player.token
          ),
        });
    }

    this.startCountdown(room);
  }


  startCountdown(room) {
    room.puzzle =
      this.puzzleLoader.random(
        room.previousPuzzleId
      );

    room.previousPuzzleId =
      room.puzzle.id;

    room.status = "countdown";
    room.startAt = Date.now() + 3000;
    room.winnerToken = null;

    for (const player of room.players.values()) {
      player.board =
        cloneBoard(room.puzzle.board);

      player.usedWords = new Set();

      player.remaining = 40;
      player.moveCount = 0;

      player.finishedAt = null;
      player.elapsedMs = null;
      player.rematch = false;

      this.io
        .to(player.socketId)
        .emit("match:countdown", {
          roomId: room.id,
          startAt: room.startAt,

          puzzle: {
            id: room.puzzle.id,

            // 上3行はブラウザへ送信しない。
            board: visibleBoard(player.board),

            visibleStartRow:
              PLAYABLE_START_ROW,

            visibleRows: 5,
            cols: COLS,

            targetWildcards:
              room.puzzle.targetWildcards,
          },
        });
    }

    this.emitState(room);

    setTimeout(() => {
      if (room.status !== "countdown") {
        return;
      }

      room.status = "playing";

      this.io
        .to(room.id)
        .emit("match:started", {
          startAt: room.startAt,
        });

      this.emitState(room);
    }, Math.max(
      0,
      room.startAt - Date.now()
    ));
  }


  resumeRoom(socket, payload = {}, ack) {
    const id = String(
      payload.roomId || ""
    ).toUpperCase();

    const room = this.rooms.get(id);

    const player = room?.players.get(
      String(payload.playerToken || "")
    );

    if (!room || !player) {
      return ackSafe(ack, {
        ok: false,
        error: "復帰情報が無効です",
      });
    }

    player.socketId = socket.id;
    player.connected = true;

    this.socketToPlayer.set(
      socket.id,
      {
        roomId: room.id,
        token: player.token,
      }
    );

    socket.join(room.id);

    ackSafe(ack, {
      ok: true,

      state: this.publicState(
        room,
        player.token
      ),

      // 再接続時も下5行のみ返す。
      board: visibleBoard(player.board),

      startAt: room.startAt,
      visibleStartRow:
        PLAYABLE_START_ROW,
    });

    this.emitState(room);
  }


  submitMove(socket, payload = {}, ack) {
    const reference =
      this.socketToPlayer.get(socket.id);

    const room =
      reference &&
      this.rooms.get(reference.roomId);

    const player =
      room &&
      room.players.get(reference.token);

    if (!room || !player) {
      return ackSafe(ack, {
        ok: false,
        error: "対戦に参加していません",
      });
    }

    if (
      room.status !== "playing" ||
      Date.now() < room.startAt
    ) {
      return ackSafe(ack, {
        ok: false,
        error: "まだ開始していません",
      });
    }

    if (player.finishedAt) {
      return ackSafe(ack, {
        ok: false,
        error: "すでにクリアしています",
      });
    }

    const result = validateAndApply({
      board: player.board,
      path: payload.path,

      // Fの補完文字だけ受け取る。
      fText: String(payload.fText || ""),

      usedWords: player.usedWords,

      isCountry: (word) =>
        this.puzzleLoader.isCountry(word),
    });

    if (!result.ok) {
      return ackSafe(ack, result);
    }

    player.moveCount += 1;
    player.remaining =
      result.remaining;

    ackSafe(ack, {
      ok: true,

      word: result.word,

      // 重力処理後の下5行だけ返す。
      board: visibleBoard(player.board),

      remaining: player.remaining,
      moveCount: player.moveCount,

      visibleStartRow:
        PLAYABLE_START_ROW,
    });

    socket
      .to(room.id)
      .emit("opponent:progress", {
        remaining: player.remaining,
        moveCount: player.moveCount,

        progress:
          ((40 - player.remaining) / 40) *
          100,
      });

    if (result.cleared) {
      this.finish(room, player);
    }
  }


  finish(room, player) {
    player.finishedAt = Date.now();

    player.elapsedMs =
      player.finishedAt - room.startAt;

    if (room.winnerToken) {
      return;
    }

    room.winnerToken = player.token;
    room.status = "finished";

    this.io
      .to(room.id)
      .emit("match:finished", {
        winnerToken: player.token,
        winnerName: player.name,
        elapsedMs: player.elapsedMs,

        players:
          [...room.players.values()].map(
            (entry) => ({
              token: entry.token,
              name: entry.name,
              elapsedMs: entry.elapsedMs,
              moveCount: entry.moveCount,
              remaining: entry.remaining,
            })
          ),
      });

    this.save(room);
    this.emitState(room);
  }


  requestRematch(socket, ack) {
    const reference =
      this.socketToPlayer.get(socket.id);

    const room =
      reference &&
      this.rooms.get(reference.roomId);

    const player =
      room &&
      room.players.get(reference.token);

    if (
      !player ||
      room.status !== "finished"
    ) {
      return ackSafe(ack, {
        ok: false,
        error: "再戦できません",
      });
    }

    player.rematch = true;

    ackSafe(ack, {
      ok: true,
    });

    socket.emit(
      "match:rematch-waiting"
    );

    const allRematch =
      [...room.players.values()].every(
        (entry) =>
          entry.rematch &&
          entry.connected
      );

    if (allRematch) {
      this.startCountdown(room);
    }
  }


  leaveRoom(socket, ack) {
    this.removeSocket(
      socket.id,
      true
    );

    ackSafe(ack, {
      ok: true,
    });
  }


  disconnect(socket) {
    this.removeWaitingSocket(socket.id);

    this.removeSocket(
      socket.id,
      false
    );
  }


  removeSocket(socketId, permanent) {
    const reference =
      this.socketToPlayer.get(socketId);

    if (!reference) {
      return;
    }

    this.socketToPlayer.delete(socketId);

    const room =
      this.rooms.get(reference.roomId);

    const player =
      room?.players.get(reference.token);

    if (!room || !player) {
      return;
    }

    player.connected = false;
    player.socketId = null;

    if (permanent) {
      room.players.delete(player.token);
    }

    if (room.players.size === 0) {
      this.rooms.delete(room.id);
    } else {
      this.emitState(room);
    }
  }


  save(room) {
    fs.mkdirSync(
      this.matchDir,
      {
        recursive: true,
      }
    );

    const output = {
      format:
        "keshimasu-duel-result-v3-hidden-rows",

      roomId: room.id,
      puzzleId: room.puzzle.id,

      startedAt:
        new Date(
          room.startAt
        ).toISOString(),

      finishedAt:
        new Date().toISOString(),

      winnerToken:
        room.winnerToken,

      players:
        [...room.players.values()].map(
          (player) => ({
            token: player.token,
            name: player.name,
            elapsedMs: player.elapsedMs,
            moveCount: player.moveCount,
            remaining: player.remaining,
          })
        ),
    };

    const filename =
      `${Date.now()}-${room.id}.json`;

    fs.writeFileSync(
      path.join(
        this.matchDir,
        filename
      ),
      JSON.stringify(
        output,
        null,
        2
      ),
      "utf8"
    );
  }
}


module.exports = {
  MatchManager,
};