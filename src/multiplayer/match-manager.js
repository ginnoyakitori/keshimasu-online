"use strict";


/* ==================================================
   モジュール
================================================== */

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


/* ==================================================
   盤面・対戦設定
================================================== */

const ROWS = 8;
const COLS = 5;
const VISIBLE_ROWS = 5;

const PLAYABLE_START_ROW =
  ROWS - VISIBLE_ROWS;

const COUNTDOWN_MS = 3000;
const INITIAL_CELL_COUNT = 40;


/* ==================================================
   共通関数
================================================== */

function cleanName(value) {
  return (
    String(
      value || "プレイヤー"
    )
      .trim()
      .slice(0, 20) ||
    "プレイヤー"
  );
}


function ackSafe(
  ack,
  response
) {
  if (
    typeof ack === "function"
  ) {
    ack(response);
  }
}


/*
 * サーバー内部では8行×5列を保持します。
 *
 * ブラウザへ送るのは下側の5行だけです。
 */
function visibleBoard(board) {
  if (!Array.isArray(board)) {
    return null;
  }

  return board
    .slice(
      PLAYABLE_START_ROW,
      ROWS
    )
    .map(
      (row) =>
        row.slice(0, COLS)
    );
}


/* ==================================================
   MatchManager
================================================== */

class MatchManager {
  constructor({
    io,
    puzzleLoader,
    matchDir,
  }) {
    this.io = io;

    this.puzzleLoader =
      puzzleLoader;

    this.matchDir =
      matchDir;

    /*
     * 対戦中の部屋です。
     *
     * roomId → room
     */
    this.rooms = new Map();

    /*
     * Socket.IOの接続IDから、
     * 参加中の部屋とプレイヤーを取得します。
     *
     * socketId → {
     *   roomId,
     *   token
     * }
     */
    this.socketToPlayer =
      new Map();

    /*
     * 自動マッチング待機列です。
     */
    this.waiting = [];
  }


  /* ==================================================
     公開状態
  ================================================== */

  get roomCount() {
    return this.rooms.size;
  }


  get waitingCount() {
    return this.waiting.length;
  }


  /* ==================================================
     プレイヤー作成
  ================================================== */

  newPlayer(
    socket,
    name
  ) {
    return {
      token:
        playerToken(),

      socketId:
        socket.id,

      name:
        cleanName(name),

      connected:
        true,

      /*
       * このプレイヤー専用の
       * 8行×5列盤面です。
       */
      board:
        null,

      /*
       * この問題ですでに使用した国名です。
       */
      usedWords:
        new Set(),

      remaining:
        INITIAL_CELL_COUNT,

      moveCount:
        0,

      finishedAt:
        null,

      elapsedMs:
        null,

      rematch:
        false,
    };
  }


  /* ==================================================
     部屋の公開情報
  ================================================== */

  publicState(
    room,
    viewerToken
  ) {
    return {
      roomId:
        room.id,

      status:
        room.status,

      puzzleId:
        room.puzzle?.id ??
        null,

      startAt:
        room.startAt,

      winnerToken:
        room.winnerToken,

      players:
        [...room.players.values()]
          .map(
            (player) => ({
              token:
                player.token,

              name:
                player.name,

              connected:
                player.connected,

              remaining:
                player.remaining,

              moveCount:
                player.moveCount,

              finishedAt:
                player.finishedAt,

              isYou:
                player.token ===
                viewerToken,
            })
          ),
    };
  }


  emitState(room) {
    for (
      const player
      of room.players.values()
    ) {
      if (!player.socketId) {
        continue;
      }

      this.io
        .to(player.socketId)
        .emit(
          "room:state",
          this.publicState(
            room,
            player.token
          )
        );
    }
  }


  /* ==================================================
     プレイヤー取得
  ================================================== */

  getPlayerContext(socket) {
    const reference =
      this.socketToPlayer.get(
        socket.id
      );

    if (!reference) {
      return {
        reference: null,
        room: null,
        player: null,
      };
    }

    const room =
      this.rooms.get(
        reference.roomId
      );

    const player =
      room?.players.get(
        reference.token
      ) || null;

    return {
      reference,
      room: room || null,
      player,
    };
  }


  /* ==================================================
     マッチング待機列
  ================================================== */

  removeWaitingSocket(
    socketId
  ) {
    const index =
      this.waiting.findIndex(
        (entry) =>
          entry.socketId ===
          socketId
      );

    if (index < 0) {
      return null;
    }

    return this.waiting.splice(
      index,
      1
    )[0];
  }


  joinMatchmaking(
    socket,
    payload = {},
    ack
  ) {
    /*
     * すでに部屋へ参加している場合は
     * 二重参加させません。
     */
    if (
      this.socketToPlayer.has(
        socket.id
      )
    ) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "すでに対戦へ参加しています",
        }
      );
    }

    /*
     * すでに待機中の場合です。
     */
    const alreadyWaiting =
      this.waiting.some(
        (entry) =>
          entry.socketId ===
          socket.id
      );

    if (alreadyWaiting) {
      return ackSafe(
        ack,
        {
          ok: true,
          waiting: true,
        }
      );
    }

    const player =
      this.newPlayer(
        socket,
        payload.name
      );

    this.waiting.push(
      player
    );

    ackSafe(
      ack,
      {
        ok: true,
        waiting: true,
        playerToken:
          player.token,
      }
    );

    socket.emit(
      "matchmaking:waiting",
      {
        waitingCount:
          this.waiting.length,
      }
    );

    this.matchNextPair();
  }


  cancelMatchmaking(
    socket,
    ack
  ) {
    const removed =
      this.removeWaitingSocket(
        socket.id
      );

    ackSafe(
      ack,
      {
        ok: true,
        cancelled:
          Boolean(removed),
      }
    );

    socket.emit(
      "matchmaking:cancelled"
    );
  }


  matchNextPair() {
    /*
     * 切断済みプレイヤーを
     * 待機列から除外します。
     */
    this.waiting =
      this.waiting.filter(
        (player) =>
          player.connected &&
          player.socketId &&
          this.io.sockets
            .sockets
            .has(
              player.socketId
            )
      );

    while (
      this.waiting.length >= 2
    ) {
      const first =
        this.waiting.shift();

      const second =
        this.waiting.shift();

      if (!first || !second) {
        return;
      }

      this.createMatchedRoom(
        first,
        second
      );
    }
  }


  /* ==================================================
     部屋作成
  ================================================== */

  createMatchedRoom(
    first,
    second
  ) {
    let id;

    do {
      id = roomCode();
    } while (
      this.rooms.has(id)
    );

    const room = {
      id,

      status:
        "matched",

      players:
        new Map([
          [
            first.token,
            first,
          ],
          [
            second.token,
            second,
          ],
        ]),

      puzzle:
        null,

      startAt:
        null,

      winnerToken:
        null,

      previousPuzzleId:
        null,

      countdownTimer:
        null,

      createdAt:
        Date.now(),
    };

    this.rooms.set(
      id,
      room
    );

    /*
     * 両プレイヤーを同じ
     * Socket.IOルームへ参加させます。
     */
    for (
      const player
      of room.players.values()
    ) {
      this.socketToPlayer.set(
        player.socketId,
        {
          roomId:
            id,

          token:
            player.token,
        }
      );

      const playerSocket =
        this.io.sockets
          .sockets
          .get(
            player.socketId
          );

      if (playerSocket) {
        playerSocket.join(id);
      }
    }

    /*
     * 両プレイヤーへ
     * マッチング成立を通知します。
     */
    for (
      const player
      of room.players.values()
    ) {
      const opponent =
        [...room.players.values()]
          .find(
            (candidate) =>
              candidate.token !==
              player.token
          );

      this.io
        .to(player.socketId)
        .emit(
          "matchmaking:matched",
          {
            roomId:
              id,

            playerToken:
              player.token,

            opponentName:
              opponent?.name ||
              "対戦相手",

            state:
              this.publicState(
                room,
                player.token
              ),
          }
        );
    }

    /*
     * マッチング成立後、
     * すぐに問題を選択して
     * 3秒カウントダウンを開始します。
     */
    this.startCountdown(room);
  }


  /* ==================================================
     問題選択
  ================================================== */

  choosePuzzle(
    previousPuzzleId = null
  ) {
    /*
     * 4,000問の確認済みJSONから
     * 即座にランダム選択します。
     */
    const puzzle =
      this.puzzleLoader.random(
        previousPuzzleId
      );

    console.log(
      `[puzzle] selected ` +
      `${puzzle.id} ` +
      `from hard_puzzle_boards.json`
    );

    return {
      ...puzzle,

      source:
        puzzle.source ||
        "hard_puzzle_boards.json",
    };
  }


  /* ==================================================
     3秒カウントダウン
  ================================================== */

  startCountdown(room) {
    if (!room) {
      return;
    }

    /*
     * 以前のカウントダウンが残っている場合は
     * 停止します。
     */
    if (
      room.countdownTimer
    ) {
      clearTimeout(
        room.countdownTimer
      );

      room.countdownTimer =
        null;
    }

    room.status =
      "preparing";

    room.startAt =
      null;

    room.winnerToken =
      null;

    this.emitState(room);

    /*
     * 問題を即時選択します。
     */
    room.puzzle =
      this.choosePuzzle(
        room.previousPuzzleId
      );

    room.previousPuzzleId =
      room.puzzle.id;

    /*
     * 問題選択完了から3秒後を
     * 開始時刻とします。
     */
    room.status =
      "countdown";

    room.startAt =
      Date.now() +
      COUNTDOWN_MS;

    /*
     * 両プレイヤーへ
     * 同じ初期盤面を設定します。
     *
     * boardはプレイヤーごとに
     * 独立したコピーです。
     */
    for (
      const player
      of room.players.values()
    ) {
      player.board =
        cloneBoard(
          room.puzzle.board
        );

      player.usedWords =
        new Set();

      player.remaining =
        INITIAL_CELL_COUNT;

      player.moveCount =
        0;

      player.finishedAt =
        null;

      player.elapsedMs =
        null;

      player.rematch =
        false;

      if (
        player.connected &&
        player.socketId
      ) {
        this.io
          .to(player.socketId)
          .emit(
            "match:countdown",
            {
              roomId:
                room.id,

              startAt:
                room.startAt,

              countdownSeconds:
                3,

              puzzle: {
                id:
                  room.puzzle.id,

                /*
                 * 上3行は送信しません。
                 */
                board:
                  visibleBoard(
                    player.board
                  ),

                visibleStartRow:
                  PLAYABLE_START_ROW,

                visibleRows:
                  VISIBLE_ROWS,

                cols:
                  COLS,

                targetWildcards:
                  room.puzzle
                    .targetWildcards,
              },
            }
          );
      }
    }

    this.emitState(room);

    const delay =
      Math.max(
        0,
        room.startAt -
        Date.now()
      );

    room.countdownTimer =
      setTimeout(
        () => {
          room.countdownTimer =
            null;

          if (
            room.status !==
            "countdown"
          ) {
            return;
          }

          const connectedPlayers =
            [...room.players.values()]
              .filter(
                (player) =>
                  player.connected &&
                  player.socketId
              );

          if (
            connectedPlayers.length <
            2
          ) {
            room.status =
              "waiting";

            this.io
              .to(room.id)
              .emit(
                "match:error",
                {
                  error:
                    "対戦相手との接続が切れたため、開始できませんでした。",
                }
              );

            this.emitState(room);
            return;
          }

          room.status =
            "playing";

          this.io
            .to(room.id)
            .emit(
              "match:started",
              {
                startAt:
                  room.startAt,
              }
            );

          this.emitState(room);

          console.log(
            `[match] started ` +
            `room=${room.id} ` +
            `puzzle=${room.puzzle.id} ` +
            `startAt=${room.startAt}`
          );
        },
        delay
      );
  }


  /* ==================================================
     再接続
  ================================================== */

  resumeRoom(
    socket,
    payload = {},
    ack
  ) {
    const id =
      String(
        payload.roomId || ""
      ).toUpperCase();

    const room =
      this.rooms.get(id);

    const player =
      room?.players.get(
        String(
          payload.playerToken ||
          ""
        )
      );

    if (!room || !player) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "復帰情報が無効です",
        }
      );
    }

    /*
     * 古いSocket IDの対応を削除します。
     */
    if (player.socketId) {
      this.socketToPlayer.delete(
        player.socketId
      );
    }

    player.socketId =
      socket.id;

    player.connected =
      true;

    this.socketToPlayer.set(
      socket.id,
      {
        roomId:
          room.id,

        token:
          player.token,
      }
    );

    socket.join(
      room.id
    );

    ackSafe(
      ack,
      {
        ok: true,

        state:
          this.publicState(
            room,
            player.token
          ),

        board:
          visibleBoard(
            player.board
          ),

        startAt:
          room.startAt,

        visibleStartRow:
          PLAYABLE_START_ROW,
      }
    );

    this.emitState(room);
  }


  /* ==================================================
     盤面リセット
  ================================================== */

  resetBoard(
    socket,
    ack
  ) {
    const {
      room,
      player,
    } =
      this.getPlayerContext(
        socket
      );

    if (!room || !player) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "対戦に参加していません",
        }
      );
    }

    /*
     * ゲーム開始後だけ使用できます。
     */
    if (
      room.status !==
      "playing"
    ) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "対戦中のみ盤面をリセットできます",
        }
      );
    }

    if (
      !room.puzzle ||
      !Array.isArray(
        room.puzzle.board
      )
    ) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "最初の盤面を取得できません",
        }
      );
    }

    /*
     * リセットを押したプレイヤーだけを
     * 対戦開始時の盤面へ戻します。
     */
    player.board =
      cloneBoard(
        room.puzzle.board
      );

    /*
     * 使用済み国名も初期化します。
     */
    player.usedWords =
      new Set();

    player.remaining =
      INITIAL_CELL_COUNT;

    player.moveCount =
      0;

    player.finishedAt =
      null;

    player.elapsedMs =
      null;

    /*
     * startAtは変更しません。
     *
     * そのため経過時間は継続します。
     */
    ackSafe(
      ack,
      {
        ok: true,

        board:
          visibleBoard(
            player.board
          ),

        remaining:
          INITIAL_CELL_COUNT,

        moveCount:
          0,

        startAt:
          room.startAt,
      }
    );

    /*
     * 相手側の進捗表示を
     * 残り40マスへ戻します。
     */
    socket
      .to(room.id)
      .emit(
        "opponent:progress",
        {
          remaining:
            INITIAL_CELL_COUNT,

          moveCount:
            0,

          progress:
            0,
        }
      );

    this.emitState(room);

    console.log(
      `[match] board reset ` +
      `room=${room.id} ` +
      `player=${player.name}`
    );
  }


  /* ==================================================
     回答送信
  ================================================== */

  submitMove(
    socket,
    payload = {},
    ack
  ) {
    const {
      room,
      player,
    } =
      this.getPlayerContext(
        socket
      );

    if (!room || !player) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "対戦に参加していません",
        }
      );
    }

    /*
     * カウントダウン終了前の操作を
     * サーバー側でも拒否します。
     */
    if (
      room.status !==
        "playing" ||
      !room.startAt ||
      Date.now() <
        room.startAt
    ) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "まだ開始していません",
        }
      );
    }

    if (player.finishedAt) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "すでにクリアしています",
        }
      );
    }

    /*
     * answer-validator.jsが、
     * 現在盤面と選択座標から
     * 国名を自動構築します。
     *
     * Fがある場合だけfTextを使います。
     */
    const result =
      validateAndApply({
        board:
          player.board,

        path:
          payload.path,

        fText:
          String(
            payload.fText || ""
          ),

        usedWords:
          player.usedWords,

        isCountry: (word) =>
          this.puzzleLoader
            .isCountry(word),
      });

    if (!result.ok) {
      return ackSafe(
        ack,
        result
      );
    }

    player.moveCount += 1;

    player.remaining =
      result.remaining;

    ackSafe(
      ack,
      {
        ok: true,

        word:
          result.word,

        /*
         * 消去・重力処理後の
         * 下5行だけを返します。
         */
        board:
          visibleBoard(
            player.board
          ),

        remaining:
          player.remaining,

        moveCount:
          player.moveCount,

        visibleStartRow:
          PLAYABLE_START_ROW,
      }
    );

    /*
     * 相手には盤面内容を送らず、
     * 進捗だけを送信します。
     */
    socket
      .to(room.id)
      .emit(
        "opponent:progress",
        {
          remaining:
            player.remaining,

          moveCount:
            player.moveCount,

          progress:
            (
              (
                INITIAL_CELL_COUNT -
                player.remaining
              ) /
              INITIAL_CELL_COUNT
            ) *
            100,
        }
      );

    if (result.cleared) {
      this.finish(
        room,
        player
      );
    }
  }


  /* ==================================================
     対戦終了
  ================================================== */

  finish(
    room,
    player
  ) {
    player.finishedAt =
      Date.now();

    player.elapsedMs =
      player.finishedAt -
      room.startAt;

    /*
     * すでに勝者が決まっている場合は
     * 二重終了させません。
     */
    if (room.winnerToken) {
      return;
    }

    room.winnerToken =
      player.token;

    room.status =
      "finished";

    this.io
      .to(room.id)
      .emit(
        "match:finished",
        {
          winnerToken:
            player.token,

          winnerName:
            player.name,

          elapsedMs:
            player.elapsedMs,

          players:
            [...room.players.values()]
              .map(
                (entry) => ({
                  token:
                    entry.token,

                  name:
                    entry.name,

                  elapsedMs:
                    entry.elapsedMs,

                  moveCount:
                    entry.moveCount,

                  remaining:
                    entry.remaining,
                })
              ),
        }
      );

    this.save(room);
    this.emitState(room);
  }


  /* ==================================================
     再戦
  ================================================== */

  requestRematch(
    socket,
    ack
  ) {
    const {
      room,
      player,
    } =
      this.getPlayerContext(
        socket
      );

    if (
      !room ||
      !player ||
      room.status !==
        "finished"
    ) {
      return ackSafe(
        ack,
        {
          ok: false,
          error:
            "再戦できません",
        }
      );
    }

    player.rematch =
      true;

    ackSafe(
      ack,
      {
        ok: true,
      }
    );

    socket.emit(
      "match:rematch-waiting"
    );

    const allRematch =
      [...room.players.values()]
        .every(
          (entry) =>
            entry.rematch &&
            entry.connected
        );

    if (allRematch) {
      /*
       * 再戦時も別の問題を選び、
       * 3秒後に開始します。
       */
      this.startCountdown(room);
    }
  }


  /* ==================================================
     退出・切断
  ================================================== */

  leaveRoom(
    socket,
    ack
  ) {
    this.removeSocket(
      socket.id,
      true
    );

    ackSafe(
      ack,
      {
        ok: true,
      }
    );
  }


  disconnect(socket) {
    this.removeWaitingSocket(
      socket.id
    );

    this.removeSocket(
      socket.id,
      false
    );
  }


  removeSocket(
    socketId,
    permanent
  ) {
    const reference =
      this.socketToPlayer.get(
        socketId
      );

    if (!reference) {
      return;
    }

    this.socketToPlayer.delete(
      socketId
    );

    const room =
      this.rooms.get(
        reference.roomId
      );

    const player =
      room?.players.get(
        reference.token
      );

    if (!room || !player) {
      return;
    }

    player.connected =
      false;

    player.socketId =
      null;

    if (permanent) {
      room.players.delete(
        player.token
      );
    }

    if (
      room.players.size === 0
    ) {
      if (
        room.countdownTimer
      ) {
        clearTimeout(
          room.countdownTimer
        );

        room.countdownTimer =
          null;
      }

      this.rooms.delete(
        room.id
      );

      return;
    }

    this.emitState(room);
  }


  /* ==================================================
     対戦結果保存
  ================================================== */

  save(room) {
    try {
      fs.mkdirSync(
        this.matchDir,
        {
          recursive: true,
        }
      );

      const output = {
        format:
          "keshimasu-duel-result-v6-board-reset",

        roomId:
          room.id,

        puzzleId:
          room.puzzle.id,

        puzzleSource:
          room.puzzle.source,

        targetWildcards:
          room.puzzle
            .targetWildcards,

        startedAt:
          new Date(
            room.startAt
          ).toISOString(),

        finishedAt:
          new Date()
            .toISOString(),

        winnerToken:
          room.winnerToken,

        players:
          [...room.players.values()]
            .map(
              (entry) => ({
                token:
                  entry.token,

                name:
                  entry.name,

                elapsedMs:
                  entry.elapsedMs,

                moveCount:
                  entry.moveCount,

                remaining:
                  entry.remaining,
              })
            ),
      };

      const fileName =
        `${Date.now()}-${room.id}.json`;

      fs.writeFileSync(
        path.join(
          this.matchDir,
          fileName
        ),

        JSON.stringify(
          output,
          null,
          2
        ),

        "utf8"
      );
    } catch (error) {
      console.warn(
        "[match save] failed:",
        error.message
      );
    }
  }
}


/* ==================================================
   エクスポート
================================================== */

module.exports = {
  MatchManager,
};