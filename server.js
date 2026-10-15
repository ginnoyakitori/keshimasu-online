"use strict";

/* ==================================================
   モジュール
================================================== */

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const express = require("express");
const { Server } = require("socket.io");

const { PuzzleLoader } = require("./src/multiplayer/puzzle-loader");
const { MatchManager } = require("./src/multiplayer/match-manager");
const {
  MATCH_DURATION_MS,
  LEVELS,
} = require("./src/multiplayer/levels");


/* ==================================================
   パス・環境変数
================================================== */

const ROOT = __dirname;
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "0.0.0.0";

function resolveProjectPath(environmentValue, defaultRelativePath) {
  const value = environmentValue || defaultRelativePath;

  return path.isAbsolute(value) ? value : path.resolve(ROOT, value);
}

const PUBLIC_DIR = resolveProjectPath(process.env.PUBLIC_DIR, "./public");
const WORDS_DIR = resolveProjectPath(process.env.WORDS_DIR, "./data/words");
const POOLS_DIR = resolveProjectPath(
  process.env.POOLS_DIR,
  "./data/puzzles/pools"
);
const MATCH_DIR = resolveProjectPath(process.env.MATCH_DIR, "./data/matches");


/* ==================================================
   Express・HTTP・Socket.IO
================================================== */

const app = express();
const httpServer = http.createServer(app);

const io = new Server(httpServer, {
  maxHttpBufferSize: 100_000,
  pingInterval: 25_000,
  pingTimeout: 20_000,
  transports: ["polling", "websocket"],
  cors: { origin: false },
});

app.disable("x-powered-by");
app.use(express.json({ limit: "64kb" }));
app.use(express.urlencoded({ extended: false, limit: "64kb" }));

/*
 * public/ (index.html, style.css, client.js) を配信します。
 * index.html だけキャッシュを弱くし、再デプロイ後に古いHTMLが残りにくくします。
 */
app.use(
  express.static(PUBLIC_DIR, {
    extensions: ["html"],
    etag: true,
    maxAge: process.env.NODE_ENV === "production" ? "5m" : 0,
    setHeaders(response, filePath) {
      if (path.basename(filePath) === "index.html") {
        response.setHeader("Cache-Control", "no-cache");
      }
    },
  })
);


/* ==================================================
   問題ローダー・対戦管理
================================================== */

const puzzleLoader = new PuzzleLoader({
  wordsDir: WORDS_DIR,
  poolsDir: POOLS_DIR,
});

const matchManager = new MatchManager({
  io,
  puzzleLoader,
  matchDir: MATCH_DIR,
});


/* ==================================================
   HTTP API
================================================== */

// Renderのヘルスチェック用
app.get("/api/health", (_request, response) => {
  response.status(200).json({
    ok: true,
    service: "kokumei-keshimasu-duel",
    environment: process.env.NODE_ENV || "development",
    modes: puzzleLoader.availableModes().map((mode) => mode.id),
    puzzleCount: puzzleLoader.count,
    activeMatches: matchManager.roomCount,
    waitingPlayers: matchManager.waitingCount,
    timestamp: new Date().toISOString(),
  });
});

// 軽量な状態API
app.get("/api/status", (_request, response) => {
  response.status(200).json({
    ok: true,
    puzzles: puzzleLoader.count,
    activeMatches: matchManager.roomCount,
    waitingPlayers: matchManager.waitingCount,
  });
});

// 遊べるモードと対戦ルールの一覧（ロビー画面が使います）
app.get("/api/modes", (_request, response) => {
  response.status(200).json({
    ok: true,
    modes: puzzleLoader.availableModes(),
    durationMs: MATCH_DURATION_MS,
    levelCount: LEVELS.length,
  });
});

// 選択中のワードが成立するかを画面で示すための単語一覧（公開してよい辞書）
app.get("/api/words/:mode", (request, response) => {
  const mode = String(request.params.mode || "");

  if (!puzzleLoader.hasMode(mode)) {
    response.status(404).json({ ok: false, error: "不明なモードです" });

    return;
  }

  response.set("Cache-Control", "public, max-age=3600");
  response.status(200).json({ ok: true, mode, words: puzzleLoader.wordList(mode) });
});

app.get("/", (_request, response) => {
  response.sendFile(path.join(PUBLIC_DIR, "index.html"));
});


/* ==================================================
   Socket.IO
================================================== */

function sendSocketError(ack, errorMessage) {
  if (typeof ack === "function") {
    ack({ ok: false, error: errorMessage });
  }
}

io.on("connection", (socket) => {
  console.log(`[socket] connected: ${socket.id}`);

  /*
   * イベント名ごとの処理を登録します。
   * 例外が出ても対戦全体が止まらないよう、ここでまとめて受け止めます。
   */
  function on(eventName, errorMessage, handler) {
    socket.on(eventName, (payload, ack) => {
      // payload を省略して ack だけ渡された場合にも対応する
      if (typeof payload === "function") {
        ack = payload;
        payload = {};
      }

      try {
        handler(payload || {}, ack);
      } catch (error) {
        console.error(`[${eventName}]`, error);
        sendSocketError(ack, errorMessage);
      }
    });
  }

  // 自動マッチングへ参加（payload: { name, mode }）
  on("matchmaking:join", "マッチングへの参加に失敗しました", (payload, ack) =>
    matchManager.joinMatchmaking(socket, payload, ack)
  );

  // マッチング待機を中止
  on("matchmaking:cancel", "マッチングの中止に失敗しました", (_payload, ack) =>
    matchManager.cancelMatchmaking(socket, ack)
  );

  // ページ更新・再接続後の復帰
  on("room:resume", "対戦への復帰に失敗しました", (payload, ack) =>
    matchManager.resumeRoom(socket, payload, ack)
  );

  /*
   * 選択した文字を送信
   *
   * payload: { path: [[row, col], ...], fText: "" }
   * row は盤面全体での行番号（画面に見えている最初の行は visibleStartRow）。
   */
  on("move:submit", "文字の判定中にエラーが発生しました", (payload, ack) =>
    matchManager.submitMove(socket, payload, ack)
  );

  // 今のレベルの盤面を最初の状態へ戻す（経過時間・得点は戻らない）
  on("match:reset", "盤面のリセットに失敗しました", (_payload, ack) =>
    matchManager.resetBoard(socket, ack)
  );

  // 同じ相手と再戦（2人とも押すと、新しい問題列で再開）
  on("match:rematch", "再戦の受付に失敗しました", (_payload, ack) =>
    matchManager.requestRematch(socket, ack)
  );

  // 対戦から退出
  on("match:leave", "対戦からの退出に失敗しました", (_payload, ack) =>
    matchManager.leaveRoom(socket, ack)
  );

  socket.on("disconnect", (reason) => {
    console.log(`[socket] disconnected: ${socket.id}`, reason);

    try {
      matchManager.disconnect(socket);
    } catch (error) {
      console.error("[socket disconnect]", error);
    }
  });

  socket.on("error", (error) => {
    console.error(`[socket] error: ${socket.id}`, error);
  });
});


/* ==================================================
   404・フォールバック・エラー
================================================== */

app.use("/api", (_request, response) => {
  response.status(404).json({
    ok: false,
    error: "指定されたAPIが見つかりません",
  });
});

// API以外の存在しないパスは index.html へ戻す
app.use((request, response, next) => {
  if (request.method !== "GET") {
    next();

    return;
  }

  response.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

app.use((error, _request, response, _next) => {
  console.error("[express error]", error);

  if (response.headersSent) {
    return;
  }

  response.status(500).json({
    ok: false,
    error: "サーバー内部でエラーが発生しました",
  });
});


/* ==================================================
   起動
================================================== */

function ensureFileExists(filePath, label) {
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

function startServer() {
  for (const fileName of ["index.html", "style.css", "client.js"]) {
    ensureFileExists(path.join(PUBLIC_DIR, fileName), fileName);
  }

  // 辞書と問題プールを読み込む（遊べるモードが1つも無ければ起動に失敗します）
  puzzleLoader.load();

  // Renderの標準ファイルシステムは再起動で消えますが、
  // 書き込めなくても対戦自体は動きます。
  try {
    fs.mkdirSync(MATCH_DIR, { recursive: true });
  } catch (error) {
    console.warn("[match output] directory creation failed:", error.message);
  }

  httpServer.listen(PORT, HOST, () => {
    console.log("================================");
    console.log(" KOKUMEI KESHIMASU DUEL");
    console.log("================================");
    console.log(`Environment : ${process.env.NODE_ENV || "development"}`);
    console.log(`Host / Port : ${HOST}:${PORT}`);
    console.log(`Public      : ${PUBLIC_DIR}`);
    console.log(`Words       : ${WORDS_DIR}`);
    console.log(`Pools       : ${POOLS_DIR}`);
    console.log(
      `Modes       : ${puzzleLoader
        .availableModes()
        .map((mode) => mode.id)
        .join(", ")}`
    );
    console.log(`Puzzles     : ${puzzleLoader.count}`);
    console.log(`Match output: ${MATCH_DIR}`);

    if (process.env.NODE_ENV !== "production") {
      console.log(`Local URL   : http://localhost:${PORT}`);
    }

    console.log("================================");
  });
}


/* ==================================================
   正常終了・予期しないエラー
================================================== */

let shuttingDown = false;

function shutdown(signal) {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;

  console.log(`[server] ${signal} received`);

  io.close(() => console.log("[socket.io] closed"));

  httpServer.close((error) => {
    if (error) {
      console.error("[server] close error", error);
      process.exitCode = 1;

      return;
    }

    console.log("[server] closed");
    process.exitCode = 0;
  });

  // 接続が残り続けた場合の強制終了
  setTimeout(() => {
    console.error("[server] forced shutdown");
    process.exit(1);
  }, 10_000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

process.on("uncaughtException", (error) => {
  console.error("[uncaughtException]", error);
  shutdown("uncaughtException");
});

process.on("unhandledRejection", (reason) => {
  console.error("[unhandledRejection]", reason);
});

try {
  startServer();
} catch (error) {
  console.error("[startup error]", error);
  process.exit(1);
}
