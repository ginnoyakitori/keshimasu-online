"use strict";


/* ==================================================
   モジュール
================================================== */

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const express = require("express");
const { Server } = require("socket.io");

const {
  PuzzleLoader,
} = require("./src/multiplayer/puzzle-loader");

const {
  MatchManager,
} = require("./src/multiplayer/match-manager");


/* ==================================================
   パス・環境変数
================================================== */

const ROOT = __dirname;

const PORT = Number(
  process.env.PORT || 3000
);

const HOST =
  process.env.HOST || "0.0.0.0";


function resolveProjectPath(
  environmentValue,
  defaultRelativePath
) {
  const value =
    environmentValue ||
    defaultRelativePath;

  if (path.isAbsolute(value)) {
    return value;
  }

  return path.resolve(
    ROOT,
    value
  );
}


const PUBLIC_DIR =
  resolveProjectPath(
    process.env.PUBLIC_DIR,
    "./public"
  );

const PUZZLE_DIR =
  resolveProjectPath(
    process.env.PUZZLE_DIR,
    "./data/ga_reverse"
  );

const COUNTRY_FILE =
  resolveProjectPath(
    process.env.COUNTRY_FILE,
    "./data/words/countries.json"
  );

const MATCH_DIR =
  resolveProjectPath(
    process.env.MATCH_DIR,
    "./data/matches"
  );


/* ==================================================
   Express・HTTP・Socket.IO
================================================== */

const app = express();

const httpServer =
  http.createServer(app);

const io = new Server(
  httpServer,
  {
    maxHttpBufferSize: 100_000,

    pingInterval: 25_000,
    pingTimeout: 20_000,

    transports: [
      "polling",
      "websocket",
    ],

    cors: {
      origin: false,
    },
  }
);


/* ==================================================
   Express基本設定
================================================== */

app.disable("x-powered-by");

app.use(
  express.json({
    limit: "64kb",
  })
);

app.use(
  express.urlencoded({
    extended: false,
    limit: "64kb",
  })
);


/*
 * public/
 * ├─ index.html
 * ├─ style.css
 * └─ client.js
 *
 * を配信します。
 */
app.use(
  express.static(
    PUBLIC_DIR,
    {
      extensions: [
        "html",
      ],

      etag: true,

      maxAge:
        process.env.NODE_ENV ===
        "production"
          ? "5m"
          : 0,

      setHeaders(response, filePath) {
        /*
         * index.htmlだけキャッシュを弱くします。
         * Renderへ再デプロイしたときに、
         * 古いHTMLが残りにくくなります。
         */
        if (
          path.basename(filePath) ===
          "index.html"
        ) {
          response.setHeader(
            "Cache-Control",
            "no-cache"
          );
        }
      },
    }
  )
);


/* ==================================================
   問題ローダー・対戦管理
================================================== */

const puzzleLoader =
  new PuzzleLoader({
    puzzleDir: PUZZLE_DIR,
    countryFile: COUNTRY_FILE,
  });

const matchManager =
  new MatchManager({
    io,
    puzzleLoader,
    matchDir: MATCH_DIR,
  });


/* ==================================================
   HTTP API
================================================== */

/*
 * Renderのヘルスチェック用です。
 *
 * GET /api/health
 */
app.get(
  "/api/health",
  (_request, response) => {
    response.status(200).json({
      ok: true,

      service:
        "kokumei-keshimasu-duel",

      environment:
        process.env.NODE_ENV ||
        "development",

      puzzleCount:
        puzzleLoader.count,

      countryCount:
        puzzleLoader.countryCount,

      activeMatches:
        matchManager.roomCount,

      waitingPlayers:
        matchManager.waitingCount,

      timestamp:
        new Date().toISOString(),
    });
  }
);


/*
 * ブラウザや管理画面で確認するための
 * 軽量な状態APIです。
 *
 * GET /api/status
 */
app.get(
  "/api/status",
  (_request, response) => {
    response.status(200).json({
      ok: true,

      puzzles:
        puzzleLoader.count,

      countries:
        puzzleLoader.countryCount,

      activeMatches:
        matchManager.roomCount,

      waitingPlayers:
        matchManager.waitingCount,
    });
  }
);


/*
 * トップページです。
 */
app.get(
  "/",
  (_request, response) => {
    response.sendFile(
      path.join(
        PUBLIC_DIR,
        "index.html"
      )
    );
  }
);


/* ==================================================
   Socket.IO共通エラー応答
================================================== */

function sendSocketError(
  ack,
  errorMessage
) {
  if (
    typeof ack !== "function"
  ) {
    return;
  }

  ack({
    ok: false,
    error: errorMessage,
  });
}


/* ==================================================
   Socket.IO接続
================================================== */

io.on(
  "connection",
  (socket) => {
    console.log(
      `[socket] connected: ${socket.id}`
    );


    /* ----------------------------------------------
       自動マッチングへ参加
    ---------------------------------------------- */

    socket.on(
      "matchmaking:join",
      (payload = {}, ack) => {
        try {
          matchManager.joinMatchmaking(
            socket,
            payload,
            ack
          );
        } catch (error) {
          console.error(
            "[matchmaking:join]",
            error
          );

          sendSocketError(
            ack,
            "マッチングへの参加に失敗しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       マッチング待機を中止
    ---------------------------------------------- */

    socket.on(
      "matchmaking:cancel",
      (_payload = {}, ack) => {
        try {
          matchManager.cancelMatchmaking(
            socket,
            ack
          );
        } catch (error) {
          console.error(
            "[matchmaking:cancel]",
            error
          );

          sendSocketError(
            ack,
            "マッチングの中止に失敗しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       ページ更新・再接続後の復帰
    ---------------------------------------------- */

    socket.on(
      "room:resume",
      (payload = {}, ack) => {
        try {
          matchManager.resumeRoom(
            socket,
            payload,
            ack
          );
        } catch (error) {
          console.error(
            "[room:resume]",
            error
          );

          sendSocketError(
            ack,
            "対戦への復帰に失敗しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       選択した文字を送信
    ---------------------------------------------- */

    socket.on(
      "move:submit",
      (payload = {}, ack) => {
        try {
          /*
           * payloadの形式:
           *
           * {
           *   path: [
           *     [5, 0],
           *     [6, 0],
           *     [7, 0]
           *   ],
           *   fText: ""
           * }
           *
           * pathの行番号は8行盤面上の絶対座標です。
           *
           * ブラウザには下5行しか表示しませんが、
           * 表示1行目は絶対座標の行3です。
           *
           * Fがない場合:
           *   fText = ""
           *
           * Fが1個の場合:
           *   fText = "ラ"
           *
           * Fが2個の場合:
           *   fText = "イン"
           */
          matchManager.submitMove(
            socket,
            payload,
            ack
          );
        } catch (error) {
          console.error(
            "[move:submit]",
            error
          );

          sendSocketError(
            ack,
            "文字の判定中にエラーが発生しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       同じ相手と再戦
    ---------------------------------------------- */

    socket.on(
      "match:rematch",
      (_payload = {}, ack) => {
        try {
          /*
           * 両方のプレイヤーが再戦を押すと、
           * 新しい問題が選択されます。
           *
           * その後、再び3秒カウントダウンして
           * 同時に開始します。
           */
          matchManager.requestRematch(
            socket,
            ack
          );
        } catch (error) {
          console.error(
            "[match:rematch]",
            error
          );

          sendSocketError(
            ack,
            "再戦の受付に失敗しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       対戦から退出
    ---------------------------------------------- */

    socket.on(
      "match:leave",
      (_payload = {}, ack) => {
        try {
          matchManager.leaveRoom(
            socket,
            ack
          );
        } catch (error) {
          console.error(
            "[match:leave]",
            error
          );

          sendSocketError(
            ack,
            "対戦からの退出に失敗しました"
          );
        }
      }
    );


    /* ----------------------------------------------
       Socket.IO切断
    ---------------------------------------------- */

    socket.on(
      "disconnect",
      (reason) => {
        console.log(
          `[socket] disconnected: ${socket.id}`,
          reason
        );

        try {
          matchManager.disconnect(
            socket
          );
        } catch (error) {
          console.error(
            "[socket disconnect]",
            error
          );
        }
      }
    );


    /* ----------------------------------------------
       Socket.IOエラー
    ---------------------------------------------- */

    socket.on(
      "error",
      (error) => {
        console.error(
          `[socket] error: ${socket.id}`,
          error
        );
      }
    );
  }
);


/* ==================================================
   API 404
================================================== */

app.use(
  "/api",
  (
    _request,
    response
  ) => {
    response.status(404).json({
      ok: false,
      error:
        "指定されたAPIが見つかりません",
    });
  }
);


/* ==================================================
   ブラウザページのフォールバック
================================================== */

/*
 * API以外で存在しないパスへアクセスした場合は、
 * index.htmlへ戻します。
 *
 * Express 5でも問題が起きにくいように、
 * "*" のルートではなくミドルウェアを使用します。
 */
app.use(
  (
    request,
    response,
    next
  ) => {
    if (
      request.method !== "GET"
    ) {
      next();
      return;
    }

    response.sendFile(
      path.join(
        PUBLIC_DIR,
        "index.html"
      )
    );
  }
);


/* ==================================================
   Expressエラーハンドラー
================================================== */

app.use(
  (
    error,
    _request,
    response,
    _next
  ) => {
    console.error(
      "[express error]",
      error
    );

    if (
      response.headersSent
    ) {
      return;
    }

    response.status(500).json({
      ok: false,
      error:
        "サーバー内部でエラーが発生しました",
    });
  }
);


/* ==================================================
   起動前確認
================================================== */

function ensureDirectoryExists(
  directoryPath,
  label
) {
  if (
    !fs.existsSync(directoryPath)
  ) {
    throw new Error(
      `${label} not found: ${directoryPath}`
    );
  }

  const stat =
    fs.statSync(directoryPath);

  if (!stat.isDirectory()) {
    throw new Error(
      `${label} is not a directory: ${directoryPath}`
    );
  }
}


function ensureFileExists(
  filePath,
  label
) {
  if (
    !fs.existsSync(filePath)
  ) {
    throw new Error(
      `${label} not found: ${filePath}`
    );
  }

  const stat =
    fs.statSync(filePath);

  if (!stat.isFile()) {
    throw new Error(
      `${label} is not a file: ${filePath}`
    );
  }
}


function ensureRequiredFiles() {
  ensureDirectoryExists(
    PUBLIC_DIR,
    "Public directory"
  );

  ensureDirectoryExists(
    PUZZLE_DIR,
    "Puzzle directory"
  );

  ensureFileExists(
    path.join(
      PUBLIC_DIR,
      "index.html"
    ),
    "index.html"
  );

  ensureFileExists(
    path.join(
      PUBLIC_DIR,
      "style.css"
    ),
    "style.css"
  );

  ensureFileExists(
    path.join(
      PUBLIC_DIR,
      "client.js"
    ),
    "client.js"
  );

  ensureFileExists(
    COUNTRY_FILE,
    "Country file"
  );
}


/* ==================================================
   初期化
================================================== */

function initializeApplication() {
  ensureRequiredFiles();

  /*
   * data/ga_reverse 内の有効な問題と、
   * countries.json の国名を読み込みます。
   */
  puzzleLoader.load();

  /*
   * Renderの標準ファイルシステムでは、
   * 再起動や再デプロイ後に書き込み内容が
   * 消えることがあります。
   *
   * ただし、このディレクトリへ書き込めなくても
   * 問題配信自体は動作できます。
   */
  try {
    fs.mkdirSync(
      MATCH_DIR,
      {
        recursive: true,
      }
    );
  } catch (error) {
    console.warn(
      "[match output] directory creation failed:",
      error.message
    );
  }
}


/* ==================================================
   サーバー起動
================================================== */

function startServer() {
  initializeApplication();

  httpServer.listen(
    PORT,
    HOST,
    () => {
      console.log(
        "================================"
      );

      console.log(
        " KOKUMEI KESHIMASU DUEL"
      );

      console.log(
        "================================"
      );

      console.log(
        `Environment   : ${
          process.env.NODE_ENV ||
          "development"
        }`
      );

      console.log(
        `Host          : ${HOST}`
      );

      console.log(
        `Port          : ${PORT}`
      );

      console.log(
        `Public        : ${PUBLIC_DIR}`
      );

      console.log(
        `Puzzle dir    : ${PUZZLE_DIR}`
      );

      console.log(
        `Puzzle count  : ${puzzleLoader.count}`
      );

      console.log(
        `Country file  : ${COUNTRY_FILE}`
      );

      console.log(
        `Country count : ${puzzleLoader.countryCount}`
      );

      console.log(
        `Match output  : ${MATCH_DIR}`
      );

      if (
        process.env.NODE_ENV !==
        "production"
      ) {
        console.log(
          `Local URL     : http://localhost:${PORT}`
        );

        console.log(
          `Health check  : http://localhost:${PORT}/api/health`
        );
      }

      console.log(
        "================================"
      );
    }
  );
}


/* ==================================================
   正常終了
================================================== */

let shuttingDown = false;


function shutdown(signal) {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;

  console.log(
    `[server] ${signal} received`
  );

  io.close(() => {
    console.log(
      "[socket.io] closed"
    );
  });

  httpServer.close(
    (error) => {
      if (error) {
        console.error(
          "[server] close error",
          error
        );

        process.exitCode = 1;
        return;
      }

      console.log(
        "[server] closed"
      );

      process.exitCode = 0;
    }
  );

  /*
   * 一部の接続が残り続けた場合の
   * 強制終了用タイマーです。
   */
  const forceExitTimer =
    setTimeout(
      () => {
        console.error(
          "[server] forced shutdown"
        );

        process.exit(1);
      },
      10_000
    );

  forceExitTimer.unref();
}


process.on(
  "SIGTERM",
  () => {
    shutdown("SIGTERM");
  }
);


process.on(
  "SIGINT",
  () => {
    shutdown("SIGINT");
  }
);


/* ==================================================
   予期しないエラー
================================================== */

process.on(
  "uncaughtException",
  (error) => {
    console.error(
      "[uncaughtException]",
      error
    );

    shutdown(
      "uncaughtException"
    );
  }
);


process.on(
  "unhandledRejection",
  (reason) => {
    console.error(
      "[unhandledRejection]",
      reason
    );
  }
);


/* ==================================================
   実行
================================================== */

try {
  startServer();
} catch (error) {
  console.error(
    "[startup error]",
    error
  );

  process.exit(1);
}