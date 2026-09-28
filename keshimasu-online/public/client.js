"use strict";


/* ==================================================
   Socket.IO・DOM
================================================== */

const socket = io();

const $ = (id) =>
  document.getElementById(id);


/* ==================================================
   ゲーム設定
================================================== */

/*
 * サーバー内部の盤面は8行×5列です。
 *
 * ブラウザに表示されるのは、
 * 元盤面の行3〜7の5行だけです。
 */
const VISIBLE_START_ROW = 3;
const VISIBLE_ROWS = 5;
const COLS = 5;

const EMPTY = "・";

const WILDS = new Set([
  "F",
  "Ｆ",
]);

/*
 * 対戦結果を表示してから
 * ロビーへ戻るまでの時間です。
 */
const RESULT_DISPLAY_MS = 5000;


/* ==================================================
   状態
================================================== */

let roomId = null;
let playerToken = null;

let board = null;
let selected = [];

let startAt = null;

let timerHandle = null;
let countdownHandle = null;
let returnToLobbyHandle = null;

let status = "idle";


/* ==================================================
   画面表示
================================================== */

function show(screen) {
  $("lobby").classList.toggle(
    "hidden",
    screen !== "lobby"
  );

  $("game").classList.toggle(
    "hidden",
    screen !== "game"
  );
}


function setMessage(text) {
  $("message").textContent =
    text || "";
}


function setGameMessage(text) {
  $("gameMessage").textContent =
    text || "";
}


/* ==================================================
   Socket.IO送信
================================================== */

function emitAck(
  eventName,
  payload = {}
) {
  return new Promise((resolve) => {
    socket.emit(
      eventName,
      payload,
      (response) => {
        resolve(
          response || {
            ok: false,
            error:
              "サーバーから応答がありません",
          }
        );
      }
    );
  });
}


/* ==================================================
   セッション保存
================================================== */

function saveSession() {
  sessionStorage.setItem(
    "keshimasuDuel",
    JSON.stringify({
      roomId,
      playerToken,
    })
  );
}


function loadSession() {
  try {
    return JSON.parse(
      sessionStorage.getItem(
        "keshimasuDuel"
      ) || "null"
    );
  } catch {
    return null;
  }
}


function clearSession() {
  sessionStorage.removeItem(
    "keshimasuDuel"
  );

  roomId = null;
  playerToken = null;
}


/* ==================================================
   タイマー停止
================================================== */

function stopGameTimer() {
  if (timerHandle !== null) {
    clearInterval(timerHandle);
    timerHandle = null;
  }
}


function stopCountdown() {
  if (countdownHandle !== null) {
    clearInterval(countdownHandle);
    countdownHandle = null;
  }
}


function stopReturnToLobbyTimer() {
  if (returnToLobbyHandle !== null) {
    clearTimeout(returnToLobbyHandle);
    returnToLobbyHandle = null;
  }
}


/* ==================================================
   ロビー状態の初期化
================================================== */

function resetLobbyUi() {
  $("timer").textContent = "0.000";

  $("opponentRemaining").textContent =
    "40";

  $("opponentProgress").style.width =
    "0%";

  $("opponentProgress")
    .parentElement
    ?.setAttribute(
      "aria-valuenow",
      "0"
    );

  $("countdown").textContent = "";

  $("selectedText").textContent =
    "未選択";

  $("fText").value = "";

  $("fText").maxLength = 5;

  $("fInputHelp").textContent = "";

  $("fInputArea")
    .classList
    .add("hidden");

  $("result")
    .classList
    .add("hidden");

  $("matchingIndicator")
    .classList
    .add("hidden");

  $("cancelMatch")
    .classList
    .add("hidden");

  $("matchButton").disabled = false;
  $("submitMove").disabled = true;

  setGameMessage("");
}


/* ==================================================
   最初の画面へ戻る
================================================== */

async function returnToLobby({
  notifyServer = true,
  message = "",
} = {}) {
  stopGameTimer();
  stopCountdown();
  stopReturnToLobbyTimer();

  if (
    notifyServer &&
    socket.connected &&
    roomId
  ) {
    try {
      await emitAck(
        "match:leave",
        {}
      );
    } catch (error) {
      console.warn(
        "対戦退出の通知に失敗しました",
        error
      );
    }
  }

  clearSession();

  board = null;
  selected = [];

  startAt = null;
  status = "idle";

  resetLobbyUi();

  setMessage(message);

  show("lobby");
}


/* ==================================================
   表示盤面の文字取得
================================================== */

function getVisibleCell(
  absoluteRow,
  col
) {
  if (!Array.isArray(board)) {
    return null;
  }

  const visibleRow =
    absoluteRow -
    VISIBLE_START_ROW;

  if (
    visibleRow < 0 ||
    visibleRow >= VISIBLE_ROWS ||
    col < 0 ||
    col >= COLS
  ) {
    return null;
  }

  return (
    board[visibleRow]?.[col] ??
    null
  );
}


function getSelectedCharacters() {
  return selected.map(
    ([absoluteRow, col]) =>
      getVisibleCell(
        absoluteRow,
        col
      ) || ""
  );
}


function getSelectedText() {
  return getSelectedCharacters()
    .join("");
}


function getSelectedFCount() {
  return getSelectedCharacters()
    .filter(
      (character) =>
        WILDS.has(character)
    )
    .length;
}


/* ==================================================
   回答欄更新
================================================== */

function updateAnswerControls() {
  const selectedText =
    getSelectedText();

  const fCount =
    getSelectedFCount();

  $("selectedText").textContent =
    selectedText || "未選択";

  if (fCount > 0) {
    $("fInputArea")
      .classList
      .remove("hidden");

    $("fInputHelp").textContent =
      `（${fCount}文字）`;

    $("fText").maxLength =
      fCount;

    $("fText").placeholder =
      `Fに入る文字を${fCount}文字入力`;
  } else {
    $("fInputArea")
      .classList
      .add("hidden");

    $("fInputHelp").textContent =
      "";

    $("fText").value = "";
    $("fText").maxLength = 5;

    $("fText").placeholder =
      "Fに入る文字を入力";
  }

  $("submitMove").disabled =
    status !== "playing" ||
    selected.length < 2 ||
    selected.length > 5;
}


/* ==================================================
   盤面表示
================================================== */

function renderBoard() {
  const boardElement =
    $("board");

  boardElement.innerHTML = "";

  if (!Array.isArray(board)) {
    updateAnswerControls();
    return;
  }

  board.forEach(
    (row, visibleRow) => {
      row.forEach(
        (value, col) => {
          const absoluteRow =
            visibleRow +
            VISIBLE_START_ROW;

          const button =
            document.createElement(
              "button"
            );

          button.type = "button";
          button.setAttribute(
            "role",
            "gridcell"
          );

          const selectedNow =
            selected.some(
              ([
                selectedRow,
                selectedCol,
              ]) =>
                selectedRow ===
                  absoluteRow &&
                selectedCol === col
            );

          button.className =
            "cell" +
            (
              value === EMPTY
                ? " empty"
                : ""
            ) +
            (
              selectedNow
                ? " selected"
                : ""
            );

          button.textContent =
            value;

          button.dataset.row =
            String(absoluteRow);

          button.dataset.col =
            String(col);

          button.disabled =
            value === EMPTY ||
            status !== "playing";

          button.setAttribute(
            "aria-label",
            `行${visibleRow + 1} 列${col + 1} ${value}`
          );

          button.setAttribute(
            "aria-selected",
            selectedNow
              ? "true"
              : "false"
          );

          button.onclick = () => {
            selectCell(
              absoluteRow,
              col
            );
          };

          boardElement.appendChild(
            button
          );
        }
      );
    }
  );

  updateAnswerControls();
}


/* ==================================================
   マス選択
================================================== */

function selectCell(row, col) {
  if (status !== "playing") {
    return;
  }

  const cell =
    getVisibleCell(row, col);

  if (
    !cell ||
    cell === EMPTY
  ) {
    return;
  }

  /*
   * すでに選択済みのマスを押した場合、
   * その位置まで選択を戻します。
   */
  const existingIndex =
    selected.findIndex(
      ([
        selectedRow,
        selectedCol,
      ]) =>
        selectedRow === row &&
        selectedCol === col
    );

  if (existingIndex >= 0) {
    selected = selected.slice(
      0,
      existingIndex + 1
    );

    renderBoard();
    return;
  }

  /*
   * 1文字目
   */
  if (selected.length === 0) {
    selected = [[row, col]];

    setGameMessage("");
    renderBoard();
    return;
  }

  const [
    firstRow,
    firstCol,
  ] = selected[0];

  const [
    lastRow,
    lastCol,
  ] = selected[
    selected.length - 1
  ];

  /*
   * 2文字目で方向を決定します。
   */
  if (selected.length === 1) {
    const rowDifference =
      row - firstRow;

    const colDifference =
      col - firstCol;

    const adjacent =
      (
        Math.abs(rowDifference) === 1 &&
        colDifference === 0
      ) ||
      (
        Math.abs(colDifference) === 1 &&
        rowDifference === 0
      );

    if (!adjacent) {
      selected = [[row, col]];

      setGameMessage(
        "縦または横に連続して選択してください"
      );

      renderBoard();
      return;
    }

    selected.push([row, col]);

    setGameMessage("");
    renderBoard();
    return;
  }

  /*
   * 3文字目以降は、2文字目までと
   * 同じ方向にだけ伸ばせます。
   */
  const rowDirection =
    selected[1][0] -
    firstRow;

  const colDirection =
    selected[1][1] -
    firstCol;

  const nextRowDifference =
    row - lastRow;

  const nextColDifference =
    col - lastCol;

  const continuesSameDirection =
    nextRowDifference ===
      rowDirection &&
    nextColDifference ===
      colDirection;

  if (
    !continuesSameDirection ||
    selected.length >= 5
  ) {
    selected = [[row, col]];

    setGameMessage(
      "新しい選択を開始しました"
    );
  } else {
    selected.push([row, col]);

    setGameMessage("");
  }

  renderBoard();
}


/* ==================================================
   カウントダウン
================================================== */

function runCountdown() {
  stopCountdown();

  function updateCountdown() {
    if (!startAt) {
      $("countdown").textContent = "";
      return;
    }

    const remainingMilliseconds =
      startAt - Date.now();

    const remainingSeconds =
      Math.ceil(
        remainingMilliseconds /
        1000
      );

    $("countdown").textContent =
      remainingSeconds > 0
        ? String(remainingSeconds)
        : "開始!";

    if (remainingMilliseconds <= 0) {
      stopCountdown();
    }
  }

  updateCountdown();

  countdownHandle =
    setInterval(
      updateCountdown,
      50
    );
}


/* ==================================================
   ストップウォッチ
================================================== */

function startTimer() {
  stopGameTimer();

  function updateTimer() {
    if (!startAt) {
      $("timer").textContent =
        "0.000";

      return;
    }

    const elapsed =
      Math.max(
        0,
        Date.now() - startAt
      );

    $("timer").textContent =
      (
        elapsed / 1000
      ).toFixed(3);
  }

  updateTimer();

  timerHandle =
    setInterval(
      updateTimer,
      33
    );
}


/* ==================================================
   対戦する
================================================== */

$("matchButton").onclick =
  async () => {
    stopReturnToLobbyTimer();

    $("matchButton").disabled =
      true;

    setMessage(
      "対戦相手を探しています…"
    );

    $("matchingIndicator")
      .classList
      .remove("hidden");

    $("cancelMatch")
      .classList
      .remove("hidden");

    const result =
      await emitAck(
        "matchmaking:join",
        {
          name:
            $("name").value
              .trim() ||
            "プレイヤー",
        }
      );

    if (!result.ok) {
      $("matchButton").disabled =
        false;

      $("matchingIndicator")
        .classList
        .add("hidden");

      $("cancelMatch")
        .classList
        .add("hidden");

      setMessage(result.error);
      return;
    }

    playerToken =
      result.playerToken ||
      playerToken;

    status = "matching";
  };


/* ==================================================
   マッチング中止
================================================== */

$("cancelMatch").onclick =
  async () => {
    await emitAck(
      "matchmaking:cancel",
      {}
    );

    status = "idle";

    $("matchButton").disabled =
      false;

    $("matchingIndicator")
      .classList
      .add("hidden");

    $("cancelMatch")
      .classList
      .add("hidden");

    setMessage(
      "マッチングを中止しました"
    );
  };


/* ==================================================
   マッチング待機イベント
================================================== */

socket.on(
  "matchmaking:waiting",
  () => {
    status = "matching";

    setMessage(
      "対戦相手を探しています…"
    );

    $("matchingIndicator")
      .classList
      .remove("hidden");
  }
);


/* ==================================================
   マッチング中止イベント
================================================== */

socket.on(
  "matchmaking:cancelled",
  () => {
    status = "idle";

    $("matchButton").disabled =
      false;

    $("matchingIndicator")
      .classList
      .add("hidden");

    $("cancelMatch")
      .classList
      .add("hidden");
  }
);


/* ==================================================
   マッチング成立
================================================== */

socket.on(
  "matchmaking:matched",
  (data) => {
    stopReturnToLobbyTimer();

    roomId = data.roomId;

    playerToken =
      data.playerToken;

    saveSession();

    $("yourName").textContent =
      $("name").value.trim() ||
      "あなた";

    $("opponentName").textContent =
      data.opponentName ||
      "対戦相手";

    $("opponentRemaining")
      .textContent = "40";

    $("opponentProgress")
      .style.width = "0%";

    $("matchingIndicator")
      .classList
      .add("hidden");

    $("cancelMatch")
      .classList
      .add("hidden");

    setMessage(
      "マッチングしました"
    );
  }
);


/* ==================================================
   カウントダウン開始
================================================== */

socket.on(
  "match:countdown",
  (data) => {
    stopGameTimer();
    stopCountdown();
    stopReturnToLobbyTimer();

    roomId = data.roomId;

    board =
      data.puzzle.board.map(
        (row) => row.slice()
      );

    selected = [];

    startAt = data.startAt;
    status = "countdown";

    $("timer").textContent =
      "0.000";

    $("opponentRemaining")
      .textContent = "40";

    $("opponentProgress")
      .style.width = "0%";

    $("result")
      .classList
      .add("hidden");

    $("fText").value = "";

    setGameMessage("");

    show("game");
    renderBoard();
    runCountdown();
  }
);


/* ==================================================
   対戦開始
================================================== */

socket.on(
  "match:started",
  (data) => {
    startAt = data.startAt;
    status = "playing";

    renderBoard();
    startTimer();

    $("countdown").textContent =
      "開始!";

    setTimeout(
      () => {
        if (status === "playing") {
          $("countdown")
            .textContent = "";
        }
      },
      600
    );
  }
);


/* ==================================================
   相手の進捗
================================================== */

socket.on(
  "opponent:progress",
  (progress) => {
    const remaining =
      Number(progress.remaining);

    const safeRemaining =
      Number.isFinite(remaining)
        ? Math.max(
            0,
            Math.min(40, remaining)
          )
        : 40;

    const completed =
      40 - safeRemaining;

    const progressPercent =
      (
        completed /
        40
      ) * 100;

    $("opponentRemaining")
      .textContent =
        String(safeRemaining);

    $("opponentProgress")
      .style.width =
        `${progressPercent}%`;

    $("opponentProgress")
      .parentElement
      ?.setAttribute(
        "aria-valuenow",
        String(completed)
      );
  }
);


/* ==================================================
   対戦終了
================================================== */

socket.on(
  "match:finished",
  (result) => {
    status = "finished";

    stopGameTimer();
    stopCountdown();
    stopReturnToLobbyTimer();

    renderBoard();

    const won =
      result.winnerToken ===
      playerToken;

    $("resultTitle").textContent =
      won
        ? "勝利!"
        : "相手が先にクリアしました";

    $("resultTitle").className =
      won
        ? "winner"
        : "loser";

    $("result")
      .classList
      .remove("hidden");

    /*
     * 結果画面を5秒表示した後、
     * 自動的に最初の画面へ戻します。
     */
    returnToLobbyHandle =
      setTimeout(
        () => {
          returnToLobby({
            notifyServer: true,
            message:
              won
                ? "勝利しました。もう一度「対戦する」を押してください。"
                : "対戦が終了しました。もう一度「対戦する」を押してください。",
          });
        },
        RESULT_DISPLAY_MS
      );
  }
);


/* ==================================================
   選択解除
================================================== */

$("clearSelection").onclick =
  () => {
    selected = [];

    $("fText").value = "";

    setGameMessage("");

    renderBoard();
  };


/* ==================================================
   決定
================================================== */

$("submitMove").onclick =
  async () => {
    if (status !== "playing") {
      setGameMessage(
        "まだ開始していません"
      );

      return;
    }

    if (
      selected.length < 2 ||
      selected.length > 5
    ) {
      setGameMessage(
        "2〜5文字を一直線に選択してください"
      );

      return;
    }

    const fCount =
      getSelectedFCount();

    let fText = "";

    /*
     * Fが含まれる場合だけ、
     * Fへ入れる文字を取得します。
     */
    if (fCount > 0) {
      fText =
        $("fText").value.trim();

      if (
        [...fText].length !==
        fCount
      ) {
        $("fInputArea")
          .classList
          .remove("hidden");

        $("fText").focus();

        setGameMessage(
          `Fに入る文字を${fCount}文字入力してください`
        );

        return;
      }
    }

    $("submitMove").disabled =
      true;

    const result =
      await emitAck(
        "move:submit",
        {
          path: selected,
          fText,
        }
      );

    if (!result.ok) {
      setGameMessage(
        result.error ||
        "正しい国名ではありません"
      );

      if (fCount > 0) {
        $("fText").focus();
        $("fText").select();
      }

      updateAnswerControls();
      return;
    }

    board =
      result.board.map(
        (row) => row.slice()
      );

    selected = [];

    $("fText").value = "";

    setGameMessage(
      `「${result.word}」正解　残り${result.remaining}マス`
    );

    renderBoard();
  };


/* ==================================================
   F入力欄のEnterキー
================================================== */

$("fText").addEventListener(
  "keydown",
  (event) => {
    if (event.key === "Enter") {
      event.preventDefault();

      $("submitMove").click();
    }
  }
);


/* ==================================================
   同じ相手と再戦
================================================== */

$("rematch").onclick =
  async () => {
    /*
     * 再戦を選択した場合は、
     * 自動ロビー復帰を止めます。
     */
    stopReturnToLobbyTimer();

    const result =
      await emitAck(
        "match:rematch",
        {}
      );

    if (!result.ok) {
      setGameMessage(
        result.error ||
        "再戦の受付に失敗しました"
      );

      /*
       * 再戦受付に失敗した場合は
       * ロビーへ戻します。
       */
      returnToLobbyHandle =
        setTimeout(
          () => {
            returnToLobby({
              notifyServer: true,
              message:
                "再戦できなかったため、最初の画面へ戻りました。",
            });
          },
          2000
        );

      return;
    }

    $("result")
      .classList
      .add("hidden");

    $("countdown").textContent =
      "相手の再戦操作を待っています…";

    setGameMessage(
      "相手も再戦を選択すると、新しい問題を開始します。"
    );
  };


/* ==================================================
   新しい相手と対戦
================================================== */

$("newMatch").onclick =
  async () => {
    await returnToLobby({
      notifyServer: true,
      message:
        "「対戦する」を押すと、新しい相手を探します。",
    });
  };


/* ==================================================
   サーバーへの接続
================================================== */

socket.on(
  "connect",
  async () => {
    const saved =
      loadSession();

    if (
      !saved?.roomId ||
      !saved?.playerToken
    ) {
      return;
    }

    const result =
      await emitAck(
        "room:resume",
        saved
      );

    if (!result.ok) {
      clearSession();

      resetLobbyUi();

      setMessage(
        "以前の対戦は終了しています。"
      );

      show("lobby");
      return;
    }

    roomId = saved.roomId;
    playerToken =
      saved.playerToken;

    status =
      result.state?.status ||
      "idle";

    startAt =
      result.startAt ||
      result.state?.startAt ||
      null;

    const players =
      result.state?.players || [];

    const currentPlayer =
      players.find(
        (player) =>
          player.isYou
      );

    const opponent =
      players.find(
        (player) =>
          !player.isYou
      );

    if (currentPlayer?.name) {
      $("yourName").textContent =
        currentPlayer.name;
    }

    if (opponent?.name) {
      $("opponentName").textContent =
        opponent.name;
    }

    if (
      Number.isFinite(
        Number(opponent?.remaining)
      )
    ) {
      const opponentRemaining =
        Number(
          opponent.remaining
        );

      $("opponentRemaining")
        .textContent =
          String(opponentRemaining);

      $("opponentProgress")
        .style.width =
          `${
            (
              (40 -
                opponentRemaining) /
              40
            ) * 100
          }%`;
    }

    if (
      Array.isArray(result.board)
    ) {
      board =
        result.board.map(
          (row) => row.slice()
        );

      selected = [];

      show("game");
      renderBoard();

      if (status === "playing") {
        startTimer();

        $("countdown")
          .textContent = "";
      } else if (
        status === "countdown"
      ) {
        runCountdown();
      } else if (
        status === "finished"
      ) {
        /*
         * 終了済み対戦へ復帰した場合は、
         * 長く残さずロビーへ戻します。
         */
        returnToLobbyHandle =
          setTimeout(
            () => {
              returnToLobby({
                notifyServer: true,
                message:
                  "対戦は終了しています。もう一度「対戦する」を押してください。",
              });
            },
            1000
          );
      }
    }
  }
);


/* ==================================================
   Socket.IO切断
================================================== */

socket.on(
  "disconnect",
  () => {
    if (
      status === "matching"
    ) {
      setMessage(
        "サーバーとの接続が切れました。再接続しています…"
      );
    } else if (
      status === "playing" ||
      status === "countdown"
    ) {
      setGameMessage(
        "サーバーとの接続が切れました。再接続しています…"
      );
    }
  }
);


/* ==================================================
   初期表示
================================================== */

resetLobbyUi();
show("lobby");