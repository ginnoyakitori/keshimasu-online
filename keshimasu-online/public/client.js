"use strict";


const socket = io();

const $ = (id) =>
  document.getElementById(id);


const VISIBLE_START_ROW = 3;
const VISIBLE_ROWS = 5;
const COLS = 5;

const EMPTY = "・";
const WILDS = new Set(["F", "Ｆ"]);


let roomId = null;
let playerToken = null;

let board = null;
let selected = [];

let startAt = null;
let timerHandle = null;

let status = "idle";


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


function emitAck(
  event,
  payload = {}
) {
  return new Promise((resolve) => {
    socket.emit(
      event,
      payload,
      resolve
    );
  });
}


function saveSession() {
  sessionStorage.setItem(
    "keshimasuDuel",
    JSON.stringify({
      roomId,
      playerToken,
    })
  );
}


function clearSession() {
  sessionStorage.removeItem(
    "keshimasuDuel"
  );

  roomId = null;
  playerToken = null;
}


function getVisibleCell(
  absoluteRow,
  col
) {
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
    board?.[visibleRow]?.[col] ??
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
  }

  $("submitMove").disabled =
    status !== "playing" ||
    selected.length < 2 ||
    selected.length > 5;
}


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

          button.disabled =
            value === EMPTY ||
            status !== "playing";

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


function selectCell(row, col) {
  if (status !== "playing") {
    return;
  }

  const cell =
    getVisibleCell(row, col);

  if (!cell || cell === EMPTY) {
    return;
  }

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

  if (selected.length === 0) {
    selected = [[row, col]];

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

      renderBoard();
      return;
    }

    selected.push([row, col]);

    renderBoard();
    return;
  }

  const rowDirection =
    selected[1][0] - firstRow;

  const colDirection =
    selected[1][1] - firstCol;

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
  } else {
    selected.push([row, col]);
  }

  renderBoard();
}


function runCountdown() {
  const handle = setInterval(
    () => {
      const remaining =
        Math.ceil(
          (
            startAt -
            Date.now()
          ) / 1000
        );

      $("countdown").textContent =
        remaining > 0
          ? String(remaining)
          : "開始!";

      if (remaining <= 0) {
        clearInterval(handle);
      }
    },
    50
  );
}


function startTimer() {
  clearInterval(timerHandle);

  timerHandle = setInterval(
    () => {
      const elapsed =
        Math.max(
          0,
          Date.now() - startAt
        );

      $("timer").textContent =
        (
          elapsed / 1000
        ).toFixed(3);
    },
    33
  );
}


$("matchButton").onclick =
  async () => {
    $("matchButton").disabled =
      true;

    const result =
      await emitAck(
        "matchmaking:join",
        {
          name: $("name").value,
        }
      );

    if (!result.ok) {
      $("matchButton").disabled =
        false;

      setMessage(result.error);
      return;
    }

    playerToken =
      result.playerToken ||
      playerToken;

    status = "matching";

    setMessage(
      "対戦相手を探しています…"
    );

    $("matchingIndicator")
      .classList
      .remove("hidden");

    $("cancelMatch")
      .classList
      .remove("hidden");
  };


$("cancelMatch").onclick =
  async () => {
    await emitAck(
      "matchmaking:cancel"
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


socket.on(
  "matchmaking:waiting",
  () => {
    setMessage(
      "対戦相手を探しています…"
    );
  }
);


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


socket.on(
  "matchmaking:matched",
  (data) => {
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


socket.on(
  "match:countdown",
  (data) => {
    roomId = data.roomId;

    board =
      data.puzzle.board.map(
        (row) => row.slice()
      );

    selected = [];

    startAt = data.startAt;
    status = "countdown";

    $("result")
      .classList
      .add("hidden");

    $("fText").value = "";

    show("game");
    renderBoard();
    runCountdown();
  }
);


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
        $("countdown")
          .textContent = "";
      },
      600
    );
  }
);


socket.on(
  "opponent:progress",
  (progress) => {
    $("opponentRemaining")
      .textContent =
        String(progress.remaining);

    $("opponentProgress")
      .style.width =
        `${progress.progress}%`;
  }
);


socket.on(
  "match:finished",
  (result) => {
    status = "finished";

    clearInterval(timerHandle);

    renderBoard();

    $("result")
      .classList
      .remove("hidden");

    const won =
      result.winnerToken ===
      playerToken;

    $("resultTitle")
      .textContent =
        won
          ? "勝利!"
          : "相手が先にクリアしました";

    $("resultTitle")
      .className =
        won
          ? "winner"
          : "loser";
  }
);


$("clearSelection").onclick =
  () => {
    selected = [];

    $("fText").value = "";

    setGameMessage("");

    renderBoard();
  };


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

    $("submitMove").disabled =
      false;

    if (!result.ok) {
      setGameMessage(
        result.error
      );

      if (fCount > 0) {
        $("fText").focus();
        $("fText").select();
      }

      updateAnswerControls();
      return;
    }

    board = result.board;
    selected = [];

    $("fText").value = "";

    setGameMessage(
      `「${result.word}」正解　残り${result.remaining}マス`
    );

    renderBoard();
  };


$("fText").addEventListener(
  "keydown",
  (event) => {
    if (event.key === "Enter") {
      $("submitMove").click();
    }
  }
);


$("rematch").onclick =
  async () => {
    const result =
      await emitAck(
        "match:rematch"
      );

    if (!result.ok) {
      setGameMessage(
        result.error
      );

      return;
    }

    $("result")
      .classList
      .add("hidden");

    $("countdown").textContent =
      "相手の再戦操作を待っています…";
  };


$("newMatch").onclick =
  async () => {
    await emitAck(
      "match:leave"
    );

    clearSession();

    status = "idle";
    board = null;
    selected = [];

    $("result")
      .classList
      .add("hidden");

    $("matchButton").disabled =
      false;

    $("matchingIndicator")
      .classList
      .add("hidden");

    setMessage("");
    setGameMessage("");

    show("lobby");
  };


socket.on(
  "connect",
  async () => {
    try {
      const saved = JSON.parse(
        sessionStorage.getItem(
          "keshimasuDuel"
        ) || "null"
      );

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
        return;
      }

      roomId = saved.roomId;
      playerToken =
        saved.playerToken;

      status =
        result.state.status;

      startAt =
        result.startAt;

      if (result.board) {
        board =
          result.board.map(
            (row) => row.slice()
          );

        selected = [];

        show("game");
        renderBoard();

        if (
          status === "playing"
        ) {
          startTimer();
        } else if (
          status === "countdown"
        ) {
          runCountdown();
        }
      }
    } catch {
      clearSession();
    }
  }
);