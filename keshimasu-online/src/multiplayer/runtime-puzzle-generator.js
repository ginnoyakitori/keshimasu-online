"use strict";

const crypto = require("node:crypto");


/* ==================================================
   基本設定
================================================== */

const ROWS = 8;
const COLS = 5;
const VISIBLE_ROWS = 5;

const MIN_WORD_LENGTH = 2;
const MAX_WORD_LENGTH = 5;

const EMPTY = "・";
const WILDCARD = "F";


const DEFAULT_OPTIONS = Object.freeze({
  /*
   * 生成するFの個数。
   */
  minWildcards: 2,
  maxWildcards: 6,

  /*
   * 1問を生成するときの最大試行回数。
   */
  maxAttempts: 300,

  /*
   * 1回の盤面構築で試す最大手数。
   */
  maxConstructionSteps: 30,

  /*
   * 想定解の条件。
   */
  minSteps: 9,
  maxSteps: 18,

  minHorizontalMoves: 4,
  minVerticalMoves: 2,

  /*
   * 縦手の割合が高すぎる盤面を除外。
   */
  maxVerticalMoveRate: 0.65,

  /*
   * 縦だけで全消去できる盤面を除外。
   */
  rejectVerticalOnlySolution: true,

  /*
   * 縦だけ探索の上限。
   */
  verticalOnlyMaxNodes: 30000,

  /*
   * 生成中に同じ国名を複数回使わない。
   */
  noDuplicateWords: true,

  /*
   * Fを置く場合も、想定解で全消去できることを
   * 最後に再確認します。
   */
  useWildcards: true,
});


/* ==================================================
   汎用処理
================================================== */

function cloneBoard(board) {
  return board.map(
    (row) => row.slice()
  );
}


function cloneColumns(columns) {
  return columns.map(
    (column) => column.slice()
  );
}


function randomInt(min, max) {
  if (max < min) {
    return min;
  }

  return crypto.randomInt(
    min,
    max + 1
  );
}


function shuffle(values) {
  const result =
    values.slice();

  for (
    let index =
      result.length - 1;
    index > 0;
    index -= 1
  ) {
    const other =
      randomInt(0, index);

    [
      result[index],
      result[other],
    ] = [
      result[other],
      result[index],
    ];
  }

  return result;
}


function countRemaining(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell !== EMPTY) {
        count += 1;
      }
    }
  }

  return count;
}


function countWildcards(board) {
  let count = 0;

  for (const row of board) {
    for (const cell of row) {
      if (cell === WILDCARD) {
        count += 1;
      }
    }
  }

  return count;
}


function createEmptyBoard() {
  return Array.from(
    {
      length: ROWS,
    },
    () =>
      Array(COLS).fill(EMPTY)
  );
}


function isBoardFull(board) {
  return (
    countRemaining(board) ===
    ROWS * COLS
  );
}


function isColumnsFull(columns) {
  return columns.every(
    (column) =>
      column.length === ROWS
  );
}


/* ==================================================
   国名辞書
================================================== */

function normalizeWords(words) {
  return [
    ...new Set(
      (
        Array.isArray(words)
          ? words
          : []
      )
        .filter(
          (word) =>
            typeof word ===
            "string"
        )
        .map(
          (word) =>
            word.trim()
        )
        .filter(
          (word) => {
            const length =
              [...word].length;

            return (
              length >=
                MIN_WORD_LENGTH &&
              length <=
                MAX_WORD_LENGTH
            );
          }
        )
    ),
  ];
}


function groupWordsByLength(words) {
  const byLength =
    new Map();

  for (const word of words) {
    const length =
      [...word].length;

    if (!byLength.has(length)) {
      byLength.set(
        length,
        []
      );
    }

    byLength
      .get(length)
      .push(word);
  }

  return byLength;
}


function pickUnusedWord(
  words,
  usedWords
) {
  const shuffled =
    shuffle(words);

  for (const word of shuffled) {
    if (!usedWords.has(word)) {
      return word;
    }
  }

  return null;
}


/* ==================================================
   パターン一致
================================================== */

function matchesPattern(
  pattern,
  word
) {
  const patternCharacters =
    [...String(pattern)];

  const wordCharacters =
    [...String(word)];

  if (
    patternCharacters.length !==
    wordCharacters.length
  ) {
    return false;
  }

  for (
    let index = 0;
    index <
    patternCharacters.length;
    index += 1
  ) {
    if (
      patternCharacters[index] !==
        WILDCARD &&
      patternCharacters[index] !==
        wordCharacters[index]
    ) {
      return false;
    }
  }

  return true;
}


/* ==================================================
   重力処理
================================================== */

function applyGravity(board) {
  for (
    let col = 0;
    col < COLS;
    col += 1
  ) {
    const characters = [];

    for (
      let row = ROWS - 1;
      row >= 0;
      row -= 1
    ) {
      const character =
        board[row][col];

      if (
        character !== EMPTY
      ) {
        characters.push(
          character
        );
      }
    }

    for (
      let row = ROWS - 1,
        index = 0;
      row >= 0;
      row -= 1,
        index += 1
    ) {
      board[row][col] =
        index <
        characters.length
          ? characters[index]
          : EMPTY;
    }
  }
}


/* ==================================================
   積み上げ状態から盤面を作成
================================================== */

/*
 * columns[col][0] が最下段の文字です。
 */
function buildBoardFromColumns(
  columns
) {
  const board =
    createEmptyBoard();

  for (
    let col = 0;
    col < COLS;
    col += 1
  ) {
    const column =
      columns[col];

    if (
      column.length !== ROWS
    ) {
      return null;
    }

    for (
      let index = 0;
      index <
      column.length;
      index += 1
    ) {
      const row =
        ROWS - 1 - index;

      board[row][col] =
        column[index];
    }
  }

  return board;
}


/*
 * 盤面が完成していない途中状態を作ります。
 */
function buildPartialBoardFromColumns(
  columns
) {
  const board =
    createEmptyBoard();

  for (
    let col = 0;
    col < COLS;
    col += 1
  ) {
    const column =
      columns[col];

    for (
      let index = 0;
      index <
        column.length &&
      index < ROWS;
      index += 1
    ) {
      const row =
        ROWS - 1 - index;

      board[row][col] =
        column[index];
    }
  }

  return board;
}


/* ==================================================
   配置可能な手の生成
================================================== */

function createHorizontalMoves({
  columns,
  byLength,
  usedWords,
}) {
  const moves = [];

  for (
    let length =
      MIN_WORD_LENGTH;
    length <=
      Math.min(
        MAX_WORD_LENGTH,
        COLS
      );
    length += 1
  ) {
    const words =
      byLength.get(length) ||
      [];

    if (words.length === 0) {
      continue;
    }

    for (
      let startCol = 0;
      startCol <=
        COLS - length;
      startCol += 1
    ) {
      /*
       * 横単語の各文字が同じ高さへ積まれる必要があります。
       *
       * 高さが異なる列へ置くと、消去時に一直線に
       * 並ばないため、同じ高さの列だけを対象にします。
       */
      const height =
        columns[startCol].length;

      let canPlace = true;

      for (
        let col = startCol;
        col <
          startCol + length;
        col += 1
      ) {
        if (
          columns[col].length !==
            height ||
          columns[col].length >=
            ROWS
        ) {
          canPlace = false;
          break;
        }
      }

      if (!canPlace) {
        continue;
      }

      const word =
        pickUnusedWord(
          words,
          usedWords
        );

      if (!word) {
        continue;
      }

      moves.push({
        direction: "H",
        word,
        length,
        startCol,
        height,
      });
    }
  }

  return moves;
}


function createVerticalMoves({
  columns,
  byLength,
  usedWords,
}) {
  const moves = [];

  for (
    let length =
      MIN_WORD_LENGTH;
    length <=
      MAX_WORD_LENGTH;
    length += 1
  ) {
    const words =
      byLength.get(length) ||
      [];

    if (words.length === 0) {
      continue;
    }

    for (
      let col = 0;
      col < COLS;
      col += 1
    ) {
      if (
        columns[col].length +
          length >
        ROWS
      ) {
        continue;
      }

      const word =
        pickUnusedWord(
          words,
          usedWords
        );

      if (!word) {
        continue;
      }

      moves.push({
        direction: "V",
        word,
        length,
        col,
        height:
          columns[col].length,
      });
    }
  }

  return moves;
}


/* ==================================================
   手の評価
================================================== */

function countDirections(rawMoves) {
  let horizontalMoves = 0;
  let verticalMoves = 0;

  for (const move of rawMoves) {
    if (
      move.direction === "H"
    ) {
      horizontalMoves += 1;
    }

    if (
      move.direction === "V"
    ) {
      verticalMoves += 1;
    }
  }

  return {
    horizontalMoves,
    verticalMoves,
  };
}


function scoreConstructionMove({
  move,
  rawMoves,
  columns,
  options,
}) {
  const directionCounts =
    countDirections(rawMoves);

  let score =
    Math.random() * 20;

  /*
   * 横手が不足している間は
   * 横配置を強く優先します。
   */
  if (
    directionCounts
      .horizontalMoves <
    options.minHorizontalMoves
  ) {
    score +=
      move.direction === "H"
        ? 300
        : -80;
  }

  /*
   * 縦手が不足している間は、
   * 縦配置にも優先度を与えます。
   */
  if (
    directionCounts
      .verticalMoves <
    options.minVerticalMoves
  ) {
    score +=
      move.direction === "V"
        ? 140
        : 0;
  }

  /*
   * 現時点で縦率が高すぎる場合は、
   * 縦手を減点します。
   */
  const totalAfter =
    rawMoves.length + 1;

  const verticalAfter =
    directionCounts
      .verticalMoves +
    (
      move.direction === "V"
        ? 1
        : 0
    );

  const verticalRate =
    verticalAfter /
    totalAfter;

  if (
    verticalRate >
    options.maxVerticalMoveRate
  ) {
    score -=
      (
        verticalRate -
        options.maxVerticalMoveRate
      ) *
      500;
  }

  /*
   * 長い単語を少し優先します。
   */
  score +=
    move.length * 15;

  /*
   * 盤面の列の高さが大きくばらつくと
   * 横手が置けなくなるため、高さの差を減点します。
   */
  const testColumns =
    cloneColumns(columns);

  applyConstructionMove(
    testColumns,
    move
  );

  const heights =
    testColumns.map(
      (column) =>
        column.length
    );

  const heightRange =
    Math.max(...heights) -
    Math.min(...heights);

  score -=
    heightRange * 14;

  /*
   * 横手は複数列の高さを同時に上げるため、
   * 混合盤面を作りやすくなります。
   */
  if (
    move.direction === "H"
  ) {
    score += 35;
  }

  return score;
}


function chooseConstructionMove({
  moves,
  rawMoves,
  columns,
  options,
}) {
  if (moves.length === 0) {
    return null;
  }

  const scored =
    moves.map(
      (move) => ({
        move,
        score:
          scoreConstructionMove({
            move,
            rawMoves,
            columns,
            options,
          }),
      })
    );

  scored.sort(
    (left, right) =>
      right.score -
      left.score
  );

  /*
   * 最高点だけに固定すると似た盤面が増えるため、
   * 上位候補からランダムに選択します。
   */
  const topCount =
    Math.min(
      8,
      scored.length
    );

  const top =
    scored.slice(
      0,
      topCount
    );

  return top[
    randomInt(
      0,
      top.length - 1
    )
  ].move;
}


/* ==================================================
   文字の積み上げ
================================================== */

function applyConstructionMove(
  columns,
  move
) {
  const characters =
    [...move.word];

  if (
    move.direction === "H"
  ) {
    for (
      let index = 0;
      index <
      characters.length;
      index += 1
    ) {
      columns[
        move.startCol + index
      ].push(
        characters[index]
      );
    }

    return;
  }

  /*
   * 縦単語は最初の文字が上、
   * 最後の文字が下になるように積みます。
   *
   * columnsでは先頭側が下なので逆順に追加します。
   */
  for (
    let index =
      characters.length - 1;
    index >= 0;
    index -= 1
  ) {
    columns[move.col].push(
      characters[index]
    );
  }
}


/* ==================================================
   想定解の作成
================================================== */

/*
 * 盤面を構築した手順の逆順ではなく、
 * 最初に積んだ文字が最下段にあるため、
 * 構築順と同じ順番で消去します。
 */
function buildExpectedPlan(
  initialBoard,
  rawMoves
) {
  const board =
    cloneBoard(initialBoard);

  const plan = [];

  for (
    let moveIndex = 0;
    moveIndex <
    rawMoves.length;
    moveIndex += 1
  ) {
    const move =
      rawMoves[moveIndex];

    const cells = [];

    if (
      move.direction === "H"
    ) {
      /*
       * この手より前に置いた文字はすでに消えているため、
       * 対象文字は各列の最下段にあります。
       */
      for (
        let index = 0;
        index <
        move.length;
        index += 1
      ) {
        cells.push([
          ROWS - 1,
          move.startCol +
            index,
        ]);
      }
    } else {
      /*
       * 縦単語は対象列の下側に現れます。
       */
      const startRow =
        ROWS -
        move.length;

      for (
        let row = startRow;
        row < ROWS;
        row += 1
      ) {
        cells.push([
          row,
          move.col,
        ]);
      }
    }

    /*
     * 表示されている下5行だけで消せる必要があります。
     */
    if (
      cells.some(
        ([row]) =>
          row <
          ROWS -
            VISIBLE_ROWS
      )
    ) {
      return null;
    }

    const pattern =
      cells
        .map(
          ([row, col]) =>
            board[row][col]
        )
        .join("");

    if (
      !matchesPattern(
        pattern,
        move.word
      )
    ) {
      return null;
    }

    plan.push({
      stepNo:
        moveIndex + 1,

      word:
        move.word,

      pattern,

      direction:
        move.direction,

      cells:
        cells.map(
          ([row, col]) => [
            row,
            col,
          ]
        ),

      start:
        cells[0].slice(),

      end:
        cells[
          cells.length - 1
        ].slice(),
    });

    for (
      const [row, col]
      of cells
    ) {
      board[row][col] =
        EMPTY;
    }

    applyGravity(board);
  }

  if (
    countRemaining(board) !== 0
  ) {
    return null;
  }

  return plan;
}


/* ==================================================
   Fの配置
================================================== */

function addWildcards({
  board,
  expectedPlan,
  targetCount,
}) {
  if (targetCount <= 0) {
    return cloneBoard(board);
  }

  /*
   * 想定解で使われる全マスを候補にします。
   *
   * 同じ初期座標が複数手で使われる場合があるため、
   * 盤面上の座標として重複を除きます。
   */
  const candidatePositions =
    shuffle(
      Array.from(
        {
          length:
            ROWS * COLS,
        },
        (_, index) => ({
          row:
            Math.floor(
              index / COLS
            ),

          col:
            index % COLS,
        })
      )
    );

  const result =
    cloneBoard(board);

  let placed = 0;

  for (
    const position
    of candidatePositions
  ) {
    if (
      placed >= targetCount
    ) {
      break;
    }

    const {
      row,
      col,
    } = position;

    if (
      result[row][col] ===
      EMPTY
    ) {
      continue;
    }

    result[row][col] =
      WILDCARD;

    /*
     * Fを配置した結果でも想定解が成立するかを
     * 逐次確認します。
     */
    const verification =
      verifyExpectedPlan(
        result,
        expectedPlan
      );

    if (!verification.ok) {
      result[row][col] =
        board[row][col];

      continue;
    }

    placed += 1;
  }

  return result;
}


/* ==================================================
   想定解の検証
================================================== */

function verifyExpectedPlan(
  initialBoard,
  expectedPlan
) {
  const board =
    cloneBoard(initialBoard);

  const usedWords =
    new Set();

  for (
    let index = 0;
    index <
    expectedPlan.length;
    index += 1
  ) {
    const step =
      expectedPlan[index];

    if (
      usedWords.has(
        step.word
      )
    ) {
      return {
        ok: false,
        reason:
          "duplicate word",
        step: index + 1,
      };
    }

    if (
      !Array.isArray(
        step.cells
      ) ||
      step.cells.length <
        MIN_WORD_LENGTH ||
      step.cells.length >
        MAX_WORD_LENGTH
    ) {
      return {
        ok: false,
        reason:
          "invalid cell length",
        step: index + 1,
      };
    }

    if (
      step.cells.some(
        ([row, col]) =>
          row < 0 ||
          row >= ROWS ||
          col < 0 ||
          col >= COLS ||
          row <
            ROWS -
              VISIBLE_ROWS
      )
    ) {
      return {
        ok: false,
        reason:
          "invalid or hidden cell",
        step: index + 1,
      };
    }

    const pattern =
      step.cells
        .map(
          ([row, col]) =>
            board[row][col]
        )
        .join("");

    if (
      !matchesPattern(
        pattern,
        step.word
      )
    ) {
      return {
        ok: false,
        reason:
          "pattern mismatch",
        step: index + 1,
        pattern,
        word: step.word,
      };
    }

    for (
      const [row, col]
      of step.cells
    ) {
      if (
        board[row][col] ===
        EMPTY
      ) {
        return {
          ok: false,
          reason:
            "empty cell",
          step: index + 1,
        };
      }

      board[row][col] =
        EMPTY;
    }

    applyGravity(board);

    usedWords.add(
      step.word
    );
  }

  const remaining =
    countRemaining(board);

  return {
    ok: remaining === 0,
    remaining,
    steps:
      expectedPlan.length,
  };
}


/* ==================================================
   想定解の難易度条件
================================================== */

function validatePlanDifficulty(
  plan,
  options
) {
  if (
    !Array.isArray(plan)
  ) {
    return {
      ok: false,
      reason:
        "plan is not array",
    };
  }

  const horizontalMoves =
    plan.filter(
      (step) =>
        step.direction === "H"
    ).length;

  const verticalMoves =
    plan.filter(
      (step) =>
        step.direction === "V"
    ).length;

  const totalMoves =
    plan.length;

  const verticalRate =
    totalMoves === 0
      ? 1
      : verticalMoves /
        totalMoves;

  if (
    totalMoves <
    options.minSteps
  ) {
    return {
      ok: false,
      reason:
        "too few steps",
    };
  }

  if (
    totalMoves >
    options.maxSteps
  ) {
    return {
      ok: false,
      reason:
        "too many steps",
    };
  }

  if (
    horizontalMoves <
    options.minHorizontalMoves
  ) {
    return {
      ok: false,
      reason:
        "too few horizontal moves",
    };
  }

  if (
    verticalMoves <
    options.minVerticalMoves
  ) {
    return {
      ok: false,
      reason:
        "too few vertical moves",
    };
  }

  if (
    verticalRate >
    options.maxVerticalMoveRate
  ) {
    return {
      ok: false,
      reason:
        "vertical rate too high",
    };
  }

  return {
    ok: true,
    totalMoves,
    horizontalMoves,
    verticalMoves,
    verticalRate,
  };
}


/* ==================================================
   縦だけ候補の生成
================================================== */

function createVerticalCandidates(
  board,
  wordsByLength,
  usedWords
) {
  const candidates = [];

  const visibleStartRow =
    ROWS - VISIBLE_ROWS;

  for (
    let col = 0;
    col < COLS;
    col += 1
  ) {
    for (
      let startRow =
        visibleStartRow;
      startRow < ROWS;
      startRow += 1
    ) {
      let pattern = "";

      for (
        let endRow =
          startRow;
        endRow < ROWS;
        endRow += 1
      ) {
        const character =
          board[endRow][col];

        if (
          character === EMPTY
        ) {
          break;
        }

        pattern += character;

        const length =
          [...pattern].length;

        if (
          length <
          MIN_WORD_LENGTH
        ) {
          continue;
        }

        if (
          length >
          MAX_WORD_LENGTH
        ) {
          break;
        }

        const words =
          wordsByLength.get(
            length
          ) || [];

        for (const word of words) {
          if (
            usedWords.has(word)
          ) {
            continue;
          }

          if (
            !matchesPattern(
              pattern,
              word
            )
          ) {
            continue;
          }

          const cells = [];

          for (
            let row =
              startRow;
            row <= endRow;
            row += 1
          ) {
            cells.push([
              row,
              col,
            ]);
          }

          candidates.push({
            word,
            cells,
          });
        }
      }
    }
  }

  return candidates;
}


/* ==================================================
   縦だけでクリア可能か検査
================================================== */

function boardKey(
  board,
  usedWords
) {
  return (
    board
      .map(
        (row) =>
          row.join("")
      )
      .join("/") +
    "::" +
    [...usedWords]
      .sort()
      .join(",")
  );
}


function canClearWithVerticalOnly({
  initialBoard,
  wordsByLength,
  maxNodes,
}) {
  let nodes = 0;

  const visited =
    new Set();

  function search(
    board,
    usedWords
  ) {
    nodes += 1;

    if (nodes > maxNodes) {
      /*
       * 上限到達時は、縦だけで解けないと断定せず
       * 安全側でtrueとして除外します。
       */
      return true;
    }

    if (
      countRemaining(board) === 0
    ) {
      return true;
    }

    const key =
      boardKey(
        board,
        usedWords
      );

    if (visited.has(key)) {
      return false;
    }

    visited.add(key);

    const candidates =
      createVerticalCandidates(
        board,
        wordsByLength,
        usedWords
      );

    if (
      candidates.length === 0
    ) {
      return false;
    }

    for (
      const candidate
      of candidates
    ) {
      const nextBoard =
        cloneBoard(board);

      for (
        const [row, col]
        of candidate.cells
      ) {
        nextBoard[row][col] =
          EMPTY;
      }

      applyGravity(nextBoard);

      const nextUsedWords =
        new Set(usedWords);

      nextUsedWords.add(
        candidate.word
      );

      if (
        search(
          nextBoard,
          nextUsedWords
        )
      ) {
        return true;
      }
    }

    return false;
  }

  return search(
    cloneBoard(
      initialBoard
    ),
    new Set()
  );
}


/* ==================================================
   1盤面の構築
================================================== */

function generateOneConstruction({
  byLength,
  options,
}) {
  const columns =
    Array.from(
      {
        length: COLS,
      },
      () => []
    );

  const usedWords =
    new Set();

  const rawMoves = [];

  for (
    let step = 0;
    step <
      options
        .maxConstructionSteps;
    step += 1
  ) {
    if (
      isColumnsFull(columns)
    ) {
      break;
    }

    const horizontalMoves =
      createHorizontalMoves({
        columns,
        byLength,
        usedWords,
      });

    const verticalMoves =
      createVerticalMoves({
        columns,
        byLength,
        usedWords,
      });

    let moves = [
      ...horizontalMoves,
      ...verticalMoves,
    ];

    /*
     * 最後まで40マスにできない手を除外します。
     */
    moves =
      moves.filter(
        (move) => {
          if (
            move.direction ===
            "H"
          ) {
            for (
              let index = 0;
              index <
              move.length;
              index += 1
            ) {
              if (
                columns[
                  move.startCol +
                    index
                ].length + 1 >
                ROWS
              ) {
                return false;
              }
            }

            return true;
          }

          return (
            columns[
              move.col
            ].length +
              move.length <=
            ROWS
          );
        }
      );

    if (
      moves.length === 0
    ) {
      return null;
    }

    const move =
      chooseConstructionMove({
        moves,
        rawMoves,
        columns,
        options,
      });

    if (!move) {
      return null;
    }

    applyConstructionMove(
      columns,
      move
    );

    rawMoves.push(move);

    if (
      options.noDuplicateWords
    ) {
      usedWords.add(
        move.word
      );
    }

    /*
     * 40マスを超えることはありませんが、
     * 念のため確認します。
     */
    const totalCells =
      columns.reduce(
        (
          total,
          column
        ) =>
          total +
          column.length,
        0
      );

    if (
      totalCells >
      ROWS * COLS
    ) {
      return null;
    }
  }

  if (
    !isColumnsFull(columns)
  ) {
    return null;
  }

  const board =
    buildBoardFromColumns(
      columns
    );

  if (
    !board ||
    !isBoardFull(board)
  ) {
    return null;
  }

  const expectedPlan =
    buildExpectedPlan(
      board,
      rawMoves
    );

  if (!expectedPlan) {
    return null;
  }

  return {
    board,
    expectedPlan,
    rawMoves,
  };
}


/* ==================================================
   その場生成クラス
================================================== */

class RuntimePuzzleGenerator {
  constructor({
    getWords,

    minWildcards,
    maxWildcards,
    maxAttempts,

    minSteps,
    maxSteps,

    minHorizontalMoves,
    minVerticalMoves,

    maxVerticalMoveRate,

    rejectVerticalOnlySolution,

    verticalOnlyMaxNodes,
  } = {}) {
    if (
      typeof getWords !==
      "function"
    ) {
      throw new TypeError(
        "RuntimePuzzleGenerator requires getWords()"
      );
    }

    this.getWords =
      getWords;

    this.options = {
      ...DEFAULT_OPTIONS,
    };

    const integerOptions = {
      minWildcards,
      maxWildcards,
      maxAttempts,
      minSteps,
      maxSteps,
      minHorizontalMoves,
      minVerticalMoves,
      verticalOnlyMaxNodes,
    };

    for (
      const [
        key,
        value,
      ]
      of Object.entries(
        integerOptions
      )
    ) {
      if (
        Number.isInteger(value)
      ) {
        this.options[key] =
          value;
      }
    }

    if (
      Number.isFinite(
        maxVerticalMoveRate
      )
    ) {
      this.options
        .maxVerticalMoveRate =
        maxVerticalMoveRate;
    }

    if (
      typeof
        rejectVerticalOnlySolution ===
      "boolean"
    ) {
      this.options
        .rejectVerticalOnlySolution =
        rejectVerticalOnlySolution;
    }
  }


  generateVerifiedPuzzle() {
    const words =
      normalizeWords(
        this.getWords()
      );

    const byLength =
      groupWordsByLength(
        words
      );

    for (
      let length =
        MIN_WORD_LENGTH;
      length <=
        MAX_WORD_LENGTH;
      length += 1
    ) {
      if (
        (
          byLength.get(length) ||
          []
        ).length === 0
      ) {
        console.warn(
          `[runtime-generator] ${length}文字の国名がありません`
        );
      }
    }

    for (
      let attempt = 1;
      attempt <=
        this.options
          .maxAttempts;
      attempt += 1
    ) {
      const generated =
        generateOneConstruction({
          byLength,
          options:
            this.options,
        });

      if (!generated) {
        continue;
      }

      /*
       * Fを置く前の想定解を確認します。
       */
      const baseVerification =
        verifyExpectedPlan(
          generated.board,
          generated.expectedPlan
        );

      if (
        !baseVerification.ok
      ) {
        continue;
      }

      const difficulty =
        validatePlanDifficulty(
          generated.expectedPlan,
          this.options
        );

      if (!difficulty.ok) {
        continue;
      }

      const minimumF =
        Math.max(
          0,
          this.options
            .minWildcards
        );

      const maximumF =
        Math.max(
          minimumF,
          this.options
            .maxWildcards
        );

      const targetWildcards =
        this.options
          .useWildcards
          ? randomInt(
              minimumF,
              maximumF
            )
          : 0;

      const boardWithWildcards =
        addWildcards({
          board:
            generated.board,

          expectedPlan:
            generated
              .expectedPlan,

          targetCount:
            targetWildcards,
        });

      /*
       * F配置後にも想定解を再生します。
       */
      const wildcardVerification =
        verifyExpectedPlan(
          boardWithWildcards,
          generated.expectedPlan
        );

      if (
        !wildcardVerification.ok
      ) {
        continue;
      }

      /*
       * 縦だけでクリアできる盤面を除外します。
       */
      if (
        this.options
          .rejectVerticalOnlySolution
      ) {
        const verticalOnly =
          canClearWithVerticalOnly({
            initialBoard:
              boardWithWildcards,

            wordsByLength:
              byLength,

            maxNodes:
              this.options
                .verticalOnlyMaxNodes,
          });

        if (verticalOnly) {
          continue;
        }
      }

      const actualFCount =
        countWildcards(
          boardWithWildcards
        );

      const id =
        "runtime-mixed-" +
        Date.now() +
        "-" +
        crypto
          .randomBytes(4)
          .toString("hex");

      return {
        id,

        source:
          "runtime-mixed-reverse-construction",

        board:
          cloneBoard(
            boardWithWildcards
          ),

        targetWildcards:
          actualFCount,

        expectedPlan:
          generated
            .expectedPlan,

        verifiedSolution:
          generated
            .expectedPlan,

        solutionSteps:
          difficulty.totalMoves,

        horizontalMoves:
          difficulty
            .horizontalMoves,

        verticalMoves:
          difficulty
            .verticalMoves,

        verticalMoveRate:
          difficulty
            .verticalRate,

        generatedAt:
          new Date()
            .toISOString(),

        generationAttempt:
          attempt,

        solved: true,
      };
    }

    throw new Error(
      "横・縦混合で全消去可能な盤面を生成できませんでした"
    );
  }
}


/* ==================================================
   エクスポート
================================================== */

module.exports = {
  RuntimePuzzleGenerator,
  cloneBoard,
  verifyExpectedPlan,
  validatePlanDifficulty,
  canClearWithVerticalOnly,
};