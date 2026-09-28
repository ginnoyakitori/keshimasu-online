// src/solver/solver.js
//
// 8x5 word puzzle solver
//
// 仕様:
// - 盤面は縦8 × 横5
// - 文字は全マスに入っていてよい
// - 消去できるのは下5段のみ
//   row 0,1,2: 消去不可
//   row 3,4,5,6,7: 消去可能
// - 単語は8方向隣接でつながっている必要がある
// - 単語を消すと、盤面全体に重力がかかる
// - 全部消えたら solved
//

const { performance } = require("perf_hooks");

const {
  cloneBoard,
  removePathAndApplyGravity,
  isBoardEmpty,
  countFilledCells,
} = require("./gravity");

const {
  findAllMoves,
  orderMoves,
} = require("./word-search");

const telemetry = require("./telemetry");

const DEFAULT_OPTIONS = {
  // 8x5 全40マスを複数手で消す想定なので深めにする
  maxDepth: 120,

  // 難問生成・評価用に多め
  maxNodes: 500000,

  // 1単語あたり候補path上限
  maxPathsPerWord: 80,

  // 下5段だけ消去可能
  // row 0,1,2 は選択不可
  // row 3,4,5,6,7 が選択可能
  playableStartRow: 3,

  // loose探索は使わない
  // つまり、盤面上で8方向隣接している単語だけ消せる
  allowLoose: false,

  // 同一盤面の再訪問防止
  useVisited: true,
};

/**
 * NN用の文字IDマップ
 */
function createCharMap() {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
    "0123456789" +
    "あいうえおかきくけこさしすせそたちつてと" +
    "なにぬねのはひふへほまみむめもやゆよ" +
    "らりるれろわをん" +
    "がぎぐげござじずぜぞだぢづでど" +
    "ばびぶべぼぱぴぷぺぽゃゅょっー" +
    "アイウエオカキクケコサシスセソタチツテト" +
    "ナニヌネノハヒフヘホマミムメモヤユヨ" +
    "ラリルレロワヲン" +
    "ガギグゲゴザジズゼゾダヂヅデド" +
    "バビブベボパピプペポャュョッー";

  const map = {};

  [...chars].forEach((ch, i) => {
    if (!map[ch]) {
      map[ch] = i + 1;
    }
  });

  return map;
}

const CHAR_MAP = createCharMap();

/**
 * board -> tensor
 *
 * null は 0
 * 未知文字は動的に追加
 */
function boardToTensor(board) {
  return board.map((row) =>
    row.map((cell) => {
      if (cell === null) {
        return 0;
      }

      if (!CHAR_MAP[cell]) {
        CHAR_MAP[cell] =
          Object.keys(CHAR_MAP).length + 1;
      }

      return CHAR_MAP[cell];
    })
  );
}

/**
 * 盤面ハッシュ
 */
function boardHash(board) {
  return board
    .map((row) =>
      row.map((cell) => cell ?? ".").join("")
    )
    .join("|");
}

/**
 * moveが本当に下5段だけを使っているか確認
 */
function isMoveInsidePlayableArea(move, playableStartRow) {
  for (const [r] of move.path) {
    if (r < playableStartRow) {
      return false;
    }
  }

  return true;
}

/**
 * 探索の停止条件
 */
function shouldStopSearch(stats, options, depth) {
  if (stats.exploredNodes >= options.maxNodes) {
    return true;
  }

  if (depth > options.maxDepth) {
    return true;
  }

  return false;
}

/**
 * DFS本体
 */
function dfs(board, stats, options, depth, visited) {
  telemetry.recordNode(stats);
  telemetry.recordDepth(stats, depth);

  if (shouldStopSearch(stats, options, depth)) {
    telemetry.recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (isBoardEmpty(board)) {
    return {
      solved: true,
      board,
    };
  }

  const hash = boardHash(board);

  if (options.useVisited && visited.has(hash)) {
    telemetry.recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (options.useVisited) {
    visited.add(hash);
  }

  let moves = findAllMoves(board, {
    words: options.words,
    maxPathsPerWord: options.maxPathsPerWord,
    playableStartRow: options.playableStartRow,
  });

  // 念のため、上3段を含むmoveは除外
  moves = moves.filter((move) =>
    isMoveInsidePlayableArea(
      move,
      options.playableStartRow
    )
  );

  moves = orderMoves(moves);

  const branch = moves.length;

  telemetry.recordBranching(stats, branch);

  if (branch === 0) {
    telemetry.recordDeadEnd(stats);

    return {
      solved: false,
      board,
    };
  }

  if (branch === 1) {
    telemetry.recordForcedMove(stats);
  }

  const beforeFilled = countFilledCells(board);

  for (const move of moves) {
    telemetry.recordChoice(
      stats,
      boardToTensor(board),
      move,
      moves,
      depth
    );

    const nextBoard =
      removePathAndApplyGravity(
        board,
        move.path
      );

    const afterFilled =
      countFilledCells(nextBoard);

    // 安全チェック:
    // 消した後に文字数が減っていなければ不正move
    if (afterFilled >= beforeFilled) {
      continue;
    }

    const nextVisited =
      options.useVisited
        ? new Set(visited)
        : visited;

    const result = dfs(
      nextBoard,
      stats,
      options,
      depth + 1,
      nextVisited
    );

    if (result.solved) {
      telemetry.recordSolvedMove(stats, move);

      return result;
    }
  }

  telemetry.recordBacktrack(stats);

  return {
    solved: false,
    board,
  };
}

/**
 * public solve
 */
function solve(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const board = cloneBoard(inputBoard);
  const stats = telemetry.createStats();

  const start = performance.now();

  const result = dfs(
    board,
    stats,
    options,
    0,
    new Set()
  );

  const elapsed = performance.now() - start;

  telemetry.finalizeStats(stats, elapsed);

  // DFSの戻り順で solvedMoves が逆に積まれるため反転
  stats.solvedMoves.reverse();

  return {
    solved: result.solved,
    board: result.board,
    stats,
  };
}

/**
 * 解の数を数える
 *
 * 一意解チェック用:
 *
 * countSolutions(board, { limit: 2 })
 *
 * 戻り値:
 * 0 -> 解なし
 * 1 -> 一意解
 * 2 -> 複数解あり
 */
function countSolutions(inputBoard, userOptions = {}) {
  const options = {
    ...DEFAULT_OPTIONS,
    ...userOptions,
  };

  const limit = options.limit ?? 2;

  let count = 0;

  function inner(board, depth, visited) {
    if (count >= limit) {
      return;
    }

    if (depth > options.maxDepth) {
      return;
    }

    if (isBoardEmpty(board)) {
      count++;
      return;
    }

    const hash = boardHash(board);

    if (options.useVisited && visited.has(hash)) {
      return;
    }

    if (options.useVisited) {
      visited.add(hash);
    }

    let moves = findAllMoves(board, {
      words: options.words,
      maxPathsPerWord: options.maxPathsPerWord,
      playableStartRow: options.playableStartRow,
    });

    moves = moves.filter((move) =>
      isMoveInsidePlayableArea(
        move,
        options.playableStartRow
      )
    );

    moves = orderMoves(moves);

    for (const move of moves) {
      const beforeFilled =
        countFilledCells(board);

      const nextBoard =
        removePathAndApplyGravity(
          board,
          move.path
        );

      const afterFilled =
        countFilledCells(nextBoard);

      if (afterFilled >= beforeFilled) {
        continue;
      }

      const nextVisited =
        options.useVisited
          ? new Set(visited)
          : visited;

      inner(
        nextBoard,
        depth + 1,
        nextVisited
      );

      if (count >= limit) {
        return;
      }
    }
  }

  inner(
    cloneBoard(inputBoard),
    0,
    new Set()
  );

  return count;
}

module.exports = {
  solve,
  countSolutions,
  boardToTensor,
  boardHash,
};