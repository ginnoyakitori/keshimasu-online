"use strict";

/*
 * ケシマスのソルバーと難易度の測定。
 *
 * 盤面を「列ごとの配列（下から上）」で持ちます。
 * 手 = 見えている行の一直線（横・縦、2〜5マス）と、それが表すワード。
 * Fを含むときは、一致するワードの1つ1つが別の手になります。
 *
 * 難易度の考え方（guessCost）:
 *   想定解の道筋を1手ずつたどり、その局面ごとに
 *     ・指せる手の数
 *     ・指したあとも全消しできる手の数（正しい手）
 *   を数えます。当てずっぽうに手を選ぶと、その局面で平均して
 *     (手の数 - 正しい手の数) / (正しい手の数 + 1)
 *   回だけ「行き止まりへ進む手」を選んでしまいます。これを全局面で足した値が
 *   guessCost で、大きいほど「罠が多く、正しい手が少ない」盤面です。
 *
 *   ワードの覚えやすさ（有名な国名かどうか）は測っていません。
 */

const EMPTY = "";

class Solver {
  constructor({ words, maxWordLength = 5 }) {
    this.words = [
      ...new Set(
        words
          .map((word) => String(word).trim())
          .filter((word) => {
            const length = [...word].length;

            return length >= 2 && length <= maxWordLength;
          })
      ),
    ];

    this.wordChars = this.words.map((word) => [...word]);
    this.byLength = new Map();
    this.index = new Map(); // `${length}:${pos}:${char}` → 単語番号の配列

    this.words.forEach((word, id) => {
      const chars = this.wordChars[id];
      const length = chars.length;

      if (!this.byLength.has(length)) {
        this.byLength.set(length, []);
      }

      this.byLength.get(length).push(id);

      chars.forEach((char, pos) => {
        const key = `${length}:${pos}:${char}`;

        if (!this.index.has(key)) {
          this.index.set(key, []);
        }

        this.index.get(key).push(id);
      });
    });

    this.matchCache = new Map();
  }

  // パターン（Fはワイルドカード）に合う単語番号の一覧
  matchWords(chars) {
    const key = chars.join("");

    if (this.matchCache.has(key)) {
      return this.matchCache.get(key);
    }

    const length = chars.length;
    let lists = [];

    chars.forEach((char, pos) => {
      if (char !== "F") {
        lists.push(this.index.get(`${length}:${pos}:${char}`) || []);
      }
    });

    let result;

    if (lists.length === 0) {
      result = this.byLength.get(length) || [];
    } else {
      lists.sort((left, right) => left.length - right.length);

      const [smallest, ...others] = lists;
      const fixed = chars
        .map((char, pos) => [char, pos])
        .filter(([char]) => char !== "F");

      result = smallest.filter((id) =>
        fixed.every(([char, pos]) => this.wordChars[id][pos] === char)
      );
    }

    this.matchCache.set(key, result);

    return result;
  }

  /*
   * 指せる手をすべて返します。
   * state: { cols: string[][] }（各列は下から上）
   */
  legalMoves(cols, visibleRows, used) {
    const moves = [];
    const colCount = cols.length;

    // 横: 高さ h の行を、連続する列で読む
    for (let h = 0; h < visibleRows; h += 1) {
      for (let start = 0; start < colCount; start += 1) {
        const chars = [];

        for (let col = start; col < colCount && col < start + 5; col += 1) {
          if (cols[col].length <= h) {
            break;
          }

          chars.push(cols[col][h]);

          if (chars.length >= 2) {
            for (const id of this.matchWords(chars)) {
              if (!used.includes(id)) {
                moves.push({ dir: "H", h, start, length: chars.length, id });
              }
            }
          }
        }
      }
    }

    // 縦: 1つの列の、見えている行の中の連続した文字
    for (let col = 0; col < colCount; col += 1) {
      const column = cols[col];
      const top = Math.min(column.length, visibleRows);

      for (let start = 0; start < top; start += 1) {
        const chars = [];

        for (let h = start; h < top && h < start + 5; h += 1) {
          chars.push(column[h]);

          if (chars.length >= 2) {
            // 縦のワードは上から下へ読む（配列は下から上なので逆順にする）
            const reading = chars.slice().reverse();

            for (const id of this.matchWords(reading)) {
              if (!used.includes(id)) {
                moves.push({ dir: "V", col, h: start, length: chars.length, id });
              }
            }
          }
        }
      }
    }

    return moves;
  }

  applyMove(cols, move) {
    const next = cols.map((column) => column.slice());

    if (move.dir === "H") {
      for (let col = move.start; col < move.start + move.length; col += 1) {
        next[col].splice(move.h, 1);
      }
    } else {
      next[move.col].splice(move.h, move.length);
    }

    return next;
  }

  stateKey(cols, used) {
    return (
      cols.map((column) => column.join("")).join("/") +
      "#" +
      used.slice().sort((a, b) => a - b).join(",")
    );
  }

  /*
   * 全消しできるか。true / false / null（探索の上限に達して不明）
   */
  isSolvable(cols, visibleRows, used, budget, memo) {
    if (cols.every((column) => column.length === 0)) {
      return true;
    }

    const key = this.stateKey(cols, used);

    if (memo.has(key)) {
      return memo.get(key);
    }

    if (budget.nodes <= 0) {
      return null;
    }

    budget.nodes -= 1;

    // 長いワードから試すと、解が早く見つかりやすい
    const moves = this.legalMoves(cols, visibleRows, used).sort(
      (left, right) => right.length - left.length
    );

    let sawUnknown = false;

    for (const move of moves) {
      const result = this.isSolvable(
        this.applyMove(cols, move),
        visibleRows,
        [...used, move.id],
        budget,
        memo
      );

      if (result === true) {
        memo.set(key, true);

        return true;
      }

      if (result === null) {
        sawUnknown = true;
      }
    }

    if (sawUnknown) {
      return null;
    }

    memo.set(key, false);

    return false;
  }

  // 盤面（board[row][col]、空は ""）を列の配列（下から上）へ
  toColumns(board) {
    const rows = board.length;
    const cols = board[0].length;

    return Array.from({ length: cols }, (_, col) => {
      const column = [];

      for (let row = rows - 1; row >= 0; row -= 1) {
        if (board[row][col] !== EMPTY) {
          column.push(board[row][col]);
        }
      }

      return column;
    });
  }

  /*
   * 想定解（[word, dir, row, col] の並び）を1手ずつたどって難易度を測ります。
   *
   * 戻り値:
   *   guessCost   … 当てずっぽうで進んだときの、行き止まりの手の期待回数
 *   guessBits   … 各局面の log2(指せる手 / 正しい手) の合計。難易度の中心になる値
 *   exhausted   … 探索の上限に達した（値が不正確になる）
   *   rootMoves   … 最初に指せる手の数
   *   rootWinning … そのうち全消しにつながる手の数
   *   steps       … 想定解の手数
   *   unknown     … 探索の上限で判定できなかった手の数
   */
  analyze({ board, solution, visibleRows, nodeBudget = 200000 }) {
    let cols = this.toColumns(board);
    const rows = board.length;
    const memo = new Map();
    const budget = { nodes: nodeBudget };

    let used = [];
    let guessCost = 0;
    let guessBits = 0;
    let unknown = 0;
    let rootMoves = 0;
    let rootWinning = 0;
    let minWinRatio = 1;

    for (let step = 0; step < solution.length; step += 1) {
      const [word, dir, row, col] = solution[step];
      const moves = this.legalMoves(cols, visibleRows, used);

      let winning = 0;
      let intended = null;

      for (const move of moves) {
        const next = this.applyMove(cols, move);
        const result = this.isSolvable(
          next,
          visibleRows,
          [...used, move.id],
          budget,
          memo
        );

        if (result === true) {
          winning += 1;
        } else if (result === null) {
          unknown += 1;
        }

        // 想定解の手に当たるかどうかを調べる
        const wordId = this.words.indexOf(word);
        const isIntended =
          move.id === wordId &&
          (dir === "H"
            ? move.dir === "H" && move.h === 0 && move.start === col
            : move.dir === "V" &&
              move.col === col &&
              move.h === 0 &&
              move.length === [...word].length);

        if (isIntended) {
          intended = move;
        }
      }

      if (!intended) {
        // 想定解が再現できない（データ不整合）
        return { ok: false, reason: `step ${step + 1} not found` };
      }

      // 想定解の手は必ず「正しい手」として数える
      winning = Math.max(1, winning);

      if (step === 0) {
        rootMoves = moves.length;
        rootWinning = winning;
      }

      guessCost += (moves.length - winning) / (winning + 1);
      guessBits += Math.log2(moves.length / winning);
      minWinRatio = Math.min(minWinRatio, winning / Math.max(1, moves.length));

      cols = this.applyMove(cols, intended);
      used = [...used, intended.id];
    }

    return {
      ok: cols.every((column) => column.length === 0),
      guessCost,
      guessBits,
      exhausted: budget.nodes <= 0,
      rootMoves,
      rootWinning,
      steps: solution.length,
      unknown,
      minWinRatio,
      nodesUsed: nodeBudget - budget.nodes,
      rows,
    };
  }
}

module.exports = { Solver };