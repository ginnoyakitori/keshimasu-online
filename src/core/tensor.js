// src/core/tensor.js
//
// NN用 tensor 化
//

const CHAR_MAP = {
  あ: 1,
  い: 2,
  う: 3,
  え: 4,
  お: 5,
};

const REVERSE_MAP = {
  1: "あ",
  2: "い",
  3: "う",
  4: "え",
  5: "お",
};

function boardToTensor(board) {
  return board.map((row) =>
    row.map((cell) => {
      if (cell === null) {
        return 0;
      }

      return CHAR_MAP[cell] || 0;
    })
  );
}

function tensorToBoard(tensor) {
  return tensor.map((row) =>
    row.map((v) => {
      if (v === 0) {
        return null;
      }

      return REVERSE_MAP[v];
    })
  );
}

//
// multi-channel tensor
//

function boardToMultiChannelTensor(
  board,
  getCandidates
) {
  const channels = [];

  //
  // channel 0
  // character id
  //

  const charChannel = board.map((row) =>
    row.map((cell) => {
      if (cell === null) return 0;

      return CHAR_MAP[cell] || 0;
    })
  );

  channels.push(charChannel);

  //
  // channel 1
  // empty mask
  //

  const emptyChannel = board.map((row) =>
    row.map((cell) =>
      cell === null ? 1 : 0
    )
  );

  channels.push(emptyChannel);

  //
  // channel 2
  // candidate count
  //

  const candidateChannel = board.map(
    (row, r) =>
      row.map((cell, c) => {
        if (cell !== null) {
          return 0;
        }

        return getCandidates(
          board,
          r,
          c
        ).length;
      })
  );

  channels.push(candidateChannel);

  return channels;
}

module.exports = {
  CHAR_MAP,
  REVERSE_MAP,
  boardToTensor,
  tensorToBoard,
  boardToMultiChannelTensor,
};