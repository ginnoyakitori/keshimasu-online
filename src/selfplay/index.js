// src/selfplay/index.js

const selfplay = require("./selfplay");
const curriculum = require("./curriculum");
const replayBuffer = require("./replay-buffer");
const league = require("./league");

module.exports = {
  ...selfplay,
  ...curriculum,
  ...replayBuffer,
  ...league,
};