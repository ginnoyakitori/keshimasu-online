// src/solver/index.js

const solver = require("./solver");
const gravity = require("./gravity");
const wordSearch = require("./word-search");
const telemetry = require("./telemetry");
const solverPolicy = require("./solver-policy");

module.exports = {
  ...solver,
  ...gravity,
  ...wordSearch,
  ...telemetry,
  ...solverPolicy,
};