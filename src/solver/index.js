// src/solver/index.js

const solver = require("./solver");
const solverPolicy = require("./solver-policy");
const gravity = require("./gravity");
const wordSearch = require("./word-search");
const telemetry = require("./telemetry");

module.exports = {
  ...solver,
  ...solverPolicy,
  ...gravity,
  ...wordSearch,
  ...telemetry,
};