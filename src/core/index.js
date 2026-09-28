// src/core/index.js
//
// core export
//

const board = require("./board");
const rules = require("./rules");
const constraints = require("./constraints");
const tensor = require("./tensor");

module.exports = {
  ...board,
  ...rules,
  ...constraints,
  ...tensor,
};