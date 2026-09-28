// src/core/config-loader.js

const fs = require("fs");
const path = require("path");

function loadPuzzleConfig() {
  const file = path.join(
    __dirname,
    "../../data/config/puzzle-config.json"
  );

  return JSON.parse(fs.readFileSync(file, "utf8"));
}

module.exports = {
  loadPuzzleConfig,
};