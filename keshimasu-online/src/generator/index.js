// src/generator/index.js

const generator = require("./generator");
const reverseGenerator = require("./reverse-generator");
const evaluate = require("./evaluate");
const words = require("./words");

module.exports = {
  ...generator,
  ...reverseGenerator,
  ...evaluate,
  ...words,
};