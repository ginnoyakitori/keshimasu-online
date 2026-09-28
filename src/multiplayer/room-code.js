"use strict";
const crypto = require("node:crypto");
const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
function randomText(length) {
  const bytes = crypto.randomBytes(length);
  let out = "";
  for (let i = 0; i < length; i++) out += ALPHABET[bytes[i] % ALPHABET.length];
  return out;
}
function roomCode() { return randomText(6); }
function playerToken() { return crypto.randomBytes(24).toString("base64url"); }
module.exports = { roomCode, playerToken };
