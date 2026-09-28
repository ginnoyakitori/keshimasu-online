"use strict";

const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const express = require("express");
const { Server } = require("socket.io");
const { PuzzleLoader } = require("./src/multiplayer/puzzle-loader");
const { MatchManager } = require("./src/multiplayer/match-manager");

const ROOT = __dirname;
const PORT = Number(process.env.PORT || 3000);
const PUZZLE_DIR = path.resolve(process.env.PUZZLE_DIR || path.join(ROOT, "data", "ga_reverse"));
const COUNTRY_FILE = path.resolve(process.env.COUNTRY_FILE || path.join(ROOT, "data", "words", "countries.json"));
const MATCH_DIR = path.resolve(process.env.MATCH_DIR || path.join(ROOT, "data", "matches"));

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: false },
  maxHttpBufferSize: 100_000
});
app.disable("x-powered-by");
app.use(express.json({ limit: "64kb" }));
app.use(express.static(path.join(ROOT, "public"), { extensions: ["html"] }));

const puzzleLoader = new PuzzleLoader({ puzzleDir: PUZZLE_DIR, countryFile: COUNTRY_FILE });
const matches = new MatchManager({ io, puzzleLoader, matchDir: MATCH_DIR });

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, puzzles: puzzleLoader.count, rooms: matches.roomCount });
});

io.on("connection", (socket) => {
  socket.on("room:create", (payload, ack) => matches.createRoom(socket, payload, ack));
  socket.on("room:join", (payload, ack) => matches.joinRoom(socket, payload, ack));
  socket.on("room:resume", (payload, ack) => matches.resumeRoom(socket, payload, ack));
  socket.on("player:ready", (payload, ack) => matches.setReady(socket, payload, ack));
  socket.on("move:submit", (payload, ack) => matches.submitMove(socket, payload, ack));
  socket.on("match:rematch", (payload, ack) => matches.requestRematch(socket, payload, ack));
  socket.on("match:leave", (payload, ack) => matches.leaveRoom(socket, payload, ack));
  socket.on("disconnect", () => matches.disconnect(socket));
});

puzzleLoader.load();
fs.mkdirSync(MATCH_DIR, { recursive: true });
server.listen(PORT, () => {
  console.log(`Keshimasu Duel: http://localhost:${PORT}`);
  console.log(`Puzzles: ${PUZZLE_DIR} (${puzzleLoader.count})`);
  console.log(`Countries: ${COUNTRY_FILE} (${puzzleLoader.countryCount})`);
});
