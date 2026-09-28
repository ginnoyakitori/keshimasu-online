"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { roomCode, playerToken } = require("./room-code");
const { cloneBoard } = require("./puzzle-loader");
const { validateAndApply } = require("./answer-validator");

function cleanName(v) { return String(v || "プレイヤー").trim().slice(0, 20) || "プレイヤー"; }
function ackSafe(ack, data) { if (typeof ack === "function") ack(data); }
class MatchManager {
  constructor({ io, puzzleLoader, matchDir }) {
    this.io = io; this.puzzleLoader = puzzleLoader; this.matchDir = matchDir;
    this.rooms = new Map(); this.socketToPlayer = new Map();
  }
  get roomCount() { return this.rooms.size; }
  publicState(room, viewerToken) {
    return {
      roomId: room.id, status: room.status, puzzleId: room.puzzle?.id ?? null,
      startAt: room.startAt, winnerToken: room.winnerToken,
      players: [...room.players.values()].map(p => ({
        token: p.token, name: p.name, ready: p.ready, connected: p.connected,
        remaining: p.remaining, moveCount: p.moveCount, finishedAt: p.finishedAt,
        isYou: p.token === viewerToken
      }))
    };
  }
  emitState(room) {
    for (const p of room.players.values()) if (p.socketId) this.io.to(p.socketId).emit("room:state", this.publicState(room, p.token));
  }
  newPlayer(socket, name) {
    return { token: playerToken(), socketId: socket.id, name: cleanName(name), ready: false, connected: true,
      board: null, usedWords: new Set(), remaining: 40, moveCount: 0, finishedAt: null, elapsedMs: null, rematch: false };
  }
  createRoom(socket, payload = {}, ack) {
    let id; do { id = roomCode(); } while (this.rooms.has(id));
    const p = this.newPlayer(socket, payload.name);
    const room = { id, status: "waiting", players: new Map([[p.token,p]]), puzzle: null, startAt: null,
      winnerToken: null, previousPuzzleId: null, createdAt: Date.now() };
    this.rooms.set(id, room); this.socketToPlayer.set(socket.id, { roomId:id, token:p.token }); socket.join(id);
    ackSafe(ack, { ok:true, roomId:id, playerToken:p.token, state:this.publicState(room,p.token) }); this.emitState(room);
  }
  joinRoom(socket, payload = {}, ack) {
    const id = String(payload.roomId || "").trim().toUpperCase(); const room = this.rooms.get(id);
    if (!room) return ackSafe(ack,{ok:false,error:"部屋が見つかりません"});
    if (room.players.size >= 2) return ackSafe(ack,{ok:false,error:"部屋は満員です"});
    if (room.status !== "waiting") return ackSafe(ack,{ok:false,error:"対戦開始後は参加できません"});
    const p=this.newPlayer(socket,payload.name); room.players.set(p.token,p); this.socketToPlayer.set(socket.id,{roomId:id,token:p.token}); socket.join(id);
    ackSafe(ack,{ok:true,roomId:id,playerToken:p.token,state:this.publicState(room,p.token)}); this.emitState(room);
  }
  resumeRoom(socket,payload={},ack) {
    const room=this.rooms.get(String(payload.roomId||"").toUpperCase()); const p=room?.players.get(String(payload.playerToken||""));
    if(!room||!p) return ackSafe(ack,{ok:false,error:"復帰情報が無効です"});
    p.socketId=socket.id;p.connected=true;this.socketToPlayer.set(socket.id,{roomId:room.id,token:p.token});socket.join(room.id);
    ackSafe(ack,{ok:true,state:this.publicState(room,p.token),board:p.board});this.emitState(room);
  }
  setReady(socket,payload={},ack) {
    const ref=this.socketToPlayer.get(socket.id), room=ref&&this.rooms.get(ref.roomId), p=room&&room.players.get(ref.token);
    if(!p) return ackSafe(ack,{ok:false,error:"部屋に参加していません"});
    p.ready=Boolean(payload.ready); ackSafe(ack,{ok:true}); this.emitState(room);
    if(room.players.size===2 && [...room.players.values()].every(x=>x.ready)) this.start(room);
  }
  start(room) {
    room.puzzle=this.puzzleLoader.random(room.previousPuzzleId);room.previousPuzzleId=room.puzzle.id;room.status="countdown";room.startAt=Date.now()+3000;room.winnerToken=null;
    for(const p of room.players.values()){p.board=cloneBoard(room.puzzle.board);p.usedWords=new Set();p.remaining=40;p.moveCount=0;p.finishedAt=null;p.elapsedMs=null;p.ready=false;p.rematch=false;}
    for(const p of room.players.values()) this.io.to(p.socketId).emit("match:countdown",{roomId:room.id,startAt:room.startAt,puzzle:{id:room.puzzle.id,board:cloneBoard(room.puzzle.board),targetWildcards:room.puzzle.targetWildcards}});
    setTimeout(()=>{if(room.status!=="countdown")return;room.status="playing";this.io.to(room.id).emit("match:started",{startAt:room.startAt});this.emitState(room);},Math.max(0,room.startAt-Date.now()));
  }
  submitMove(socket,payload={},ack) {
    const ref=this.socketToPlayer.get(socket.id), room=ref&&this.rooms.get(ref.roomId), p=room&&room.players.get(ref.token);
    if(!room||!p) return ackSafe(ack,{ok:false,error:"部屋に参加していません"});
    if(room.status!=="playing"||Date.now()<room.startAt) return ackSafe(ack,{ok:false,error:"まだ開始していません"});
    if(p.finishedAt) return ackSafe(ack,{ok:false,error:"すでにクリアしています"});
    const result=validateAndApply({board:p.board,word:String(payload.word||""),path:payload.path,usedWords:p.usedWords,isCountry:w=>this.puzzleLoader.isCountry(w)});
    if(!result.ok) return ackSafe(ack,result);
    p.moveCount++;p.remaining=result.remaining;ackSafe(ack,{ok:true,board:cloneBoard(p.board),remaining:p.remaining,moveCount:p.moveCount});
    socket.to(room.id).emit("opponent:progress",{remaining:p.remaining,moveCount:p.moveCount,progress:(40-p.remaining)/40*100});
    if(result.cleared) this.finish(room,p);
  }
  finish(room,p) {
    p.finishedAt=Date.now();p.elapsedMs=p.finishedAt-room.startAt;
    if(!room.winnerToken){room.winnerToken=p.token;room.status="finished";this.io.to(room.id).emit("match:finished",{winnerToken:p.token,winnerName:p.name,elapsedMs:p.elapsedMs,players:[...room.players.values()].map(x=>({token:x.token,name:x.name,elapsedMs:x.elapsedMs,moveCount:x.moveCount,remaining:x.remaining}))});this.save(room);}
    this.emitState(room);
  }
  requestRematch(socket,_payload={},ack) {
    const ref=this.socketToPlayer.get(socket.id),room=ref&&this.rooms.get(ref.roomId),p=room&&room.players.get(ref.token);
    if(!p||room.status!=="finished")return ackSafe(ack,{ok:false,error:"再戦できません"});p.rematch=true;ackSafe(ack,{ok:true});
    if([...room.players.values()].every(x=>x.rematch)){room.status="waiting";for(const x of room.players.values()){x.ready=false;x.rematch=false;}this.emitState(room);}
  }
  leaveRoom(socket,_payload={},ack){this.removeSocket(socket.id,true);ackSafe(ack,{ok:true});}
  disconnect(socket){this.removeSocket(socket.id,false);}
  removeSocket(socketId,permanent){const ref=this.socketToPlayer.get(socketId);if(!ref)return;this.socketToPlayer.delete(socketId);const room=this.rooms.get(ref.roomId),p=room&&room.players.get(ref.token);if(!p)return;p.connected=false;p.socketId=null;if(permanent)room.players.delete(p.token);if(!room.players.size)this.rooms.delete(room.id);else this.emitState(room);}
  save(room){fs.mkdirSync(this.matchDir,{recursive:true});const out={format:"keshimasu-duel-result-v1",roomId:room.id,puzzleId:room.puzzle.id,startedAt:new Date(room.startAt).toISOString(),finishedAt:new Date().toISOString(),winnerToken:room.winnerToken,players:[...room.players.values()].map(p=>({token:p.token,name:p.name,elapsedMs:p.elapsedMs,moveCount:p.moveCount,remaining:p.remaining}))};fs.writeFileSync(path.join(this.matchDir,`${Date.now()}-${room.id}.json`),JSON.stringify(out,null,2),"utf8");}
}
module.exports={MatchManager};
