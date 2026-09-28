"use strict";
const socket=io();
const $=id=>document.getElementById(id);
let roomId=null,playerToken=null,board=null,selected=[],startAt=null,timerHandle=null,status="waiting";
function show(id){for(const x of ["lobby","room","game"])$(x).classList.toggle("hidden",x!==id)}
function msg(t){$("message").textContent=t||""}function gameMsg(t){$("gameMessage").textContent=t||""}
function saveSession(){sessionStorage.setItem("keshimasuDuel",JSON.stringify({roomId,playerToken}))}
function emitAck(event,payload){return new Promise(resolve=>socket.emit(event,payload,resolve))}
$("create").onclick=async()=>{const r=await emitAck("room:create",{name:$("name").value});if(!r.ok)return msg(r.error);roomId=r.roomId;playerToken=r.playerToken;saveSession();show("room");renderState(r.state)};
$("join").onclick=async()=>{const r=await emitAck("room:join",{roomId:$("roomCode").value,name:$("name").value});if(!r.ok)return msg(r.error);roomId=r.roomId;playerToken=r.playerToken;saveSession();show("room");renderState(r.state)};
$("ready").onclick=()=>emitAck("player:ready",{ready:true});
$("rematch").onclick=async()=>{const r=await emitAck("match:rematch",{});if(!r.ok)gameMsg(r.error);else{$("result").classList.add("hidden");show("room")}};
function renderState(s){if(!s)return;status=s.status;roomId=s.roomId;$("roomId").textContent=s.roomId;$("players").innerHTML=s.players.map(p=>`<div class="player">${escapeHtml(p.name)} ${p.isYou?"(あなた)":""} ${p.ready?"✅":""} ${p.connected?"":"切断中"}</div>`).join("");$("roomStatus").textContent=s.players.length<2?"相手を待っています":"2人揃いました。準備OKを押してください";}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
socket.on("room:state",renderState);
socket.on("match:countdown",data=>{board=data.puzzle.board.map(r=>r.slice());selected=[];startAt=data.startAt;show("game");renderBoard();countdown()});
socket.on("match:started",data=>{startAt=data.startAt;status="playing";startTimer();$("countdown").textContent="開始!";setTimeout(()=>$("countdown").textContent="",600)});
socket.on("opponent:progress",p=>{$("opponentRemaining").textContent=p.remaining});
socket.on("match:finished",r=>{status="finished";clearInterval(timerHandle);$("result").classList.remove("hidden");const win=r.winnerToken===playerToken;$("resultTitle").textContent=win?"勝利!":"相手が先にクリアしました";$("resultTitle").className=win?"winner":"loser"});
function countdown(){const h=setInterval(()=>{const n=Math.ceil((startAt-Date.now())/1000);$("countdown").textContent=n>0?n:"開始!";if(n<=0)clearInterval(h)},100)}
function startTimer(){clearInterval(timerHandle);timerHandle=setInterval(()=>{$("timer").textContent=(Math.max(0,Date.now()-startAt)/1000).toFixed(3)},33)}
function renderBoard(){const el=$("board");el.innerHTML="";board.forEach((row,r)=>row.forEach((v,c)=>{const b=document.createElement("button");b.className="cell"+(r>=3?" playable":"")+(v==="・"?" empty":"")+(selected.some(p=>p[0]===r&&p[1]===c)?" selected":"");b.textContent=v;b.disabled=v==="・"||status==="finished";b.onclick=()=>selectCell(r,c);el.appendChild(b)}))}
function selectCell(r,c){if(!selected.length){selected=[[r,c]];renderBoard();return}const [r0,c0]=selected[0];if(selected.length===1){const dr=r-r0,dc=c-c0;if(!((Math.abs(dr)===1&&dc===0)||(Math.abs(dc)===1&&dr===0))){selected=[[r,c]];renderBoard();return}}const [rl,cl]=selected[selected.length-1];const dr=r-rl,dc=c-cl;const firstDr=selected.length>1?selected[1][0]-r0:r-r0;const firstDc=selected.length>1?selected[1][1]-c0:c-c0;if(dr!==firstDr||dc!==firstDc||selected.length>=5){selected=[[r,c]]}else selected.push([r,c]);renderBoard()}
$("clearSelection").onclick=()=>{selected=[];renderBoard()};
$("submitMove").onclick=async()=>{if(status!=="playing")return gameMsg("まだ開始していません");const word=$("word").value.trim();const r=await emitAck("move:submit",{word,path:selected});if(!r.ok)return gameMsg(r.error);board=r.board;selected=[];$("word").value="";gameMsg(`正解 残り${r.remaining}マス`);renderBoard()};
$("word").addEventListener("keydown",e=>{if(e.key==="Enter")$("submitMove").click()});
socket.on("connect",async()=>{try{const s=JSON.parse(sessionStorage.getItem("keshimasuDuel")||"null");if(s?.roomId&&s?.playerToken){const r=await emitAck("room:resume",s);if(r.ok){roomId=s.roomId;playerToken=s.playerToken;renderState(r.state);if(r.board){board=r.board;show("game");renderBoard()}else show("room")}}}catch{}});
