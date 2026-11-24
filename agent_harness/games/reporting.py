"""Self-contained local evidence viewer; no server, account or network assets."""
import json
from pathlib import Path


def write_replay(output,summary,records,decisions=()):
    payload=json.dumps({'summary':summary,'episodes':records,'decisions':decisions},ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent game benchmark evidence</title>
<style>
:root{font:17px/1.5 system-ui,sans-serif;color:#152d3c;background:#f5f7f8}body{max-width:1120px;margin:36px auto;padding:0 22px}h1{font-size:34px;margin:0 0 10px}h2{font-size:23px}p{max-width:90ch}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{padding:11px;text-align:left;border-bottom:1px solid #d5dfe4}th{background:#173749;color:white}label{display:block;margin:16px 0 6px}select{width:100%;font:inherit;padding:10px}button{font:inherit;background:#087b81;color:white;border:0;padding:9px 18px;cursor:pointer}button:disabled{background:#78939a;cursor:default}input[type=range]{width:100%;margin:20px 0}.layout{display:grid;grid-template-columns:minmax(260px,520px) 1fr;gap:30px;margin-top:24px}#board{display:grid;gap:3px;background:#c6d4da;padding:3px;max-width:500px;aspect-ratio:1}.cell{display:flex;align-items:center;justify-content:center;background:white;font-weight:650;min-width:0}.wall{background:#294758;color:white}.agent{background:#087b81!important;color:white}.key{background:#fff0bb}.door{background:#e7d3be}.goal{background:#cdebdc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 ui-monospace,monospace}small{color:#47616e}.note{padding:14px 0;border-bottom:1px solid #b8ccd3}#frame{font-weight:650}details{margin:24px 0}summary{cursor:pointer}#outcome{font-size:20px;font-weight:650}@media(max-width:700px){.layout{grid-template-columns:1fr}table{font-size:13px}th,td{padding:6px}}
</style><style>.cell.hit{background:#f7c4bd;color:#7e261d}.cell.miss{background:#d8e9f4;color:#31566f}.cell.unknown{background:#eef3f6;color:#6b818d}</style></head><body>
<h1>Agent game benchmark</h1><p id="identity"></p><p id="disclosure" class="note"></p>
<h2>Frozen evaluation results</h2><div style="overflow-x:auto"><table><thead><tr><th>Game</th><th>Phase</th><th>Episodes</th><th>Score</th><th>Win / goal</th><th>Model calls</th><th>Illegal moves</th></tr></thead><tbody id="results"></tbody></table></div>
<h2>Replay recorded episodes</h2><p>These frames reconstruct saved actions. No agent is running in this viewer. The exact model requests and responses remain in each episode's trajectory and object files.</p>
<label for="episode">Episode</label><select id="episode"></select>
<div class="layout"><div><div id="board" role="img" aria-label="Recorded game board"></div><input id="slider" type="range" min="0" value="0"><div><button id="prev">Previous</button> <button id="next">Next</button></div></div><div><div id="outcome"></div><p id="frame"></p><pre id="action"></pre><pre id="metrics"></pre><small>Maze: A = agent, K = key, D = locked door, G = goal, # = wall. Tic-tac-toe cells use row-major indices 0 through 8.</small></div></div>
<details><summary>Accepted and rejected updates</summary><pre id="updates"></pre></details>
<details><summary>Interpretation and limitations</summary><pre id="limits"></pre></details>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('evidence').textContent),$=id=>document.getElementById(id);
const summary=data.summary, episodes=data.episodes;
$('identity').textContent=`Model: ${summary.model.model}. Status: ${summary.status}. Requests: ${summary.request_attempts}/${summary.request_cap}.`;
$('disclosure').textContent=summary.interpretation+(summary.stop_reason?' STOPPED: '+summary.stop_reason:'');
for(const [game,groups] of Object.entries(summary.results))for(const phase of ['initial','learned','without_notes'])if(groups[phase]){const r=groups[phase],tr=document.createElement('tr');for(const value of [game,phase,r.episodes,r.mean_game_score.toFixed(3),(100*r.win_or_goal_rate).toFixed(1)+'%',r.model_calls,r.illegal_moves]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}$('results').appendChild(tr)}
if(!Object.keys(summary.results).length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.textContent='No completed before/after comparison. Partial episode evidence is available below.';tr.appendChild(td);$('results').appendChild(tr)}
episodes.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1}. ${r.game} / ${r.phase} / ${r.task_id} / ${r.outcome}`;$('episode').appendChild(option)});
$('updates').textContent=JSON.stringify(data.decisions||[],null,2);
$('limits').textContent=summary.limitations.join('\n\n')+'\n\nVerification overhead:\n'+JSON.stringify(summary.verification_overhead,null,2);
function render(){const r=episodes[Number($('episode').value)];if(!r)return;let step=Number($('slider').value),rows,pos=null,key=false,action={};const history=r.game_actions;
$('slider').max=history.length;step=Math.min(step,history.length);$('slider').value=step;
if(r.game==='tic_tac_toe'){let board=Array(9).fill('.');for(const move of history.slice(0,step))board[move.cell]=move.mark;rows=[board.slice(0,3),board.slice(3,6),board.slice(6,9)];action=step?history[step-1]:{description:'Empty board before the first recorded move'};}
else if(r.game==='battleship'){const obs=step?history[step-1].observation:r.initial_observation;rows=obs.rows.map(row=>row.split(''));const move=step?history[step-1]:null;action=move?{cells:move.cells,results:move.results,error:move.error,sunk_ships:obs.sunk_ships}:{description:'Hidden fleet; all cells untested. X = hit, o = miss, ? = unknown.'};}
else{rows=r.initial_observation.grid.map(row=>row.split(''));pos=[...r.initial_observation.position];key=r.initial_observation.has_key;for(const move of history.slice(0,step)){pos=move.position;if(rows[pos[1]][pos[0]]==='K'){key=true;rows[pos[1]][pos[0]]='.'}if(rows[pos[1]][pos[0]]==='D'&&key)rows[pos[1]][pos[0]]='.'}action=step?history[step-1]:{description:'Initial map and agent position'};}
const board=$('board');board.replaceChildren();board.style.gridTemplateColumns=`repeat(${rows[0].length},1fr)`;board.style.fontSize=rows.length===3?'48px':'24px';rows.forEach((row,y)=>row.forEach((c,x)=>{const cell=document.createElement('div');cell.className='cell '+({'#':'wall','K':'key','D':'door','G':'goal'}[c]||'');cell.textContent=c==='.'?'':c;if(pos&&pos[0]===x&&pos[1]===y){cell.textContent='A';cell.classList.add('agent')}cell.title=`[${x},${y}]`;board.appendChild(cell)}));
if(r.game==='battleship')for(const cell of board.children)cell.classList.add({'X':'hit','o':'miss','?':'unknown'}[cell.textContent]);
$('outcome').textContent=`Recorded final outcome: ${r.outcome}`;$('frame').textContent=`Frame ${step} of ${history.length}${pos?' / position '+JSON.stringify(pos)+' / key '+key:''}`;$('action').textContent=JSON.stringify(action,null,2);$('metrics').textContent=`Game score: ${r.game_score}\nModel calls: ${r.model_calls}\nTokens: ${r.tokens}\nIllegal moves: ${r.illegal_moves}\nHarness version: ${r.harness_version}\n${r.error?'Agent error: '+r.error:''}`;$('prev').disabled=step===0;$('next').disabled=step===history.length;}
$('episode').addEventListener('change',()=>{$('slider').value=0;render()});$('slider').addEventListener('input',render);$('prev').addEventListener('click',()=>{$('slider').value=Number($('slider').value)-1;render()});$('next').addEventListener('click',()=>{$('slider').value=Number($('slider').value)+1;render()});render();
</script></body></html>'''
    # Decisions are embedded separately by the controller after all proposals are saved.
    Path(output,'replay.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
