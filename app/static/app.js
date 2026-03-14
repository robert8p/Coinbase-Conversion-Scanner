function fmt(v,d=2){return (v===null||v===undefined||Number.isNaN(v))?"—":Number(v).toFixed(d)}
function prob(v){return (v===null||v===undefined)?"—":(100*Number(v)).toFixed(1)+"%"}
async function refresh(){
  const s=await (await fetch('/api/status')).json();
  document.getElementById('sourceStatus').textContent=(s.data_source?.ok?'OK':'WARN')+` (${s.data_source?.message||''})`;
  document.getElementById('modelStatus').textContent=s.model?.pt2?.trained?'pt2 ✅':'pt2 heuristic';
  document.getElementById('lastBar').textContent=s.data_source?.last_bar_timestamp||'—';
  const c=s.coverage||{};
  document.getElementById('coverageStatus').textContent=`U ${c.universe_count||0} · Stage1 ${c.stage1_candidate_count||0} → Stage2 ${c.stage2_scored_count||0}`;
  const tc=c.threshold_counts||{};
  document.getElementById('tailCounts').textContent=`0.60 ${tc.ge_0_60||0} · 0.70 ${tc.ge_0_70||0} · 0.75 ${tc.ge_0_75||0} · 0.80 ${tc.ge_0_80||0}`;
  document.getElementById('skipReasons').textContent=Object.entries(c.top_skip_reasons||{}).map(([k,v])=>`${k}:${v}`).join(' · ')||'—';
  const t=await (await fetch('/api/training/status')).json();
  document.getElementById('trainRunning').textContent=t.running?'Running':'Idle';

  const d=await (await fetch('/api/scores')).json();
  const rows=d.rows||[];
  const tbody=document.getElementById('rows'); tbody.innerHTML='';
  rows.forEach(r=>{const tr=document.createElement('tr'); tr.className='border-b'; tr.innerHTML=`<td>${r.symbol}</td><td>${fmt(r.price,4)}</td><td>${prob(r.prob_2)}</td><td>${r.risk||''}</td><td>${r.risk_reasons||''}</td><td>${fmt(r.downside_risk,2)}</td><td>${r.uncertainty||''}</td><td>${fmt(r.btc_relative,3)}</td><td>${r.reasons||''}</td>`; tbody.appendChild(tr);});
}
async function startTraining(){
  const fd=new FormData(); fd.append('admin_password',document.getElementById('adminPassword').value||'');
  const r=await fetch('/train',{method:'POST',body:fd}); const j=await r.json(); if(!r.ok) alert(j.error||'failed');
  refresh();
}
window.addEventListener('DOMContentLoaded',()=>{document.getElementById('startTraining').addEventListener('click',startTraining);refresh();setInterval(refresh,10000);});
