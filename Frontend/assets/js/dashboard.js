layout('Dashboard');const $=s=>document.querySelector(s);let rows=[];
function countUp(el,to,fmt){const t0=performance.now();(function f(n){const p=Math.min(1,(n-t0)/1100);el.textContent=fmt(to*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(f)})(t0)}
api.getOverview().then(o=>{const k=o.kpis;const items=[['Transactions scored',k.scored,x=>Math.round(x).toLocaleString('en-IN'),'Average latency '+k.latency],['Alerts raised',k.alerts,Math.round,'False-positive rate '+k.fp],['Blocked or held',k.blocked,Math.round,'Stopped before money moved'],['Amount protected',k.protected,BDT,'Kept from scammers']];
 $('#kpis').innerHTML=items.map((x,i)=>`<div class="card kpi" style="--i:${i}"><div class=t>${x[0]}<i class=arr>↗</i></div><b>0</b><small>${x[3]}</small></div>`).join('');
 items.forEach((x,i)=>countUp($('#kpis').children[i].querySelector('b'),x[1],x[2]));
 Chart.defaults.font.family="'Plus Jakarta Sans',sans-serif";
 const mx=Math.max(...o.trend),top=[...o.trend].sort((a,b)=>b-a).slice(0,3);
 $('#bars').innerHTML=o.trend.map((v,i)=>{const r=top.indexOf(v);return `<div><i class="${['b1 pk','b2','b3'][r]||'hh'}" data-v="${v} alerts" style="height:${Math.max(14,v/mx*100)}%;animation-delay:${i*50}ms"></i>${i*2}h</div>`}).join('');
 $('#rsn').innerHTML=o.reasons.map((r,i)=>`<div class=li><div class=d>${i+1}</div><b>${esc(r[0])}</b><em>${r[1]}</em></div>`).join('');
 const tot=Object.values(o.dist).reduce((a,b)=>a+b,0),hi=(o.dist.high||0)+(o.dist.critical||0);
 $('#gv').innerHTML=Math.round(hi/tot*100)+'%<small>High or critical</small>';
 $('#lg').innerHTML=Object.keys(o.dist).map(l=>`<span style="--c:${RC[l]}">${riskLabel(l)}</span>`).join('');
 new Chart($('#c1'),{type:'doughnut',data:{labels:Object.keys(o.dist).map(riskLabel),datasets:[{data:Object.values(o.dist),backgroundColor:Object.keys(o.dist).map(l=>RC[l]),borderWidth:3,borderRadius:6}]},options:{rotation:-90,circumference:180,cutout:'64%',aspectRatio:1.8,plugins:{legend:{display:false}}}});
 $('#agents').innerHTML=o.agents.map(a=>`<div class="ag"><b>${esc(a.id)}</b><div class="track"><div class="fill" style="background:${RC[riskLvl(a.score)]}" data-w="${a.score*100}"></div></div><span>${a.score.toFixed(2)}</span></div>`).join('')+'<p class="mut">Anomaly score vs. peer median 0.20</p>';
 setTimeout(()=>document.querySelectorAll('.fill').forEach(f=>f.style.width=f.dataset.w+'%'),100)});
api.getCases({alert_type:''}).then(cs=>{const c=cs.find(c=>c.severity==='critical')||cs[0];if(c)$('#alert').innerHTML=`<h2>Priority alert</h2>${badge(c.severity)}<h3 class=at>${c.alert_type.replace(/_/g,' ')}</h3><p class=mut>${esc(c.what_happened.slice(0,110))}…</p><a class="btn dark" href="case.html?id=${c.case_id}">Investigate ${esc(c.case_id)}</a>`});
let run=true,sec=0;setInterval(()=>{if(run){sec++;$('#clk').textContent=new Date(sec*1000).toISOString().slice(11,19)}},1000);
$('#pz').onclick=e=>{run=!run;e.target.textContent=run?'❚❚':'▶';$('#st').textContent=run?'Scoring transactions live':'Paused'};
$('#rs').onclick=()=>{run=false;sec=0;$('#clk').textContent='00:00:00';$('#pz').textContent='▶';$('#st').textContent='Stopped'};
function feed(first){const f=$('#fr').value;$('#feed').innerHTML=rows.filter(t=>f==='all'||t.risk_level===f).slice(0,6).map((t,i)=>`<div class="tx ${first&&i===0?'new':''} ${t.risk_level}" data-id="${t.id}" tabindex=0><div class=a>${esc(t.sender).slice(-3)}</div><div><b>${BDT(t.amount_bdt)} · ${t.type.replace('_',' ')}</b><small>${esc(t.sender)} → ${esc(t.receiver)} · ${fmtTime(t.timestamp)}</small></div><div class=r>${badge(t.risk_level)}<small>${DEC[t.decision]}</small></div></div>`).join('')||'<p class=mut>No transactions at this risk level yet.</p>'}
$('#fr').onchange=()=>feed();
$('#feed').onclick=e=>{const tr=e.target.closest('.tx');if(!tr||!tr.dataset.id)return;const t=rows.find(x=>x.id===tr.dataset.id),c=MOCK.cases.find(c=>c.entity.id===t.sender);c?location.href='case.html?id='+c.case_id:toast(`${DEC[t.decision]}: ${t.reasons.find(r=>r.contribution>0)?.label||'low-risk pattern'}. No case opened.`)};
api.getTransactions().then(t=>{rows=t.slice(0,40);feed();setInterval(()=>{if(!run)return;const n=genTxn(Date.now()%1e5);n.timestamp=new Date().toISOString();n.id='TXN-L'+Date.now();rows.unshift(n);feed(true)},CFG().speed)});
function queue(){api.getCases({alert_type:$('#fa').value}).then(cs=>{const l=cs.filter(c=>c.severity==='critical'||c.severity==='high').slice(0,5);$('#q').innerHTML=l.map(c=>`<div class="qrow">${badge(c.severity)}<span><b>${esc(c.case_id)}</b> · ${c.alert_type.replace(/_/g,' ')}<br><span class=mut>${esc(c.what_happened.slice(0,90))}…</span></span><a class="btn" href="case.html?id=${c.case_id}">Investigate</a></div>`).join('')||'<p class=mut>No open alerts of this type. Pick another type.</p>'})}
$('#fa').onchange=queue;queue();
