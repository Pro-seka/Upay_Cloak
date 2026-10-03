/* Only file that knows mock vs real: set USE_MOCK=false and BASE_URL. */
const USE_MOCK=true,BASE_URL='/api',AUD={};let FULLG=null;
const wait=ms=>new Promise(r=>setTimeout(r,ms)),http=(p,o)=>fetch(BASE_URL+p,o).then(r=>r.json());
const findCase=id=>MOCK.cases.find(c=>c.case_id===id);
function fullGraph(){if(FULLG)return FULLG;const hub='019XX-XXX305',N=[],E=[],add=(id,type,risk,ring)=>{while(N.some(n=>n.id===id))id=wid();N.push({id,type,label:id,risk_score:risk,total_volume_bdt:0,flags:ring?['Mule ring']:[],ring:!!ring});return id},edge=(s,t,a)=>E.push({source:s,target:t,amount_bdt:a,count:1,last_seen:'today'});
 add(hub,'wallet',.94,1);['AGT-1190','AGT-1204'].forEach(a=>{add(a,'agent',.82,1);edge(hub,a,rnd(2e4,4e4))});['DEV-552','DEV-553'].forEach(d=>{add(d,'device',.7,1);edge(d,hub,1e3)});
 const sp=Array.from({length:14},()=>add(wid(),'wallet',rnd(.45,.75),1));sp.forEach(s=>edge(s,hub,rnd(3e3,9e3)));
 const bg=Array.from({length:150},(_,i)=>add(wid(),i%12===0?'merchant':i%20===1?'agent':'wallet',rnd(.02,.5)));
 for(let i=0;i<220;i++)edge(pick(bg),pick(bg),rnd(300,8e3));bg.slice(0,8).forEach((b,i)=>edge(b,sp[i],rnd(500,2e3)));
 E.forEach(e=>{[e.source,e.target].forEach(id=>N.find(n=>n.id===id).total_volume_bdt+=e.amount_bdt)});return FULLG={nodes:N,edges:E}}
const api={
 getOverview:()=>USE_MOCK?wait(300).then(()=>{const d={low:0,medium:0,high:0,critical:0},rs={};MOCK.txns.forEach(t=>{d[t.risk_level]++;t.reasons.filter(r=>r.contribution>0).forEach(r=>rs[r.label]=(rs[r.label]||0)+1)});const bl=MOCK.txns.filter(t=>t.decision==='block'||t.decision==='hold');
  return{kpis:{scored:MOCK.txns.length*160,alerts:MOCK.cases.length,blocked:bl.length,protected:bl.reduce((a,t)=>a+t.amount_bdt,0),latency:'142 ms',fp:'3.1%'},dist:d,trend:[4,6,5,9,7,12,10,14,11,16,13,18],reasons:Object.entries(rs).sort((a,b)=>b[1]-a[1]).slice(0,6),agents:MOCK.agents}}):http('/overview'),
 getTransactions:()=>USE_MOCK?wait(150).then(()=>MOCK.txns):http('/transactions'),
 scoreTransaction:t=>USE_MOCK?(()=>{const ms=Math.round(rnd(300,600));return wait(ms).then(()=>({...scoreFlags(t),latency_ms:ms}))})():http('/score',{method:'POST',body:JSON.stringify(t)}),
 getCases:(f={})=>USE_MOCK?wait(150).then(()=>MOCK.cases.filter(c=>(!f.alert_type||c.alert_type===f.alert_type)).sort((a,b)=>b.why_risky[0].contribution-a.why_risky[0].contribution)):http('/cases'),
 getCase:id=>USE_MOCK?wait(200).then(()=>findCase(id)||Promise.reject(new Error('not found'))):http('/cases/'+id),
 applyAction:(id,a)=>USE_MOCK?wait(250).then(()=>{const c=findCase(id);c.status={hold:'investigating',block:'resolved',kyc:'investigating',escalate:'escalated',false_positive:'false_positive'}[a];(AUD[id]=AUD[id]||[]).push({ts:new Date().toISOString(),action:a,by:'Analyst (demo)'});return{status:c.status,audit:AUD[id]}}):http('/cases/'+id+'/action',{method:'POST',body:JSON.stringify({a})}),
 askAssistant:(id,q)=>USE_MOCK?wait(500).then(()=>{const c=findCase(id),e=c.timeline.map(x=>x.id),top=c.why_risky.filter(r=>r.contribution>0).slice(0,3).map(r=>r.label).join(', ')||'no raising signals';
  if(/connect|who|link/i.test(q))return{text:`${c.subgraph.nodes.length-1} entities connect to ${c.entity.id}. The largest flow is ${BDT(Math.max(...c.subgraph.edges.map(x=>x.amount_bdt)))}.`,evidence_ids:e.slice(-1)};
  if(/summar|compliance|report/i.test(q))return{text:`Case ${id} (${c.alert_type.replace(/_/g,' ')}, ${c.severity}): ${c.what_happened} Recommended: ${c.recommended_actions[0].label}.`,evidence_ids:e};
  return{text:`Flagged mainly because of: ${top}. ${c.narrative}`,evidence_ids:e.slice(0,2)}}):http('/cases/'+id+'/ask',{method:'POST',body:JSON.stringify({q})}),
 getGraph:(eid,depth)=>USE_MOCK?wait(200).then(()=>{const g=fullGraph();if(!eid||!g.nodes.some(n=>n.id===eid))return g;let seen=new Set([eid]),fr=[eid];for(let d=0;d<depth;d++){const nx=[];g.edges.forEach(e=>{if(fr.includes(e.source)&&!seen.has(e.target)){seen.add(e.target);nx.push(e.target)}if(fr.includes(e.target)&&!seen.has(e.source)){seen.add(e.source);nx.push(e.source)}});fr=nx}return{nodes:g.nodes.filter(n=>seen.has(n.id)),edges:g.edges.filter(e=>seen.has(e.source)&&seen.has(e.target))}}):http(`/graph/${eid}?depth=${depth}`)};
