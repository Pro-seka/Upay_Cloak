function layout(active){
 const nav=[['index.html','Dashboard','▦'],['index.html#queue','Cases','▤'],['graph.html','Graph','◉'],['demo.html','Warning demo','▢']];
 document.body.insertAdjacentHTML('afterbegin',`<aside class="side"><div class="brand"><div class="logo">u</div>Trust Radar</div>${nav.map(n=>`<a href="${n[0]}" class="${n[1]===active?'on':''}"><span>${n[2]}</span>${n[1]}</a>`).join('')}<p class="note">upay-style demo · synthetic data</p></aside>`);
 const m=document.getElementById('main'),w=document.createElement('div');w.className='wrap';
 w.innerHTML='<header class="top"><input id="gs" aria-label="Search" placeholder="Search wallet, agent or case ID"><span class="demo-badge">Synthetic data — demo only</span></header>';
 m.replaceWith(w);w.appendChild(m);
 document.getElementById('gs').onkeydown=e=>{if(e.key==='Enter'){const v=e.target.value.trim();location.href=/^CASE/i.test(v)?'case.html?id='+v.toUpperCase():'graph.html?q='+encodeURIComponent(v)}};
}
function toast(m){const t=document.createElement('div');t.className='toast';t.textContent=m;document.body.appendChild(t);setTimeout(()=>t.remove(),2600)}
