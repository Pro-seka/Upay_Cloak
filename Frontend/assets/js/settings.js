layout('Settings');const $=s=>document.querySelector(s),F=['nm','rl','sp','lg'];
function fill(c){$('#nm').value=c.name;$('#rl').value=c.role;$('#sp').value=c.speed;$('#lg').value=c.lang}fill(CFG());
const me=()=>document.querySelector('.me').innerHTML=meHTML();
$('#save').onclick=()=>{try{localStorage.setItem('uc',JSON.stringify({name:$('#nm').value.trim()||'Ridwan Siddique',role:$('#rl').value,speed:+$('#sp').value,lang:$('#lg').value}));me();fill(CFG());toast('Settings saved')}catch(e){toast('Could not save. Browser storage is blocked.')}};
$('#rst').onclick=()=>{try{localStorage.removeItem('uc')}catch(e){}fill(CFG());me();toast('Defaults restored')};
