const root=document.documentElement;
// Follow the system theme until the viewer picks one; storage can be unavailable in private windows.
let saved=null;
try{saved=localStorage.getItem('vera-demo-theme')}catch(error){}
root.dataset.theme=saved||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
const toggle=document.getElementById('themeToggle');
function updateTheme(){if(toggle)toggle.textContent=root.dataset.theme==='dark'?'☀ Light':'☾ Dark'}
updateTheme();
if(toggle)toggle.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('vera-demo-theme',root.dataset.theme)}catch(error){}updateTheme()});
