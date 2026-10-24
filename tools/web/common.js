const root=document.documentElement;
const saved=localStorage.getItem('vera-demo-theme');
root.dataset.theme=saved==='dark'?'dark':'light';
const toggle=document.getElementById('themeToggle');
function updateTheme(){if(toggle)toggle.textContent=root.dataset.theme==='dark'?'☀ Light':'☾ Dark'}
updateTheme();
if(toggle)toggle.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';localStorage.setItem('vera-demo-theme',root.dataset.theme);updateTheme()});
