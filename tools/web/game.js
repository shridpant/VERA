const root=document.getElementById('eight');
const selected=new Set([2,3,5]);
function game(mask){const has=i=>mask.has(i);return +(0.10+(has(2)&&has(3)?0.55:0)+(has(5)?0.18:0)+(has(0)?0.04:0)).toFixed(2)}
function fact(n){let v=1;for(let i=2;i<=n;i++)v*=i;return v}
function shapley(player){let total=0;for(let mask=0;mask<256;mask++){if(mask&(1<<player))continue;const coalition=new Set();for(let j=0;j<8;j++)if(mask&(1<<j))coalition.add(j);const k=coalition.size;const before=game(coalition);coalition.add(player);total+=fact(k)*fact(7-k)/fact(8)*(game(coalition)-before)}return total}
function draw(){root.replaceChildren();for(let i=0;i<8;i++){const b=document.createElement('button');b.className='frame'+(selected.has(i)?' active':'');b.type='button';b.setAttribute('aria-pressed',selected.has(i));const label=document.createElement('b');label.textContent=String(i+1);b.append(label,document.createTextNode((i*1.25).toFixed(2)+' s'));b.onclick=()=>{selected.has(i)?selected.delete(i):selected.add(i);draw()};root.append(b)}document.getElementById('coalition').textContent='Toy score for selected frames: '+game(selected).toFixed(2);document.getElementById('shapley').textContent='Toy Shapley values: '+Array.from({length:8},(_,i)=>'F'+(i+1)+' '+shapley(i).toFixed(2)).join(' · ')}
draw();
