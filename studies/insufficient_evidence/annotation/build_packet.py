"""Build blinded, local-only frame packet; verify all source JPEG hashes first."""
import argparse,csv,hashlib,html,json,random
from pathlib import Path
from PIL import Image,ImageDraw
QUESTIONS={'collision':'Do the shown frames establish physical contact in a vehicle collision?', 'fire':'Do the shown frames establish visible fire?', 'aggression':'Do the shown frames establish physical aggression by one person against another?', 'fall':'Do the shown frames establish a person falling, rather than merely lying or bending?'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--private-map',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 inputs=json.loads(a.manifest.read_text())['inputs']; items=[]
 for x in inputs:
  for i,h in zip(x['frame_indices'],x['frame_sha256']):
   assert hashlib.sha256((a.frames/x['frames_dir']/f'{i:06d}.jpg').read_bytes()).hexdigest()==h
  for kind,order in [('ordered',list(range(8))),('repeat_slot3',[3]*8)]:
   key=hashlib.sha256((x['window_uid']+'|'+kind+'|annotation-v1').encode()).hexdigest()[:12];items.append((key,x,kind,order))
 random.Random(1701).shuffle(items);mapping=[];cards=[];rows=[]
 for key,x,kind,order in items:
  im=Image.new('RGB',(448*4,472*2),'white');draw=ImageDraw.Draw(im)
  for slot,i in enumerate(order):
   frame=Image.open(a.frames/x['frames_dir']/f"{x['frame_indices'][i]:06d}.jpg").convert('RGB').resize((448,448))
   cx=(slot%4)*448;cy=(slot//4)*472;im.paste(frame,(cx,cy+24));draw.text((cx+8,cy+5),f'Frame {slot+1}',fill='black')
  im.save(a.out/f'{key}.png')
  mapping.append({'item_id':key,'window_uid':x['window_uid'],'condition':kind,'source_sha256':x['video_sha256'],'order':order,'sheet_sha256':hashlib.sha256((a.out/f'{key}.png').read_bytes()).hexdigest()})
  cards.append(f'<section><h2>{key}</h2><a href="{key}.png"><img width="100%" src="{key}.png"></a><ol>'+''.join('<li>'+html.escape(q)+'</li>' for q in QUESTIONS.values())+'</ol></section>')
  rows.extend({'item_id':key,'question_id':qid,'label':'','evidence_slots':'','reason':'','reviewer_id':'','seconds_spent':''} for qid in QUESTIONS)
 for reviewer in ['A','B']:
  with (a.out/f'reviewer_{reviewer}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VERA annotation calibration</title><style>body{max-width:1500px;margin:30px auto;font:18px sans-serif}section{margin-bottom:70px}</style><h1>Annotation calibration: displayed frames only</h1><p>Read RUBRIC.md first. No model answers or source-category names are provided. Click a sheet for full resolution.</p>'+''.join(cards))
 a.private_map.write_text(json.dumps({'scope':'previously consulted test material; calibration only, never learning or confirmatory evaluation','items':mapping},indent=2)+'\n')
 print(f'{len(items)} views, {len(rows)} questions per reviewer; all input hashes verified')
if __name__=='__main__':main()
