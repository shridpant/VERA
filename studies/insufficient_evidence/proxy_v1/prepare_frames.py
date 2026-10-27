"""CPU-only extraction, sequential decoding and source-hash verification."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2
from PIL import Image,ImageStat

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--videos',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);manifest=json.loads(a.manifest.read_text());cases=[];start=time.time()
 for v in manifest['videos']:
  path=a.videos/(v['video_id']+'.mp4');assert sha(path)==v['expected_video_sha256'],path
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>0
  duration=count/fps;assert abs(duration-v['annotation_duration_seconds'])<=max(2,.02*duration),(path,duration,v['annotation_duration_seconds'])
  indices={kind:[int(t*fps) for t in info['sample_seconds']] for kind,info in v['views'].items()};wanted=set(sum(indices.values(),[]));assert len(wanted)==16 and min(wanted)>=0 and max(wanted)<count
  folder=a.out/v['video_id'];folder.mkdir();saved={}
  for i in range(max(wanted)+1):
   ok,frame=cap.read();assert ok,(path,i)
   if i in wanted:
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));dest=folder/f'{i:06d}.jpg';image.save(dest,quality=95)
    check=Image.open(dest).convert('RGB');saved[i]={'sha256':sha(dest),'mean_intensity':sum(ImageStat.Stat(check).mean)/3}
  cap.release()
  for kind,inds in indices.items():
   cases.append({'case_id':v['video_id']+'|'+kind,'video_id':v['video_id'],'category':v['category'],'condition':kind,'anchor':inds[4],'frames_dir':str(folder.resolve()),'frame_indices':inds,'frame_sha256':[saved[i]['sha256'] for i in inds],'mean_intensities':[saved[i]['mean_intensity'] for i in inds],'video_sha256':v['expected_video_sha256'],'fps':fps,'frame_count':count,'manifest_entry':v['manifest_entry'],'manifest_length':v['manifest_length']})
  print(v['video_id'],flush=True)
 (a.out/'inputs.json').write_text(json.dumps({'cases':cases,'manifest_sha256':sha(a.manifest),'extractor_sha256':sha(__file__),'decoder':'OpenCV sequential decode; RGB Pillow JPEG quality95','opencv':cv2.__version__,'wall_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
