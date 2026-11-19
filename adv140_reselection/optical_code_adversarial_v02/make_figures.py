from pathlib import Path
import csv,json,platform,sys,hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def read(cond,name):
    with (OUT/f'holdout_final_{cond}_{name}.csv').open() as f:return next(csv.DictReader(f))
ms=['base138','frame140_v01','spare140','advgraph140','combined142']
name={'base138':'Baseline 138B','frame140_v01':'Old frame 140B','spare140':'Unused cells 140B','advgraph140':'Adversarial 140B','combined142':'Adversarial 142B'}
colors=None
rows=[]
for cond in ['medium','hard']:
 for mode in ms:
  try:
    r=read(cond,mode)
    r['wilson95']=binomtest(int(r['success']),int(r['n'])).proportion_ci(confidence_level=.95,method='wilson')
    rows.append(r)
  except FileNotFoundError:pass
with (OUT/'benchmark_summary.csv').open('w',newline='') as f:
 names=['condition','mode','capacity_bytes','n','success','p_success','expected_verified_bytes','median_ms','p95_ms','wilson95_low','wilson95_high']
 w=csv.DictWriter(f,fieldnames=names);w.writeheader()
 for r in rows:
  w.writerow({'condition':r['condition'],'mode':r['mode'],'capacity_bytes':r['capacity_bytes'],
   'n':r['n'],'success':r['success'],'p_success':r['p_success'],'expected_verified_bytes':r['expected_verified_bytes'],
   'median_ms':r['median_ms'],'p95_ms':r['p95_ms'],'wilson95_low':r['wilson95'].low,
   'wilson95_high':r['wilson95'].high})
fig,ax=plt.subplots(figsize=(10,5.5))
labels=[name[x] for x in ms]
y=np.array([float(read('hard',m)['p_success'])*100 for m in ms])
bar=ax.bar(labels,y)
ax.set_title('Exact file-chunk recovery under strong synthetic distortions (n=1,000 each)')
ax.set_ylabel('Exact recovery success (%)');ax.set_ylim(0,100)
for b,yy in zip(bar,y): ax.text(b.get_x()+b.get_width()/2,yy+1.5,f'{yy:.1f}%',ha='center',fontsize=10)
ax.tick_params(axis='x',rotation=20)
ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(OUT/'hard_success.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5.5))
by=np.array([float(read('hard',m)['expected_verified_bytes']) for m in ms])
bar=ax.bar(labels,by)
ax.set_title('Verified information recovered per one-shot scan, strong distortion')
ax.set_ylabel('Verified bytes per scan');ax.set_ylim(0,145)
for b,yy in zip(bar,by):ax.text(b.get_x()+b.get_width()/2,yy+2,f'{yy:.1f}',ha='center',fontsize=10)
ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(OUT/'hard_goodput.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
sc=['base138','frame140_v01','advgraph140','combined142']
x=np.array([float(read('hard',m)['p95_ms']) for m in sc])
bar=ax.bar([name[m] for m in sc],x)
ax.set_ylim(0,10);ax.set_ylabel('P95 decode time (ms)')
ax.set_title('CPU decoding latency, single-threaded, batch size 1')
for b,yy in zip(bar,x):ax.text(b.get_x()+b.get_width()/2,yy+.12,f'{yy:.2f}',ha='center',fontsize=10)
ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(OUT/'latency_p95.png',dpi=160);plt.close(fig)

# shareable visual comparison showing actual encoded symbols, not AI-art
import cv2
import lab_v01 as old
import robust_code as rob
from pathlib import Path
p=np.random.default_rng(20261115).bytes(142)
images=[]
for tag,img in [
 ('Legacy 140B',old.encode_joint(p[:140],.44,extra_bytes=2)),
 ('Adversarial 140B',rob.encode(p[:140],graph='adversarial')),
 ('Adversarial 142B',cv2.cvtColor(cv2.imread(str(OUT/'example_capacity142.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255),
]:
 im=Image.fromarray(np.uint8(np.clip(img*255,0,255)))
 im=im.resize((405,405),resample=Image.Resampling.NEAREST)
 im=ImageOps.expand(im,border=(10,46,10,10),fill='white')
 d=ImageDraw.Draw(im);d.text((16,14),tag,fill='black')
 images.append(im)
full=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)),'white')
xx=0
for im in images:full.paste(im,(xx,0));xx+=im.width
full.save(OUT/'actual_symbols_comparison.png')
print('Generated 4 figures, benchmark_summary.csv')
