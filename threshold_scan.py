import numpy as np,time
import robust_code as r
import lab_v01 as l
for cond in ['medium','hard']:
 n=120
 cases=[]
 for i in range(n):
  x=np.random.default_rng(918200+i).bytes(142)
  sd=191100+i+(100000 if cond=='hard' else 0)
  cases.append((x,l.camera(r.encode(x),l.CHANNELS[cond],np.random.default_rng(sd))))
 for morph in [1,3,5]:
  for thresh in [100,120,140,160,180,200,220]:
   l.DETECTION_THRESHOLD=thresh;l.MORPH_KERNEL=morph
   ok=0;det=0;aux=0
   for x,y in cases:
    d=r.decode(y,extra_bytes=4)
    ok+=d.payload==x
    det+=d.reason.startswith('detection')
    aux+=d.reason=='aux_crc'
   print(cond,morph,thresh,ok,det,aux,flush=True)
