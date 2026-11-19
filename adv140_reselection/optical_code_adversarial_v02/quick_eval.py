import sys, numpy as np
import lab_v01 as l
import robust_code as r
l.DETECTION_THRESHOLD=160
for cond in ['medium','hard']:
 for extra in [2,3,4]:
  counts={};reasons={};n=110
  for i in range(n):
   x=np.random.default_rng(19239+i).bytes(138+extra)
   img=r.encode(x)
   pix=l.camera(img,l.CHANNELS[cond],np.random.default_rng(871310+i+100000*(cond=='hard')))
   out=r.decode(pix,extra_bytes=extra)
   counts['ok']=counts.get('ok',0)+int(out.payload==x)
   reasons[out.reason]=reasons.get(out.reason,0)+1
  print(cond,'+',extra,n,counts,reasons,flush=True)
