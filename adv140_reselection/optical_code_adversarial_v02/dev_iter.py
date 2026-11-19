import numpy as np,time
import lab_v01 as l
import robust_code as r
n=170
samples=[]
for i in range(n):
 p=np.random.default_rng(200001+i).bytes(140)
 y=l.camera(r.encode(p,graph='adversarial'),l.CHANNELS['hard'],np.random.default_rng(300001+i))
 samples.append((p,y))
for ite in (12,18,24,30,42,60):
 r.CORE_MAX_ITER=ite
 good=0; reasons={};times=[]
 for p,y in samples:
  d=r.decode(y,2,graph='adversarial')
  good+=d.payload==p
  reasons[d.reason]=reasons.get(d.reason,0)+1;times.append(d.elapsed_ms)
 print(ite,good,n,'p50',round(np.median(times),2),'p95',round(np.quantile(times,.95),2),'causes',reasons,flush=True)
