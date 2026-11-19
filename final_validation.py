"""Frozen designs, unseen seeds. One-shot full-byte exact recovery from a single image.
No model carries user data. No leaked oracle pose. Reference code remains untouched.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,time
from pathlib import Path
import numpy as np
import lab_v01 as old
import robust_code as code

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
l=np.load(OUT/'optimized_layout.npz')
TRAINED=code.Layout(str(l['name']),l['core'],l['extra'])
SCHEMES={
 'base138': (138,'base',None,'legacy',140),
 'frame140_v01': (140,'frame',None,'legacy',160),
 'spare140': (140,'spare',code.LEGACY_LAYOUT,'legacy',140),
 'spare142': (142,'spare',code.LEGACY_LAYOUT,'legacy',140),
 'layout142': (142,'spare',TRAINED,'legacy',140),
 'advgraph140': (140,'spare',code.LEGACY_LAYOUT,'adversarial',140),
 'advgraph142': (142,'spare',code.LEGACY_LAYOUT,'adversarial',140),
 'combined142': (142,'spare',TRAINED,'adversarial',140),
}

def run(cond='hard',name='combined142',n=1000):
    cap,kind,lay,graph,threshold=SCHEMES[name]
    old.DETECTION_THRESHOLD=threshold;old.MORPH_KERNEL=3
    bits=[];reasons={};times=[];ci=[];ei=[]
    t0=time.time()
    for i in range(n):
        dat=np.random.default_rng(121100000+i).bytes(142)[:cap]
        if kind=='base':
            img=old.encode(dat,'rgb4')
            decoder=lambda yy:code.base_decode(yy)
        elif kind=='frame':
            img=old.encode_joint(dat,contrast=.44,extra_bytes=2)
            decoder=lambda yy:old.decode_joint(yy,contrast=.44,extra_bytes=2)
        else:
            img=code.encode(dat,layout=lay,graph=graph)
            decoder=lambda yy:code.decode(yy,extra_bytes=cap-138,layout=lay,graph=graph)
        seed=221100000+i+(0 if cond=='medium' else 1000000)
        rendered=old.camera(img,old.CHANNELS[cond],np.random.default_rng(seed))
        d=decoder(rendered)
        success=int(d.payload==dat)
        bits.append(success)
        reasons[d.reason]=reasons.get(d.reason,0)+1
        times.append(d.elapsed_ms)
        ci.append(getattr(d,'core_iter',getattr(d,'iterations',0)))
        ei.append(getattr(d,'extra_iter',0))
        if (i+1)%300==0:
            print(cond,name,i+1,'/',n,'ok',sum(bits),'elapsed',round(time.time()-t0,1),flush=True)
    row=dict(condition=cond,mode=name,capacity_bytes=cap,n=n,success=sum(bits),p_success=np.mean(bits),
             expected_verified_bytes=cap*np.mean(bits),median_ms=np.median(times),p95_ms=np.quantile(times,.95),
             core_iterations_mean=float(np.mean(ci)),extra_iterations_mean=float(np.mean(ei)),
             fail_reasons=json.dumps(reasons),seed_payload=121100000,seed_channel=221100000+(cond!='medium')*1000000,
             torch='none',backend='OpenCV/NumPy CPU')
    with (OUT/f'holdout_final_{cond}_{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
    np.save(OUT/f'outcomes_{cond}_{name}.npy',np.asarray(bits,dtype=np.uint8))
    print('FINAL',cond,name,sum(bits),'/',n,'exp_bytes',round(row['expected_verified_bytes'],3),
           'median',round(float(row['median_ms']),3),'p95',round(float(row['p95_ms']),3),'seconds',round(time.time()-t0,1),flush=True)
    return row
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',default='hard');parser.add_argument('--name',default='combined142');parser.add_argument('--n',type=int,default=1000)
    a=parser.parse_args();run(a.condition,a.name,a.n)
