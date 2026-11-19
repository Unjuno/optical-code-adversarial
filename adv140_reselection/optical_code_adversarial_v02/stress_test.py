"""Unseen WORST-OF-8 held-out perturbations per payload, identical for all designs.
The adversary may choose any of eight camera parameter vectors (budget<=5.4).
One-shot read: success only if EVERY challenge view is correctly decoded.
"""
from pathlib import Path
import csv,os,json,time
import numpy as np
import robust_code as code
import lab_v01 as old
import adversary
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
a=np.load(OUT/'optimized_layout.npz')
OPT=code.Layout(str(a['name']),a['core'],a['extra'])
MODES={
 'base138':(138,'base',None,'legacy',140),
 'frame140_v01':(140,'frame',None,'legacy',160),
 'advgraph140':(140,'spare',code.LEGACY_LAYOUT,'adversarial',140),
 'combined142':(142,'spare',OPT,'adversarial',140),
}

def run(mode,n=120,views=8):
    cap,kind,lay,graph,threshold=MODES[mode]
    old.DETECTION_THRESHOLD=threshold
    total=0;survived=0;view_success=0
    t0=time.time();reasons={};worstout=[]
    for j in range(n):
        raw=np.random.default_rng(371100001+j).bytes(142)[:cap]
        if kind=='base':img=old.encode(raw,'rgb4')
        elif kind=='frame':img=old.encode_joint(raw,.44,extra_bytes=2)
        else:img=code.encode(raw,lay,graph=graph)
        view_good=[]
        for v in range(views):
            rng=np.random.default_rng(77700200+j*100+v)
            z=adversary.sample_params(rng)
            condition=adversary.channel(z)
            cam_seed=800011000+j*100+v
            picture=old.camera(img,condition,np.random.default_rng(cam_seed))
            if kind=='base':res=code.base_decode(picture)
            elif kind=='frame':res=old.decode_joint(picture,.44,extra_bytes=2)
            else:res=code.decode(picture,extra_bytes=cap-138,layout=lay,graph=graph)
            good=int(res.payload==raw)
            view_success+=good
            view_good.append(good)
            if not good:reasons[res.reason]=reasons.get(res.reason,0)+1
        total+=views; survived+=int(all(view_good))
        worstout.append(int(all(view_good)))
        if (j+1)%60==0:print(mode,j+1,'/',n,'worst-of',survived,'elapsed',round(time.time()-t0,1),flush=True)
    row=dict(mode=mode,capacity=cap,n_messages=n,n_views_each=views,view_success=view_success,
             view_total=total,view_p=view_success/total,worst_success=survived,
             worst_p=survived/n,reasons=json.dumps(reasons),seed=371100001)
    with (OUT/f'stress_{mode}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
    np.save(OUT/f'stress_outcomes_{mode}.npy',np.asarray(worstout,dtype=np.uint8))
    print('STRESS_FINAL',mode,row,flush=True)
    return row
if __name__=='__main__':run(os.environ.get('MODE','advgraph140'),n=int(os.environ.get('N',120)),views=int(os.environ.get('VIEWS',8)))
