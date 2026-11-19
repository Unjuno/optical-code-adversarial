"""Adversarial training of core LDPC Tanner graph (same 138-byte core rate)."""
import csv,json,math,time
from pathlib import Path
from functools import lru_cache
import numpy as np
import lab_v01 as base
import robust_code as robust
import adversary
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
ORIG=base.parity_neighbors
K=base.NCELLS*2//2
assert K==1152

def make_neighbors(mult, offsets):
    offs=np.asarray(offsets,dtype=np.int64)
    if math.gcd(mult,K)!=1 or len(set(offsets))!=len(offsets):raise ValueError('bad parameters')
    arr=((np.arange(K,dtype=np.int64)[:,None]*mult+offs[None,:])%K).astype(np.int32)
    return lambda k,degree=4: arr if k==K else ORIG(k,degree)

def run():
    attacks=json.loads((OUT/'training_attacks.json').read_text())
    # unexposed independent TRAIN seed subset - no holdout data
    conditions=[]
    for j in range(55):
        conditions.append(('hard',450000+j,750000+j))
    for j in range(10):
        conditions.append(('medium',550000+j,850000+j))
    choices=[('legacy',37,[0,37,151,293])]
    for degree in (3,4,5,6):
        for seed in (1,7,19,41,73):
            rg=np.random.default_rng(31001+seed+degree*1000)
            vals=rg.choice(K,degree,replace=False)
            mult=[5,7,11,19,29][seed%5]
            choices.append((f'd{degree}_s{seed}',mult,list(map(int,vals))))
    rows=[];t0=time.time()
    for tag,mult,offs in choices:
        base.parity_neighbors=make_neighbors(mult,offs)
        correct={'hard':0,'medium':0,'adv':0}
        causes={}
        for group,pseed,cseed in conditions:
            p=np.random.default_rng(pseed).bytes(142)
            v=robust.encode(p,robust.LEGACY_LAYOUT)
            image=base.camera(v,base.CHANNELS[group],np.random.default_rng(cseed))
            d=robust.decode(image,4,robust.LEGACY_LAYOUT)
            correct[group]+=int(d.payload==p)
            causes[d.reason]=causes.get(d.reason,0)+1
        for item in attacks:
            raw=np.random.default_rng(item['data_seed']).bytes(142)
            frame=robust.encode(raw)
            image=base.camera(frame,adversary.channel(np.array(item['z'])),np.random.default_rng(item['seed']))
            d=robust.decode(image,4)
            correct['adv']+=int(d.payload==raw)
        score=correct['hard']+correct['adv']*.6 + correct['medium']*.3
        row={'tag':tag,'multiplier':mult,'offsets':json.dumps(offs),**{k+'_ok':v for k,v in correct.items()},'score':score,'reasons':json.dumps(causes)}
        rows.append(row)
        print('ECC',tag,'hard',correct['hard'],'/55','adv',correct['adv'],'/40','medium',correct['medium'],'/10','score',round(score,1),'elapsed',round(time.time()-t0,1),flush=True)
    base.parity_neighbors=ORIG
    with (OUT/'core_graph_training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    chosen=max(rows,key=lambda r:r['score'])
    (OUT/'selected_core_graph.json').write_text(json.dumps(chosen,indent=2))
    print('PICKED',chosen,flush=True)
if __name__=='__main__':run()
