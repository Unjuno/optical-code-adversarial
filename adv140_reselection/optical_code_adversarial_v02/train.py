"""Select encoder geometry against generated near-worst-case channel examples.
Training seeds distinct from all holdout seeds. 142-byte identical payload in each arm.
"""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np
import lab_v01 as old
import robust_code as code
import adversary
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)


def train(n_attack:int=40):
    rng=np.random.default_rng(2026111501)
    attacks=[]
    t=time.time()
    for j in range(n_attack):
        raw=np.random.default_rng(700001+j).bytes(142)
        item=adversary.search(raw,code.LEGACY_LAYOUT,4,rng,candidates=8,steps=1)
        item.update({'data_seed':700001+j,'id':j})
        attacks.append(item)
        if (j+1)%10==0:print('crafted',j+1,'/',n_attack,'elapsed',round(time.time()-t,1),flush=True)
    (OUT/'training_attacks.json').write_text(json.dumps(attacks,ensure_ascii=False,indent=2))
    layouts=adversary.candidate_layouts()
    rows=[]
    for lay in layouts:
        correct=0;reasons={};soft=[]
        for item in attacks:
            raw=np.random.default_rng(item['data_seed']).bytes(142)
            img=code.encode(raw,lay)
            channel=adversary.channel(np.asarray(item['z']))
            out=old.camera(img,channel,np.random.default_rng(item['seed']))
            res=code.decode(out,4,lay)
            correct+=int(res.payload==raw)
            reasons[res.reason]=reasons.get(res.reason,0)+1
            loss,_=adversary.attack_loss(out,raw,lay,4)
            soft.append(float(loss))
        rows.append(dict(layout=lay.name,correct=correct,n=n_attack,loss_mean=np.mean(soft),loss_p90=np.quantile(soft,.9),reason=json.dumps(reasons)))
        print('LAYOUT',lay.name,correct,'/',n_attack,'meanloss',round(np.mean(soft),3),flush=True)
    with (OUT/'layout_training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    # development criterion: maximize successes; tie-break lower mean soft loss
    winner=sorted(rows,key=lambda x:(-x['correct'],x['loss_mean']))[0]
    lay=next(x for x in layouts if x.name==winner['layout'])
    np.savez(OUT/'optimized_layout.npz',core=lay.core,extra=lay.extra,name=np.array(lay.name))
    print('SELECTED',lay.name,winner,flush=True)
    return winner

if __name__=='__main__':train(40)
