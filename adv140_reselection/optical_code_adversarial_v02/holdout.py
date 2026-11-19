"""Strict unseen-seed test of codes and layouts; reports full-byte exact recovery.
Preselected test set; each arm receives identical camera RNG seed, but payload bits differ by size.
"""
from __future__ import annotations
import csv,json,sys,time,os
from pathlib import Path
import numpy as np
import lab_v01 as old
import robust_code as code
import adversary
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
a=np.load(OUT/'optimized_layout.npz');OPT=code.Layout(str(a['name']),a['core'],a['extra'])

SCHEMES={
 'base138':(138,lambda b:old.encode(b,'rgb4'),lambda y:code.base_decode(y),140),
 'old_frame140':(140,lambda b:old.encode_joint(b,contrast=.44,extra_bytes=2),lambda y:old.decode_joint(y,contrast=.44,extra_bytes=2),160),
 'spare140':(140,lambda b:code.encode(b,code.LEGACY_LAYOUT),lambda y:code.decode(y,2,code.LEGACY_LAYOUT),140),
 'spare142':(142,lambda b:code.encode(b,code.LEGACY_LAYOUT),lambda y:code.decode(y,4,code.LEGACY_LAYOUT),140),
 'trained142':(142,lambda b:code.encode(b,OPT),lambda y:code.decode(y,4,OPT),140),
}

def run(n_per_condition=600,adversarial_n=150):
  rows=[]; t=time.time()
  for condition,n in [(cc,n_per_condition) for cc in os.getenv('CONDITIONS','medium,hard').split(',') if cc]:
    for name,(cap,enc,dec,thr) in SCHEMES.items():
      if name not in os.getenv('SCHEMES',','.join(SCHEMES)).split(','):continue
      old.DETECTION_THRESHOLD=thr
      ok=0; times=[];reasons={};joint=[]
      for i in range(n):
        payload=np.random.default_rng(90000001+i).bytes(142)[:cap]
        img=enc(payload)
        y=old.camera(img,old.CHANNELS[condition],np.random.default_rng(65432100+i+(100000 if condition=='hard' else 0)))
        r=dec(y);good=r.payload==payload
        ok+=good
        times.append(r.elapsed_ms)
        reasons[r.reason]=reasons.get(r.reason,0)+1
        joint.append(int(good))
      row=dict(condition=condition,mode=name,capacity=cap,success=ok,n=n,p=ok/n,mean_verified_bytes=cap*ok/n,
               median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),reasons=json.dumps(reasons),
               outcome_binary=''.join(map(str,joint)))
      rows.append(row)
      print('HOLDOUT',condition,name,ok,n,'bytes/trial',round(row['mean_verified_bytes'],2),'p50ms',round(row['median_ms'],2),'p95ms',round(row['p95_ms'],2),'elapsed',round(time.time()-t,1),flush=True)
  # black-box adversarial stress test on held-out set: tune channel against fixed reference
  # and apply the same channel parameters and rendering seeds to EVERY design.
  if os.getenv('DO_ADV','1') != '1':
    path=OUT/('holdout_'+os.getenv('CONDITIONS','medium,hard').replace(',','_')+'_'+os.getenv('SCHEMES','all').replace(',','_')+'.csv')
    with path.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('wrote',path);return rows
  rng=np.random.default_rng(2998820)
  tests=[]
  for i in range(adversarial_n):
    raw=np.random.default_rng(88000000+i).bytes(142)
    atk=adversary.search(raw,code.LEGACY_LAYOUT,4,rng,candidates=7,steps=1)
    atk['i']=i;tests.append(atk)
    if (i+1)%50==0:print('attack_test',i+1,adversarial_n,'elapsed',round(time.time()-t,1),flush=True)
  (OUT/'holdout_attacks.json').write_text(json.dumps(tests,ensure_ascii=False,indent=2))
  for name,(cap,enc,dec,thr) in SCHEMES.items():
    old.DETECTION_THRESHOLD=thr
    ok=0;times=[];reasons={};outcomes=[]
    for item in tests:
      raw=np.random.default_rng(88000000+item['i']).bytes(142)[:cap]
      symbol=enc(raw)
      image=old.camera(symbol,adversary.channel(np.asarray(item['z'])),np.random.default_rng(item['seed']))
      rr=dec(image)
      good=rr.payload==raw
      ok+=good;outcomes.append(int(good));times.append(rr.elapsed_ms)
      reasons[rr.reason]=reasons.get(rr.reason,0)+1
    rows.append(dict(condition='adversarial_transfer',mode=name,capacity=cap,success=ok,n=adversarial_n,p=ok/adversarial_n,
                     mean_verified_bytes=cap*ok/adversarial_n,median_ms=float(np.median(times)),
                     p95_ms=float(np.quantile(times,.95)),reasons=json.dumps(reasons),outcome_binary=''.join(map(str,outcomes))))
    print('ADV_TEST',name,ok,'/',adversarial_n,'bytes/trial',round(cap*ok/adversarial_n,2),flush=True)
  path=OUT/'holdout.csv'
  with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
  return rows
if __name__=='__main__':
 run(int(os.environ.get('N_PER',600)),int(os.environ.get('N_ADV',150)))
