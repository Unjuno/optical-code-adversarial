"""Development-only cache/screen. NO final/holdout seed data accessed."""
from __future__ import annotations
import argparse,pickle,csv,time,hashlib,json,os
from collections import Counter
from pathlib import Path
import numpy as np
import joint_receiver as j

DIR=Path(__file__).resolve().parent
DEVS={'screen':(551100000,662100000,400),'confirm':(771100000,882100000,300)}

def cache(stage):
    pbase,cbase,n=DEVS[stage]; rows=j.load_rows()
    events=[]
    t0=time.perf_counter()
    for i in range(n):
        p=np.random.default_rng(pbase+i).bytes(142)[:140]
        sym=j.a.encode140(p,rows)
        image=j.base.camera(sym,j.base.CHANNELS['hard'],np.random.default_rng(cbase+i))
        ctx=j.get_context(image,rows)
        ans={'i':i,'payload_seed':pbase+i,'camera_seed':cbase+i,'payload_sha256':hashlib.sha256(p).hexdigest(),
             'reason':ctx.get('reason','detection:'+ctx.get('detection','')),'ctx':ctx,'payload':p}
        events.append(ans)
        if i%100==99: print(stage,i+1,'/',n,'elapsed',round(time.perf_counter()-t0,1),flush=True)
    path=DIR/'results'/f'development_{stage}_contexts.pkl'
    with path.open('wb') as f:pickle.dump(events,f,protocol=pickle.HIGHEST_PROTOCOL)
    print('cache',stage,'reason_count',dict(Counter(e['reason'] for e in events)),'path',path)

def load(stage):
    with (DIR/'results'/f'development_{stage}_contexts.pkl').open('rb') as f:return pickle.load(f)

def candidates():
    arr=[]
    for meth in ['affine','mean']:
      for ridge in [1,4,12,30]:
       for cov in ['identity','diag']:
        for beta in [0.5,1.0]:
          arr.append({'name':f'{meth}_ridge{ridge}_cov{cov}_beta{beta}',
            'method':meth,'ridge':float(ridge),'cov':cov,'prior_strength':beta,'sigma':0.19,'variance_shrink':.65})
    return arr

def screen(stage):
    events=load(stage);rows=j.load_rows()
    if stage=='screen':cand=candidates()
    else:
        listed=json.loads((DIR/'results'/'screen_top.json').read_text())
        cand=listed['selected_candidates']
    results=[]
    baseline=Counter(e['reason'] for e in events)
    for ci,config in enumerate(cand):
        rescued=[]; false=[]
        for ev in events:
            ctx=ev['ctx']
            if ctx.get('aux') is None or ctx.get('core') is not None or ctx.get('base_output') is not None or 'detection' in ctx: continue
            new=j.postprocess_context(ctx,rows,{'calibrations':[config]})
            if new['reason']=='ok':
                if new['payload']==ev['payload']: rescued.append(ev['i'])
                else: false.append(ev['i'])
        record={'name':config['name'],'rescued':len(rescued),'wrong_decoded':len(false),'indices':rescued,'config':config}
        results.append(record)
        if (ci+1)%8==0: print('screened',ci+1,'/',len(cand),'best',max(r['rescued'] for r in results),flush=True)
    results.sort(key=lambda x:(x['rescued']-100*x['wrong_decoded'],x['rescued']),reverse=True)
    with (DIR/'results'/f'{stage}_screen_results.json').open('w') as f:json.dump({'baseline':dict(baseline),'candidates':results},f,indent=2,ensure_ascii=False)
    if stage=='screen':
        top=[];seen=set()
        for r in results:
            if r['name'] in seen:continue
            top.append(r['config']);seen.add(r['name'])
            if len(top)>=6: break
        with (DIR/'results'/'screen_top.json').open('w') as f:json.dump({'selected_candidates':top},f,indent=2)
    print(stage,'baseline',dict(baseline),'TOP',[(r['name'],r['rescued'],r['wrong_decoded']) for r in results[:12]],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['cache','screen']);p.add_argument('--stage',choices=['screen','confirm'],required=True)
    args=p.parse_args()
    if args.action=='cache':cache(args.stage)
    else:screen(args.stage)
