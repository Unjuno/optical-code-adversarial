"""Pre-frozen development-only search for 42-bit auxiliary parity matrix.
Usage: python aux42_search.py --stage analytic|screen|select
No holdout seed, result, or outcome array is imported or referenced here.
"""
from __future__ import annotations
import os,json,csv,time,argparse,hashlib
from pathlib import Path
import numpy as np
import aux42_lib as a
BASE=a.ROOT/'outputs'; BASE.mkdir(exist_ok=True)
CHALLENGE=a.base.Channel('challenge',2.5,.095,.075,3.2,12.0,.055,.13)


def gen_analytic():
    a.freeze_settings()
    objs=[]
    prev=a.old_rows()
    candidates=[('old',prev)]
    for degree in (3,4,5,6,7):
        pool=[]
        for j in range(64):
            seed=44000000+degree*1000+j
            rows=a.generate_rows(degree,seed)
            s=a.spectrum(rows);g=a.rows_score(rows)
            stat={**s,**g}
            # First maximize CRC-valid min distance and reduce low-weight neighbors,
            # then favor low four-cycle count / low max column degree.
            score=(s['min_distance'],-s['weight_le_20'],-s['weight_le_24'],-g['four_cycles'],-g['column_degree_max'])
            pool.append((score,j,rows,stat))
        pool.sort(key=lambda x:x[0],reverse=True)
        # Three distinct candidates for each degree
        for ix,(_,j,rows,s) in enumerate(pool[:3]):
            candidates.append((f'd{degree}_a{ix}_s{j}',rows))
            print('ANALYTIC',f'd{degree}_a{ix}_s{j}',s,flush=True)
    obj={}
    for name,rows in candidates:
        obj[name]=rows.tolist()
        m={'name':name,'degree':rows.shape[1],**a.rows_score(rows),**a.spectrum(rows)}
        objs.append(m)
    (BASE/'analytic_candidate_rows.json').write_text(json.dumps(obj,indent=2))
    with (BASE/'analytic_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=objs[0].keys());w.writeheader();w.writerows(objs)
    print('ANALYTIC candidates=',len(candidates),'old_spectrum',a.spectrum(prev),flush=True)


def eval_candidate(name,rows,begin_hard,n_hard,begin_challenge,n_challenge):
    a.freeze_settings()
    counts={'hard_aux_ok':0,'hard_detect':0,'hard_n':n_hard,
            'challenge_aux_ok':0,'challenge_detect':0,'challenge_n':n_challenge,
            'hard_aux_crc':0,'challenge_aux_crc':0}
    for c,first,n,pseed,cseed in [('hard',begin_hard,n_hard,341000000,441000000),
                                 ('challenge',begin_challenge,n_challenge,341000000,541000000)]:
        for j in range(first,first+n):
            data=np.random.default_rng(pseed+j).bytes(140)
            symbol=a.encode140(data,rows)
            ch=a.base.CHANNELS['hard'] if c=='hard' else CHALLENGE
            captured=a.base.camera(symbol,ch,np.random.default_rng(cseed+j))
            match,reason=a.decode_aux_only(captured,rows,data[138:])
            if match is not None: counts[c+'_detect']+=1
            if match:counts[c+'_aux_ok']+=1
            if reason=='aux_crc':counts[c+'_aux_crc']+=1
    # Rate on all attempted frames, detector failures count against candidate.
    counts['score']=counts['hard_aux_ok']/max(1,n_hard)+.4*counts['challenge_aux_ok']/max(1,n_challenge)
    counts['name']=name
    return counts


def screen(start=0,stop=16):
    rows=json.loads((BASE/'analytic_candidate_rows.json').read_text())
    results=[];t0=time.time()
    for i,(name,arr) in enumerate(rows.items()):
        if i<start or i>=stop:continue
        rec=eval_candidate(name,np.asarray(arr,dtype=np.int32),0,200,0,90)
        results.append(rec)
        print('SCREEN',i+1,len(rows),name,'hard',rec['hard_aux_ok'],'/200','challenge',rec['challenge_aux_ok'],'/90','score',round(rec['score'],6),'time',round(time.time()-t0,1),flush=True)
    path=BASE/'screen_dev.csv'
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys())
        if path.stat().st_size==0:w.writeheader()
        w.writerows(results)
    print('SCREEN CHUNK DONE',start,stop,flush=True)


def select(start=0,stop=6):
    rows={k:np.asarray(v,dtype=np.int32) for k,v in json.loads((BASE/'analytic_candidate_rows.json').read_text()).items()}
    with (BASE/'screen_dev.csv').open() as f:sc=list(csv.DictReader(f))
    # exactly 5 novel candidates, plus old, cannot use holdout for any selection.
    new=sorted([x for x in sc if x['name']!='old'],key=lambda x:(-float(x['score']),-int(x['hard_aux_ok']),x['name']))[:5]
    names=['old']+[x['name'] for x in new]
    res=[];t0=time.time()
    for i,name in enumerate(names):
        if i<start or i>=stop:continue
        rec=eval_candidate(name,rows[name],200,700,90,300)
        res.append(rec)
        print('SELECT',i+1,len(names),name,'hard',rec['hard_aux_ok'],'/700','challenge',rec['challenge_aux_ok'],'/300','score',round(rec['score'],6),'time',round(time.time()-t0,1),flush=True)
    path=BASE/'selection_dev.csv'
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=res[0].keys())
        if path.stat().st_size==0:w.writeheader()
        w.writerows(res)
    print('SELECT CHUNK DONE',start,stop,flush=True)

def freeze():
    rows={k:np.asarray(v,dtype=np.int32) for k,v in json.loads((BASE/'analytic_candidate_rows.json').read_text()).items()}
    with (BASE/'selection_dev.csv').open() as f:res=[{k:float(v) if k in ('score',) else int(v) if k.endswith(('_ok','_crc','_detect','_n')) else v for k,v in rec.items()} for rec in csv.DictReader(f)]
    assert len(res)==6 and len({x['name'] for x in res})==6,(len(res),res)
    ranked=sorted([x for x in res if x['name']!='old'],key=lambda x:(-x['score'],-x['hard_aux_ok'],x['name']))
    candidate=ranked[0]; old=next(x for x in res if x['name']=='old')
    name=candidate['name'] if candidate['score']>old['score'] else 'old'
    chosen=rows[name]
    a.validate_rows(chosen)
    np.save(BASE/'selected_aux42_rows.npy',chosen)
    data={'name':name,'candidate':candidate,'old_development':old,
          'matrix_shape':list(chosen.shape),'sha256_of_npy':hashlib.sha256((BASE/'selected_aux42_rows.npy').read_bytes()).hexdigest(),
          'validation_seeds_not_used':True,'selection_condition':'dev_aux_success; fixed core and decoder',
          'aux24_iterations':24,'core18_iterations':18,'detection_threshold':140,
          'selected_graph':a.rows_score(chosen),'selected_crc16_valid_distance':a.spectrum(chosen)}
    (BASE/'frozen_selection.json').write_text(json.dumps(data,indent=2))
    print('FROZEN SELECTION',data,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('analytic','screen','select','freeze'),required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=16)
    args=ap.parse_args()
    if args.stage=='analytic':gen_analytic()
    elif args.stage=='screen':screen(args.start,args.stop)
    elif args.stage=='select':select(args.start,args.stop)
    else:freeze()
