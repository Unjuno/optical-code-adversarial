"""Final frozen paired holdout, never imported/called by auxiliary training script."""
from __future__ import annotations
import argparse,hashlib,csv,json,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest
import aux42_lib as a
ROOT=a.ROOT; OUT=ROOT/'outputs'

def assert_frozen():
    f=json.loads((OUT/'frozen_selection.json').read_text())
    raw=(OUT/'selected_aux42_rows.npy').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==f['sha256_of_npy']
    selected=a.validate_rows(np.load(OUT/'selected_aux42_rows.npy'))
    assert a.base.DETECTION_THRESHOLD==140
    assert a.robust.CORE_MAX_ITER==18
    assert a.robust.encode_aux(bytes([1,2])).shape==(74,)
    assert np.array_equal(a.old_rows(),a.robust.extra_parity_indices(32))
    assert selected.shape[0]==42
    assert len(a.robust.LEGACY_EXTRA)==37
    return selected

def run_chunk(first,last):
    a.freeze_settings()
    selected=assert_frozen()
    rows=[];t0=time.time()
    for i in range(first,last):
        p=np.random.default_rng(121100000+i).bytes(142)[:140]
        rnd=222100000+i
        old_symbol=a.robust.encode(p,layout=a.robust.LEGACY_LAYOUT,graph='adversarial')
        new_symbol=a.encode140(p,selected)
        old_y=a.base.camera(old_symbol,a.base.CHANNELS['hard'],np.random.default_rng(rnd))
        new_y=a.base.camera(new_symbol,a.base.CHANNELS['hard'],np.random.default_rng(rnd))
        x=a.robust.decode(old_y,extra_bytes=2,layout=a.robust.LEGACY_LAYOUT,graph='adversarial')
        y=a.decode140(new_y,selected)
        rows.append({
            'i':i,'payload_seed':121100000+i,'channel_seed':rnd,
            'old_success':int(x.payload==p),'new_success':int(y.payload==p),
            'old_reason':x.reason,'new_reason':y.reason,
            'old_ms':float(x.elapsed_ms),'new_ms':float(y.elapsed_ms),
            'old_coreit':x.core_iter,'new_coreit':y.core_iter,
            'old_auxit':x.extra_iter,'new_auxit':y.extra_iter})
    path=OUT/f'holdout_pair_{first:04d}_{last-1:04d}.csv'
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('HOLDOUT CHUNK',first,last,'old',sum(x['old_success'] for x in rows),
          'new',sum(x['new_success'] for x in rows),'old_aux_crc',sum(x['old_reason']=='aux_crc' for x in rows),
          'new_aux_crc',sum(x['new_reason']=='aux_crc' for x in rows),
          'seconds',round(time.time()-t0,2),'path',path,flush=True)

def wilson(k,n,z=1.959963984540054):
    ph=k/n
    denom=1+z*z/n
    mid=(ph+z*z/(2*n))/denom
    rad=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/denom
    return [float(mid-rad),float(mid+rad)]

def finish():
    a.freeze_settings();selected=assert_frozen()
    files=sorted(OUT.glob('holdout_pair_*.csv'))
    events=[]
    for f in files:
        with f.open() as fh:events.extend(csv.DictReader(fh))
    idx=[int(r['i']) for r in events]
    assert idx==list(range(1000)),(len(idx),len(set(idx)),idx[:5],idx[-5:])
    n=len(events)
    old=np.asarray([int(r['old_success']) for r in events],dtype=np.uint8)
    new=np.asarray([int(r['new_success']) for r in events],dtype=np.uint8)
    reference=np.load(ROOT/'archive_outcomes_hard_advgraph140.npy')
    assert old.shape==reference.shape and np.array_equal(old,reference),'baseline did not reproduce per-item historic results'
    primary={'old_only':int(np.sum((old==1)&(new==0))),
             'new_only':int(np.sum((old==0)&(new==1))),
             'both':int(np.sum((old==1)&(new==1))),
             'neither':int(np.sum((old==0)&(new==0)))}
    obs={'old':Counter(x['old_reason'] for x in events),'new':Counter(x['new_reason'] for x in events)}
    total_succ={k:int(v.sum()) for k,v in [('old',old),('new',new)]}
    accepted=lambda s: (s=='ok' or s=='aux_crc')
    shared=[x for x in events if accepted(x['old_reason']) and accepted(x['new_reason'])]
    aux_old=[x['old_reason']=='aux_crc' for x in shared]
    aux_new=[x['new_reason']=='aux_crc' for x in shared]
    aux_old_only=sum(v and not w for v,w in zip(aux_old,aux_new))
    aux_new_only=sum(w and not v for v,w in zip(aux_old,aux_new))
    k=aux_old_only+aux_new_only
    pval=float(binomtest(aux_old_only,k,.5,alternative='greater').pvalue) if k else 1.0
    aux_old_total=obs['old'].get('aux_crc',0);aux_new_total=obs['new'].get('aux_crc',0)
    if total_succ['new']<=total_succ['old'] or aux_new_total>=aux_old_total:
        verdict='FAIL'
    elif pval<.05:
        verdict='PASS'
    else:verdict='UNCERTAIN'
    summary={
        'frozen_selection':json.loads((OUT/'frozen_selection.json').read_text()),
        'condition':'hard','n':n,'payload_seed_start':121100000,'channel_seed_start':222100000,
        'old_success':total_succ['old'],'new_success':total_succ['new'],
        'success_wilson95':{k:wilson(total_succ[k],n) for k in ('old','new')},
        'old_aux_crc_total':aux_old_total,'new_aux_crc_total':aux_new_total,
        'aux_crc_wilson95_unconditional':{k:wilson(obs[k].get('aux_crc',0),n) for k in ('old','new')},
        'fail_reasons':{k:dict(v) for k,v in obs.items()},
        'paired_total_success':primary,
        'shared_core_accepted_count':len(shared),
        'aux_crc_on_shared_core':{'old':sum(aux_old),'new':sum(aux_new),
                                  'old_only_fail':aux_old_only,'new_only_fail':aux_new_only,
                                  'both_fail':sum(v and w for v,w in zip(aux_old,aux_new))},
        'aux_paired_one_sided_exact_p':pval,'prereg_verdict':verdict,
        'reproduced_old_outcomes_byte_for_byte':True,
        'latency_ms':{k:{'median':float(np.median([float(x[k+'_ms']) for x in events])),
                          'p95':float(np.quantile([float(x[k+'_ms']) for x in events],.95))} for k in ('old','new')},
        'component_settings':{'core_iterations':18,'aux_iterations':24,'aux_alpha':.8,'threshold':140,'palette':'rgb4',
                              'cells_aux':37,'n_parity':42,'matrix_shape':list(selected.shape)},
    }
    (OUT/'frozen_holdout_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('FROZEN HOLDOUT SUMMARY\n'+json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['chunk','finish'],required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=500)
    args=p.parse_args()
    if args.stage=='chunk':run_chunk(args.start,args.stop)
    else:finish()
