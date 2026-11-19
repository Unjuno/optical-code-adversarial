"""Frozen 140B symbol + single receiver validation on new never-selected seeds."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import norm

HERE=Path(__file__).resolve().parent
EXP=HERE/'adv140_reselection'
sys.path.insert(0,str(EXP))
import aux42_lib as a

SEED_PAYLOAD=331100000
SEED_CAMERA=442100000
N=1000
OLD_SUCC=939
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)


def wilson(k,n,z=1.959963984540054):
    p=k/n
    q=1+z*z/n
    c=(p+z*z/(2*n))/q
    r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return [float(c-r),float(c+r)]


def newcombe(k1,n1,k0,n0,z=1.959963984540054):
    # Difference p_new-p_previous, Newcombe-Wilson score interval, method 10
    p1=k1/n1;p0=k0/n0
    l1,u1=wilson(k1,n1,z)
    l0,u0=wilson(k0,n0,z)
    d=p1-p0
    return [float(d-np.hypot(p1-l1,u0-p0)),float(d+np.hypot(u1-p1,p0-l0))]


def locked():
    conf=json.loads((EXP/'outputs/frozen_selection.json').read_text())
    assert conf['name']=='d5_a2_s21'
    arrpath=EXP/'outputs/selected_aux42_rows.npy'
    checksum=hashlib.sha256(arrpath.read_bytes()).hexdigest()
    assert checksum==conf['sha256_of_npy']=="4d3f700cda39b60b4e1d2aee775f78ddd842a88a33718b98d55c7e5370a5e9bd"
    a.freeze_settings()
    rows=a.validate_rows(np.load(arrpath))
    assert rows.shape==(42,5)
    assert a.base.DETECTION_THRESHOLD==140
    assert a.robust.CORE_MAX_ITER==18
    assert len(a.robust.LEGACY_EXTRA)==37
    assert 'hard' in a.base.CHANNELS
    return rows,checksum


def run_chunk(start,end):
    assert 0<=start<end<=N
    rows,digest=locked()
    csvpath=OUT/f'independent_{start:04d}_{end-1:04d}.csv'
    event_fields=['i','payload_seed','camera_seed','success','reason','payload_sha256','decoded_sha256','elapsed_ms','core_iter','aux_iter']
    t0=time.perf_counter()
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=event_fields)
        w.writeheader()
        ctr=Counter()
        for i in range(start,end):
            pp=SEED_PAYLOAD+i;cc=SEED_CAMERA+i
            payload=np.random.default_rng(pp).bytes(142)[:140]
            symbol=a.encode140(payload,rows)
            observation=a.base.camera(symbol,a.base.CHANNELS['hard'],np.random.default_rng(cc))
            result=a.decode140(observation,rows)
            correct=(result.payload==payload)
            reason=str(result.reason)
            if not correct and reason=='ok':
                # distinguish a silent CRC collision from an ordinary success
                reason='silent_mismatch'
            if correct:
                assert reason=='ok'
            ctr[reason]+=1
            w.writerow({'i':i,'payload_seed':pp,'camera_seed':cc,'success':int(correct),'reason':reason,
                'payload_sha256':hashlib.sha256(payload).hexdigest(),
                'decoded_sha256':hashlib.sha256(result.payload).hexdigest() if result.payload is not None else '',
                'elapsed_ms':f'{float(result.elapsed_ms):.6f}',
                'core_iter':result.core_iter,'aux_iter':result.extra_iter})
        f.flush();os.fsync(f.fileno())
    print(json.dumps({'chunk':[start,end],'events':end-start,'reasons':dict(ctr),'time_s':round(time.perf_counter()-t0,3),'path':str(csvpath),'sha256_matrix':digest},ensure_ascii=False),flush=True)


def finalize():
    selected,_=locked()
    files=sorted(OUT.glob('independent_????_????.csv'))
    events=[]
    for path in files:
        with path.open(newline='') as f: events+=list(csv.DictReader(f))
    indices=[int(x['i']) for x in events]
    n=len(indices)
    if indices!=list(range(N)):
        raise ValueError(f'INCOMPLETE or DUPLICATE: processed {n} entries; unique {len(set(indices))}; first/last {indices[:5]}/{indices[-5:]}')
    for x in events:
        i=int(x['i'])
        assert int(x['payload_seed'])==SEED_PAYLOAD+i
        assert int(x['camera_seed'])==SEED_CAMERA+i
    reasons=Counter(x['reason'] for x in events)
    successes=sum(int(x['success']) for x in events)
    assert successes==reasons.get('ok',0)
    assert sum(reasons.values())==N
    detections=sum(v for k,v in reasons.items() if k.startswith('detection:'))
    failures={'core_parity':reasons.get('core_parity',0),'core_crc':reasons.get('core_crc',0),
              'aux_crc':reasons.get('aux_crc',0),'detection':detections}
    other={k:v for k,v in reasons.items() if k not in {'ok','core_parity','core_crc','aux_crc'} and not k.startswith('detection:')}
    if other: failures['other']=other
    assert sum(failures[k] for k in ['core_parity','core_crc','aux_crc','detection'])+sum(other.values())==N-successes
    old_rate=OLD_SUCC/N
    interval_new=wilson(successes,N)
    interval_aux=wilson(failures['aux_crc'],N)
    difference_interval=newcombe(successes,N,OLD_SUCC,N)
    one_sided_lower=newcombe(successes,N,OLD_SUCC,N,z=norm.ppf(.95))[0]
    if difference_interval[1]<0: verdict='低下'
    elif one_sided_lower>-.03: verdict='再現'
    else: verdict='証拠なし'
    lat=np.array([float(x['elapsed_ms']) for x in events])
    summary={
        'finished':True,'n':N,'scheme':'d5_a2_s21','condition':'hard','payload_bytes':140,
        'payload_seed_expression':'331100000+i','camera_seed_expression':'442100000+i','i_range':[0,999],
        'success':successes,'success_rate':successes/N,'success_wilson95':interval_new,
        'aux_crc_failure':failures['aux_crc'],'aux_crc_failure_wilson95':interval_aux,
        'failure_reasons':failures,'reason_detail':dict(reasons),
        'old_reference_succ':OLD_SUCC,'old_reference_n':N,'rate_difference_pp':100*(successes-OLD_SUCC)/N,
        'rate_difference_newcombe95_pp':[100*x for x in difference_interval],
        'rate_difference_one_sided95_lower_pp':100*one_sided_lower,
        'classification':verdict,
        'classification_rule':'drop if two-sided 95% difference upper<0; else reproduce if one-sided95 lower>-3 pp; else insufficient evidence',
        'median_latency_ms':float(np.median(lat)),'p95_latency_ms':float(np.quantile(lat,.95)),
        'matrix_sha256':hashlib.sha256((EXP/'outputs/selected_aux42_rows.npy').read_bytes()).hexdigest(),
        'code_source_sha256':{k:hashlib.sha256((EXP/k).read_bytes()).hexdigest() for k in ['aux42_lib.py','optical_code_adversarial_v02/robust_code.py','optical_code_adversarial_v02/lab_v01.py']},
        'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'cpu':platform.processor(),'threads':os.environ.get('OPENBLAS_NUM_THREADS'),'batch':1}
    }
    path=OUT/'independent_summary.json'
    path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--chunk',action='store_true')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int,default=500)
    p.add_argument('--finalize',action='store_true')
    args=p.parse_args()
    if args.chunk:run_chunk(args.start,args.stop)
    elif args.finalize:finalize()
    else:p.error('choose --chunk or --finalize')
