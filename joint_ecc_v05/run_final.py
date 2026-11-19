"""Frozen selection validation; invokes ONLY after selection is frozen.

Seeds requested by user: payload 331100000+i, camera 442100000+i, i=0..999.
The previous individual 1,000 record CSVs are not read until finalize(),
which must only run after all 1,000 new results have been generated.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest
import joint_receiver as j

DIR=Path(__file__).resolve().parent
OUT=DIR/'results'
MATRIX=DIR/'source/selected_aux42_rows.npy'
OLD_DIR=Path('/mnt/data/independent_140_v04/results')
SEED_P=331100000
SEED_C=442100000
N=1000

def sh(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def locked():
    frozen=json.loads((DIR/'FREEZE_SELECTION.json').read_text())
    assert frozen['name']=='aux_to_main_affine_ridge4_beta0.5'
    for f,h in frozen['sha256_source'].items():
        assert sh(DIR/f)==h,('file changed since freeze',f)
    assert sh(MATRIX)==frozen['matrix_sha256']
    rows=j.load_rows()
    assert j.base.DETECTION_THRESHOLD==140
    assert j.robust.CORE_MAX_ITER==18
    assert len(j.robust.LEGACY_EXTRA)==37
    assert len(frozen['config']['calibrations'])==1
    return rows,frozen

def wilson(k,n,z=1.959963984540054):
    q=k/n
    denom=1+z*z/n
    center=(q+z*z/(2*n))/denom
    radius=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/denom
    return [float(center-radius),float(center+radius)]

def chunk(start,stop):
    assert 0<=start<stop<=N
    rows,frozen=locked()
    path=OUT/f'joint_final_{start:04d}_{stop-1:04d}.csv'
    t0=time.perf_counter()
    ctr=Counter()
    with path.open('w',newline='') as f:
        fields=['i','payload_seed','camera_seed','success','reason','base_reason','rescued','attempted',
                'payload_sha256','decoded_sha256','decode_elapsed_ms','total_elapsed_ms']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for i in range(start,stop):
            payload=np.random.default_rng(SEED_P+i).bytes(142)[:140]
            result=j.run_one(payload,SEED_C+i,rows,frozen['config'])
            correct=(result['payload']==payload)
            reason=result['reason']
            if reason=='ok' and not correct:reason='silent_mismatch'
            if correct:assert reason=='ok'
            ctr[reason]+=1
            writer.writerow({'i':i,'payload_seed':SEED_P+i,'camera_seed':SEED_C+i,
                 'success':int(correct),'reason':reason,'base_reason':result['base_reason'],
                 'rescued':int(result['rescued']),'attempted':int(result['attempted']),
                 'payload_sha256':hashlib.sha256(payload).hexdigest(),
                 'decoded_sha256':hashlib.sha256(result['payload']).hexdigest() if result['payload'] is not None else '',
                 'decode_elapsed_ms':f"{result['decode_elapsed_ms']:.6f}",
                 'total_elapsed_ms':f"{result['elapsed_ms']:.6f}"})
        f.flush();os.fsync(f.fileno())
    print(json.dumps({'chunk':[start,stop], 'events':stop-start,'counts':dict(ctr),
          'elapsed_s':round(time.perf_counter()-t0,3),'file':str(path), 'sha256':sh(path)},ensure_ascii=False),flush=True)

def finalize():
    rows,frozen=locked()
    chunks=[OUT/'joint_final_0000_0499.csv',OUT/'joint_final_0500_0999.csv']
    assert all(p.exists() for p in chunks), 'some final chunks have not run'
    events=[]
    for p in chunks:
        with p.open(newline='') as f:events+=list(csv.DictReader(f))
    assert len(events)==N and [int(v['i']) for v in events]==list(range(N))
    for e in events:
        i=int(e['i'])
        assert int(e['payload_seed'])==SEED_P+i
        assert int(e['camera_seed'])==SEED_C+i
        # SHA-256 check uses only the publicly specified seed rule, no decoder oracle.
        assert e['payload_sha256']==hashlib.sha256(np.random.default_rng(SEED_P+i).bytes(142)[:140]).hexdigest()
        assert e['success'] in ('0','1')
    reasons=Counter(e['reason'] for e in events)
    successes=sum(int(e['success']) for e in events)
    assert successes==reasons['ok']
    assert sum(reasons.values())==N
    det=sum(v for k,v in reasons.items() if k.startswith('detection:'))
    breakdown={'core_parity':reasons['core_parity'],'core_crc':reasons['core_crc'],
          'aux_crc':reasons['aux_crc'],'detection':det,'other':sum(v for k,v in reasons.items() if k not in ('ok','core_parity','core_crc','aux_crc') and not k.startswith('detection:'))}
    assert successes+sum(breakdown.values())==N
    # Read pre-existing individual records AFTER selection & after all new outcomes are written.
    prev=[]
    for p in [OLD_DIR/'independent_0000_0499.csv',OLD_DIR/'independent_0500_0999.csv']:
        with p.open(newline='') as f:prev+=list(csv.DictReader(f))
    assert len(prev)==N
    assert [int(e['i']) for e in prev]==list(range(N))
    assert sum(int(e['success']) for e in prev)==944
    mismatches=[]
    outcomes=Counter()
    for curr,old in zip(events,prev):
        assert curr['payload_seed']==old['payload_seed']
        assert curr['camera_seed']==old['camera_seed']
        assert curr['payload_sha256']==old['payload_sha256']
        if curr['base_reason']!=old['reason']:mismatches.append((curr['i'],curr['base_reason'],old['reason']))
        outcomes[(int(old['success']),int(curr['success']))]+=1
    if mismatches:
        raise AssertionError(f'baseline mismatch count {len(mismatches)} first: {mismatches[:8]}')
    wins=outcomes[(0,1)]
    losses=outcomes[(1,0)]
    p_one=float(binomtest(wins,wins+losses,p=.5,alternative='greater').pvalue) if wins+losses else 1.0
    median=float(np.median([float(e['decode_elapsed_ms']) for e in events]))
    p95=float(np.quantile([float(e['decode_elapsed_ms']) for e in events],.95))
    point_99=(successes>=990)
    wilson_99=(wilson(successes,N)[0]>=.99)
    improvement=(successes>944 and p_one<.05)
    summary={
      'complete':True,'n':N,'payload_bytes':140,'channel':'hard',
      'payload_seed_expression':'331100000+i','camera_seed_expression':'442100000+i','index_range':[0,999],
      'matrix_name':frozen['matrix_name'],'matrix_sha256':sh(MATRIX),
      'selected_config':frozen['config'],
      'success':successes,'success_rate':successes/N,'success_wilson95':wilson(successes,N),
      'aux_crc_failure':breakdown['aux_crc'],'aux_crc_failure_wilson95':wilson(breakdown['aux_crc'],N),
      'failure_reasons':breakdown,'raw_reasons':dict(reasons),
      'previous_success':944,'delta_success':successes-944,'delta_rate_pp':(successes-944)/10,
      'paired_table':{'both_success':outcomes[(1,1)],'old_only':losses,'new_only':wins,'both_fail':outcomes[(0,0)]},
      'paired_mcnemar_one_sided_exact_p':p_one,
      'criteria':{'improvement':'new success > 944 and exact paired one-sided p<0.05',
                 '99pct_point':'new success >= 990',
                 '99pct_statistical':'Wilson95 lower bound >=0.99'},
      'judgment':{'improvement':'PASS' if improvement else ('FAIL' if successes<944 else 'UNCERTAIN'),
                  '99pct_point':'PASS' if point_99 else 'FAIL',
                  '99pct_statistical':'PASS' if wilson_99 else 'FAIL'},
      'rescue_count':sum(int(e['rescued']) for e in events),'retry_count':sum(int(e['attempted']) for e in events),
      'median_decode_ms':median,'p95_decode_ms':p95,
      'sample_records_sha256':{p.name:sh(p) for p in chunks},
      'environment':{'python':sys.version.split()[0], 'numpy':np.__version__,
          'cpu':platform.processor(),'threads':os.environ.get('OPENBLAS_NUM_THREADS'),
          'batch':1,'cpu_clock':'not pinned'},
      'no_original_regression':losses==0,'base_replay_1000_agrees':len(mismatches)==0
    }
    out=OUT/'joint_final_summary.json'
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--chunk',action='store_true')
    arg.add_argument('--finalize',action='store_true')
    arg.add_argument('--start',type=int)
    arg.add_argument('--stop',type=int)
    args=arg.parse_args()
    if args.chunk:chunk(args.start,args.stop)
    elif args.finalize:finalize()
    else:arg.error('select --chunk or --finalize')
