"""Independent structural audit: reads only saved CSV/JSON; does not run or alter receiver."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest
P=Path(__file__).resolve().parent
SRC=P/'results'
OLD=Path('/mnt/data/independent_140_v04/results')

def rd(paths):
    out=[]
    for path in paths:
        with path.open(newline='') as f:out+=list(csv.DictReader(f))
    return out

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

new_files=[SRC/'joint_final_0000_0499.csv',SRC/'joint_final_0500_0999.csv']
old_files=[OLD/'independent_0000_0499.csv',OLD/'independent_0500_0999.csv']
new=rd(new_files);old=rd(old_files)
sm=json.loads((SRC/'joint_final_summary.json').read_text());frozen=json.loads((P/'FREEZE_SELECTION.json').read_text())
assert len(new)==len(old)==1000
assert sm['complete'] and sm['n']==1000
assert sm['matrix_sha256']==sha(P/'source/selected_aux42_rows.npy')==frozen['matrix_sha256']
for f,h in frozen['sha256_source'].items():assert sha(P/f)==h
assert [int(e['i']) for e in new]==[int(e['i']) for e in old]==list(range(1000))
for n,o in zip(new,old):
    i=int(n['i'])
    assert n['payload_seed']==o['payload_seed']==str(331100000+i)
    assert n['camera_seed']==o['camera_seed']==str(442100000+i)
    assert n['payload_sha256']==o['payload_sha256']
    assert n['base_reason']==o['reason']
    p=np.random.default_rng(331100000+i).bytes(142)[:140]
    assert n['payload_sha256']==hashlib.sha256(p).hexdigest()
    if n['success']=='1':
        assert n['reason']=='ok'
        assert n['decoded_sha256']==n['payload_sha256']
    else:
        assert n['reason']!='ok'
        assert not n['decoded_sha256']
    if n['rescued']=='1':
        assert n['success']=='1' and o['success']=='0' and n['attempted']=='1'
    if o['success']=='1':assert n['success']=='1'
reasons=Counter(n['reason'] for n in new)
assert reasons['ok']==sm['success']==974
assert sm['failure_reasons']=={'core_parity':15,'core_crc':2,'aux_crc':4,'detection':5,'other':0}
assert sum(reasons.values())==1000
wins=sum(o['success']=='0' and n['success']=='1' for n,o in zip(new,old))
losses=sum(o['success']=='1' and n['success']=='0' for n,o in zip(new,old))
assert (wins,losses)==(30,0)
assert sm['paired_mcnemar_one_sided_exact_p']==binomtest(wins,wins+losses,p=.5,alternative='greater').pvalue
assert sm['rescue_count']==sum(int(n['rescued']) for n in new)
print(json.dumps({'audit':'PASS', 'n':1000,'success':974,'failed':26,'reasons':dict(reasons),
 'paired_wins':wins,'paired_losses':losses,'sha256_chunks':{p.name:sha(p) for p in new_files},
 'all_seed_hashes_confirmed':True,'all_baseline_reasons_match':True,
 'all_original_successes_kept':True},ensure_ascii=False,indent=2))
