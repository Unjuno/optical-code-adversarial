"""Read-only independent CSV audit, no optical decoding or matrix tuning."""
from pathlib import Path
import csv,hashlib,json
from collections import Counter
import numpy as np
r=Path(__file__).resolve().parent
rows=[]
for p in sorted((r/'results').glob('independent_????_????.csv')):
    with p.open(newline='') as f:rows.extend(csv.DictReader(f))
assert len(rows)==1000
assert [int(x['i']) for x in rows]==list(range(1000))
for x in rows:
    i=int(x['i']);assert int(x['payload_seed'])==331100000+i
    assert int(x['camera_seed'])==442100000+i
    dat=np.random.default_rng(331100000+i).bytes(142)[:140]
    assert hashlib.sha256(dat).hexdigest()==x['payload_sha256']
    if int(x['success']):
        assert x['payload_sha256']==x['decoded_sha256'] and x['reason']=='ok'
    else:
        assert x['reason']!='ok'
rs=Counter(x['reason'] for x in rows)
s=json.loads((r/'results/independent_summary.json').read_text())
assert s['n']==len(rows) and s['success']==rs['ok']==944
assert s['failure_reasons']=={'core_parity':39,'core_crc':8,'aux_crc':4,'detection':5}
assert sum(rs.values())==1000
print('AUDIT PASS: 1000 unique consecutive cases, required seeds, payload hashes, success flags and counts',flush=True)
print(json.dumps(rs,ensure_ascii=False),flush=True)
