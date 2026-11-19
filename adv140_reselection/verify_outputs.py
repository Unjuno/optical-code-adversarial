"""Verify frozen matrix/protocol and produced 1000-item result; no re-training."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import aux42_lib as a
O=Path(__file__).resolve().parent/'outputs'
fs=json.loads((O/'frozen_selection.json').read_text());r=np.load(O/'selected_aux42_rows.npy')
assert hashlib.sha256((O/'selected_aux42_rows.npy').read_bytes()).hexdigest()==fs['sha256_of_npy']
a.validate_rows(r)
s=json.loads((O/'frozen_holdout_summary.json').read_text())
assert s['n']==1000 and s['old_success']==928 and s['new_success']==939
assert s['old_aux_crc_total']==13 and s['new_aux_crc_total']==6
assert s['prereg_verdict']=='PASS'
records=[]
for f in sorted(O.glob('holdout_pair_*.csv')):
    with f.open() as fh:records.extend(csv.DictReader(fh))
assert [int(x['i']) for x in records]==list(range(1000))
ref=np.load(Path(__file__).resolve().parent/'archive_outcomes_hard_advgraph140.npy')
assert np.array_equal(np.asarray([int(x['old_success']) for x in records],dtype=np.uint8),ref)
assert sum(int(x['new_success']) for x in records)==939
assert a.spectrum(r)['min_distance']==19
print('VERIFIED:',len(records),'paired rows, old=928, new=939, aux_crc=13->6, PASS, frozen SHA and code distance verified.')
