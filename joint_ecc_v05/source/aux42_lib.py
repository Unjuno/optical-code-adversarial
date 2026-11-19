"""140-byte auxiliary-only parity-matrix reselection; all non-aux components frozen.

The 42x32 systematic parity support P defines the full 42x74 check matrix H=[P|I42].
No training example or observation is stored in a symbol/model; only 2 payload bytes.
"""
from __future__ import annotations
import os, sys, binascii, struct, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'optical_code_adversarial_v02'))
import robust_code as robust
import lab_v01 as base

N=74; K=32; R=42; DATA_BYTES=140
FROZEN_THRESHOLD=140

def freeze_settings():
    base.DETECTION_THRESHOLD=FROZEN_THRESHOLD
    base.MORPH_KERNEL=3
    # Use one thread for interop stability; does not change code semantics.
    import cv2
    cv2.setNumThreads(1)

def validate_rows(rows:np.ndarray):
    rows=np.asarray(rows,dtype=np.int32)
    assert rows.ndim==2 and rows.shape[0]==R, rows.shape
    assert np.all((rows>=0)&(rows<K)),rows
    assert all(len(np.unique(row))==rows.shape[1] for row in rows)
    assert min(np.bincount(rows.ravel(),minlength=K))>=1
    return rows

def old_rows()->np.ndarray:
    return validate_rows(robust.extra_parity_indices(K).copy())

def generate_rows(degree:int,seed:int)->np.ndarray:
    # Degree-constrained, load-balanced greedy column selection.
    # Penalizes pairs that already co-occur in any parity-check equation.
    rng=np.random.default_rng(seed)
    counts=np.zeros(K,dtype=np.float64)
    pair=np.zeros((K,K),dtype=np.float64)
    rows=[]
    for r in range(R):
        selected=[]
        for j in range(degree):
            cost=counts*.74
            if selected:
                cost=cost+3.6*pair[np.asarray(selected)].sum(axis=0)
            cost=cost+rng.uniform(-.58,.58,size=K)
            cost[selected]=np.inf
            x=int(np.argmin(cost))
            selected.append(x)
        rows.append(selected)
        for i in selected: counts[i]+=1
        for i in selected:
            for j in selected:
                if i!=j:pair[i,j]+=1
    return validate_rows(np.asarray(rows,dtype=np.int32))

def rows_score(rows:np.ndarray)->dict:
    # 4-cycles in Tanner graph: sum over pairs of systematic bits C(cooccurrence,2).
    counts=np.zeros((K,K),dtype=np.int16)
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1,len(row)):
                counts[row[i],row[j]]+=1;counts[row[j],row[i]]+=1
    v=counts[np.triu_indices(K,1)].astype(int)
    deg=np.bincount(rows.ravel(),minlength=K)
    return {'four_cycles':int(np.sum(v*(v-1)//2)), 'column_degree_min':int(deg.min()),
            'column_degree_max':int(deg.max()),'column_degree_std':round(float(deg.std()),4)}

def encode_aux_rows(aux:bytes,rows:np.ndarray)->np.ndarray:
    assert len(aux)==2
    source=aux+struct.pack('>H',binascii.crc_hqx(aux,0xffff))
    u=np.unpackbits(np.frombuffer(source,dtype=np.uint8))
    parity=np.bitwise_xor.reduce(u[rows],axis=1)
    return np.concatenate((u,parity)).astype(np.uint8)

def encode140(payload:bytes,rows:np.ndarray)->np.ndarray:
    assert len(payload)==DATA_BYTES
    out=robust.encode(payload,robust.LEGACY_LAYOUT,graph='adversarial')
    extra=encode_aux_rows(payload[138:],rows)
    palette=base.PALETTES['rgb4']
    idx=extra.reshape(-1,2)[:,0]*2+extra.reshape(-1,2)[:,1]
    for (r,c),symbol in zip(robust.LEGACY_EXTRA,idx):
        rr=(base.QUIET+int(r))*6;cc=(base.QUIET+int(c))*6
        out[rr:rr+6,cc:cc+6]=palette[symbol]
    return out

def decode_aux_rows(llr:np.ndarray,rows:np.ndarray,iterations:int=24,alpha:float=.8):
    assert len(llr)==N
    edges=np.concatenate((rows,np.arange(K,N,dtype=np.int32)[:,None]),axis=1)
    v_to_c=np.clip(llr[edges],-30,30).copy()
    for it in range(iterations):
        a=np.abs(v_to_c)
        min_idx=np.argmin(a,axis=1)
        mn1=a.min(axis=1)
        mn2=np.partition(a,1,axis=1)[:,1]
        signs=np.where(v_to_c<0,-1.,1.).astype(np.float32)
        products=np.prod(signs,axis=1)
        mags=np.broadcast_to(mn1[:,None],edges.shape).copy()
        mags[np.arange(R),min_idx]=mn2
        c_to_v=(alpha*mags*products[:,None]*signs).astype(np.float32)
        total=llr.copy()
        np.add.at(total,edges.ravel(),c_to_v.ravel())
        hard=(total<0).astype(np.uint8)
        if not np.bitwise_xor.reduce(hard[edges],axis=1).any():
            parity=True;break
        v_to_c=np.clip(total[edges]-c_to_v,-30,30)
    else:
        parity=False
    data=np.packbits(hard[:K]).tobytes()
    got=struct.unpack('>H',data[-2:])[0]
    if binascii.crc_hqx(data[:-2],0xffff)!=got:
        return None,'aux_crc',it+1
    return data[:-2],('ok' if parity else 'aux_crc_only'),it+1

def decode140(image:np.ndarray,rows:np.ndarray):
    start=time.perf_counter()
    try: grid=base.locate_and_calibrate(image)
    except base.DetectionError as e:
        return robust.Result(None,'detection:'+str(e),(time.perf_counter()-start)*1000)
    core_llr=base.llr_from_colors(grid[robust.LEGACY_CORE[:,0],robust.LEGACY_CORE[:,1]],'rgb4')
    info,parity,coreit=robust.decode_core(core_llr,'adversarial')
    core=base.extract_payload(info)
    if core is None:
        return robust.Result(None, 'core_crc' if parity else 'core_parity',(time.perf_counter()-start)*1000,coreit)
    aux_llr=base.llr_from_colors(grid[robust.LEGACY_EXTRA[:,0],robust.LEGACY_EXTRA[:,1]],'rgb4',llr_sigma=.22)
    ext,reason,auxit=decode_aux_rows(aux_llr,rows)
    if ext is None:
        return robust.Result(None,reason,(time.perf_counter()-start)*1000,coreit,auxit)
    return robust.Result(core+ext,'ok',(time.perf_counter()-start)*1000,coreit,auxit)

def decode_aux_only(image:np.ndarray,rows:np.ndarray,aux:bytes):
    try: grid=base.locate_and_calibrate(image)
    except base.DetectionError:
        return None, 'detection'
    extra_llr=base.llr_from_colors(grid[robust.LEGACY_EXTRA[:,0],robust.LEGACY_EXTRA[:,1]],'rgb4',llr_sigma=.22)
    ext,reason,_=decode_aux_rows(extra_llr,rows)
    return (ext==aux),reason

def get_crc_valid_generator():
    """16 independent bits -> valid (payload + CRC16 differential) 32-bit space."""
    b=[]
    for i in range(16):
        v=1<<(15-i)
        data=v.to_bytes(2,'big')
        crc=binascii.crc_hqx(data,0)
        b.append(np.unpackbits(np.frombuffer(data+crc.to_bytes(2,'big'),dtype=np.uint8)))
    return np.stack(b)

def spectrum(rows:np.ndarray)->dict:
    """Exact minimum Hamming distance across all 65536 CRC16-valid auxiliary payloads."""
    b=get_crc_valid_generator()
    g=np.concatenate((b,np.bitwise_xor.reduce(b[:,rows],axis=2)),axis=1)
    look=[]
    for ch in range(4):
        patterns=np.unpackbits(np.arange(16,dtype='>u2').view('u1').reshape(16,2),axis=1)[:,-4:]
        lookup=np.zeros((16,N),dtype=np.uint8)
        for j in range(4):lookup^=patterns[:,j,None]*g[ch*4+j]
        look.append(lookup)
    idx=np.arange(1,65536,dtype=np.uint16)
    code=look[0][idx>>12]^look[1][(idx>>8)&15]^look[2][(idx>>4)&15]^look[3][idx&15]
    weight=code.sum(axis=1,dtype=np.uint16)
    return {'min_distance':int(weight.min()),'weight_at_min':int(np.sum(weight==weight.min())),
            'weight_le_16':int(np.sum(weight<=16)), 'weight_le_20':int(np.sum(weight<=20)),
            'weight_le_24':int(np.sum(weight<=24))}

def selftest():
    freeze_settings(); rows=old_rows()
    payload=np.random.default_rng(318190000).bytes(140)
    old=robust.encode(payload,graph='adversarial')
    new=encode140(payload,rows)
    assert np.array_equal(old,new),'old-row encoder mismatch'
    for i in range(12):
        p=np.random.default_rng(910000+i).bytes(140)
        image=base.camera(robust.encode(p,graph='adversarial'),base.CHANNELS['hard'],np.random.default_rng(920000+i))
        x=robust.decode(image,extra_bytes=2,graph='adversarial')
        y=decode140(image,rows)
        assert x.reason==y.reason and x.payload==y.payload and x.extra_iter==y.extra_iter, (i,x,y)
    assert np.concatenate((rows,np.arange(32,74)[:,None]),axis=1).shape==(42,rows.shape[1]+1)
    print('SELFTEST PASS: old and variant agree for encoder, decoder, 12 independent development channels',flush=True)
if __name__=='__main__':selftest()
