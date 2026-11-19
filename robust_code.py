"""Experimental v0.2: fixed frame + 37 spatial auxiliary data modules.
Data are ON the optical symbol, not in a model. Adversarial search is for
physical channel conditions and symbol layout, not an attack on users.
Not an ISO barcode. Reproducible synthetic image-only experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import binascii
import math
import struct
import time
import numpy as np
import lab_v01 as base

base.DETECTION_THRESHOLD=160
base.MORPH_KERNEL=3

POOL=base.available_coords()
# Historical layout: first 1152 core sites, remaining 37 previously unused sites.
LEGACY_CORE=base.cell_coords('interleaved')
LEGACY_EXTRA=np.asarray([rc for rc in POOL if tuple(rc) not in set(map(tuple,LEGACY_CORE))],dtype=np.int32)
assert len(LEGACY_EXTRA)==37

@dataclass(frozen=True)
class Layout:
    name: str
    core: np.ndarray
    extra: np.ndarray

LEGACY_LAYOUT=Layout('unused_37',LEGACY_CORE,LEGACY_EXTRA)

@lru_cache(None)
def extra_parity_indices(k:int, degree:int=5):
    r=74-k
    rng=np.random.default_rng(8262146 + k)
    rows=np.empty((r,degree),dtype=np.int32)
    # systematic sparse code, randomly-generated but deterministic and reproducible
    for i in range(r):
        rows[i]=rng.choice(k,degree,replace=False)
    # assure each information bit participates in >=2 parity checks
    while True:
        deg=np.bincount(rows.ravel(),minlength=k)
        lows=np.flatnonzero(deg<2)
        if not len(lows):break
        for j in lows:
            high_candidates=np.flatnonzero(deg>2)
            if not len(high_candidates):raise RuntimeError('could not balance sparse checks')
            done=False
            for v in rng.permutation(high_candidates):
                for i in rng.permutation(r):
                    if v in rows[i] and j not in rows[i]:
                        rows[i,np.where(rows[i]==v)[0][0]]=j
                        deg[j]+=1;deg[v]-=1;done=True;break
                if done:break
            if not done:raise RuntimeError('could not balance sparse checks')
    return rows

def encode_aux(payload:bytes)->np.ndarray:
    k=(len(payload)+2)*8
    if len(payload) not in (2,3,4):raise ValueError('aux payload 2..4 bytes')
    crc=binascii.crc_hqx(payload,0xffff)
    source=payload+struct.pack('>H',crc)
    u=np.unpackbits(np.frombuffer(source,dtype=np.uint8))
    assert len(u)==k
    parity=np.bitwise_xor.reduce(u[extra_parity_indices(k)],axis=1)
    return np.concatenate((u,parity)).astype(np.uint8)

def decode_aux(llr:np.ndarray,extra_bytes:int,iterations:int=24,alpha:float=.8)->tuple[bytes|None,str,int]:
    """Normalized min-sum for systematically encoded (74, 8*(extra_bytes+2)) bits."""
    n=74;k=(extra_bytes+2)*8;r=n-k
    edges=np.concatenate((extra_parity_indices(k), np.arange(k,k+r)[:,None]),axis=1)
    c_to_v=np.zeros(edges.shape,dtype=np.float32)
    v_to_c=np.clip(llr[edges],-30,30).copy()
    total=llr.copy()
    success=False
    for it in range(iterations):
        a=np.abs(v_to_c)
        mn_idx=np.argmin(a,axis=1)
        mn1=a.min(axis=1)
        mn2=np.partition(a,1,axis=1)[:,1]
        signs=np.where(v_to_c<0,-1.,1.).astype(np.float32)
        products=np.prod(signs,axis=1)
        mags=np.broadcast_to(mn1[:,None],edges.shape).copy()
        mags[np.arange(r),mn_idx]=mn2
        c_to_v=(alpha * mags * products[:,None]*signs).astype(np.float32)
        total=llr.copy()
        np.add.at(total,edges.ravel(),c_to_v.ravel())
        hard=(total<0).astype(np.uint8)
        if not np.bitwise_xor.reduce(hard[edges],axis=1).any():
            success=True;break
        v_to_c=np.clip(total[edges]-c_to_v,-30,30)
    data=np.packbits(hard[:k]).tobytes()
    expected=struct.unpack('>H',data[-2:])[0]
    if binascii.crc_hqx(data[:-2],0xffff)!=expected:return None,'aux_crc',it+1
    return data[:-2],('ok' if success else 'aux_crc_only'),it+1

# Selected from adversarially generated development channels ONLY.
CORE_A=11
CORE_OFFSETS=(463,466,582,539,252,1098)

def core_graph_neighbors():
    k=1152
    j=np.arange(k,dtype=np.int64)[:,None]
    return ((j*CORE_A+np.asarray(CORE_OFFSETS,dtype=np.int64)[None,:])%k).astype(np.int32)

def encode_core(payload:bytes,graph:str='legacy')->np.ndarray:
    bits=base.encode_bits(payload,'rgb4')
    if graph=='legacy':return bits
    if graph!='adversarial':raise ValueError('graph must be legacy or adversarial')
    k=len(bits)//2
    parity=np.bitwise_xor.reduce(bits[:k][core_graph_neighbors()],axis=1)
    return np.concatenate([bits[:k],parity]).astype(np.uint8)

CORE_MAX_ITER=18
def decode_core(llr:np.ndarray,graph:str='legacy',iterations:int|None=None,alpha:float=.82):
    if iterations is None:iterations=CORE_MAX_ITER
    if graph=='legacy':return base.ldpc_decode(llr)
    if graph!='adversarial':raise ValueError('graph must be legacy or adversarial')
    n=len(llr);k=n//2
    edges=np.concatenate([core_graph_neighbors(),(np.arange(k)+k)[:,None]],axis=1)
    r=k
    msg=np.clip(llr[edges],-30,30).copy()
    total=llr.copy()
    for it in range(iterations):
        absmsg=np.abs(msg)
        am=np.argmin(absmsg,axis=1)
        v1=absmsg.min(axis=1)
        v2=np.partition(absmsg,1,axis=1)[:,1]
        signs=np.where(msg<0,-1.,1.).astype(np.float32)
        sd=np.prod(signs,axis=1)
        mag=np.broadcast_to(v1[:,None],msg.shape).copy()
        mag[np.arange(r),am]=v2
        c2v=alpha*mag*sd[:,None]*signs
        total=llr.copy()
        np.add.at(total,edges.ravel(),c2v.ravel())
        hard=(total<0).astype(np.uint8)
        if not np.bitwise_xor.reduce(hard[edges],axis=1).any():
            return hard[:k],True,it+1
        msg=np.clip(total[edges]-c2v,-40,40)
    return hard[:k],False,iterations

@dataclass
class Result:
    payload:bytes|None
    reason:str
    elapsed_ms:float
    core_iter:int=0
    extra_iter:int=0

def encode(payload:bytes,layout:Layout=LEGACY_LAYOUT,cell_px:int=6,graph:str="legacy")->np.ndarray:
    extra_bytes=len(payload)-138
    if extra_bytes not in (2,3,4):raise ValueError('total payload needs 140/141/142 bytes')
    core_bits=encode_core(payload[:138],graph)
    out=base.image_from_bits(core_bits,'rgb4',cell_px=cell_px)
    # image_from_bits default layout v01; rewrite core with vectorized cell-grid mapping.
    if layout.name!=LEGACY_LAYOUT.name:
        code_grid=np.ones((base.SIZE,base.SIZE,3),dtype=np.float32)
        code_grid[0,:,:]=0;code_grid[-1,:,:]=0;code_grid[:,0,:]=0;code_grid[:,-1,:]=0
        code_grid[1:4,1:4]=base.PILOTS[0]
        code_grid[1:4,-4:-1]=base.PILOTS[1]
        code_grid[-4:-1,-4:-1]=base.PILOTS[2]
        code_grid[-4:-1,1:4]=base.PILOTS[3]
        symbols=core_bits.reshape(-1,2)
        indices=symbols[:,0]*2+symbols[:,1]
        code_grid[layout.core[:,0],layout.core[:,1]]=base.PALETTES['rgb4'][indices]
        canvas=np.ones((base.GRID,base.GRID,3),dtype=np.float32)
        canvas[base.QUIET:base.QUIET+base.SIZE,base.QUIET:base.QUIET+base.SIZE]=code_grid
        out=np.repeat(np.repeat(canvas,cell_px,axis=0),cell_px,axis=1)
    extra=encode_aux(payload[138:]);syms=extra.reshape(-1,2)
    indices=syms[:,0]*2+syms[:,1]
    for (r,c),i in zip(layout.extra,indices):
        rr=(base.QUIET+int(r))*cell_px
        cc=(base.QUIET+int(c))*cell_px
        out[rr:rr+cell_px,cc:cc+cell_px]=base.PALETTES['rgb4'][i]
    return out

def decode(img:np.ndarray,extra_bytes:int=2,layout:Layout=LEGACY_LAYOUT,cell_px:int=6,graph:str="legacy")->Result:
    start=time.perf_counter()
    try:
        grid=base.locate_and_calibrate(img,cell_px=cell_px)
    except base.DetectionError as e:
        return Result(None,'detection:'+str(e),(time.perf_counter()-start)*1000)
    core_sample=grid[layout.core[:,0],layout.core[:,1]]
    llr=base.llr_from_colors(core_sample,'rgb4')
    info,parity,coreit=decode_core(llr,graph)
    payload=base.extract_payload(info)
    if payload is None:return Result(None,'core_crc' if parity else 'core_parity',(time.perf_counter()-start)*1000,coreit)
    extra_sample=grid[layout.extra[:,0],layout.extra[:,1]]
    extrallr=base.llr_from_colors(extra_sample,'rgb4',llr_sigma=.22)
    ext,reason,auxit=decode_aux(extrallr,extra_bytes)
    if ext is None:return Result(None,reason,(time.perf_counter()-start)*1000,coreit,auxit)
    return Result(payload+ext,'ok',(time.perf_counter()-start)*1000,coreit,auxit)

def base_decode(image:np.ndarray,cell_px:int=6)->Result:
    d=base.decode(image,'rgb4',cell_px=cell_px)
    return Result(d.payload,d.reason,d.elapsed_ms,d.iterations)

if __name__=='__main__':
    for extra in (2,3,4):
        data=np.random.default_rng(123).bytes(138+extra)
        im=encode(data)
        out=decode(im,extra_bytes=extra)
        print('aux',extra,'roundtrip',out.payload==data,out.reason,out.elapsed_ms)
