"""Experimental 8-color high-capacity auxiliary payload in 37 spare cells.
Same 138-byte RGB4 main core, 8-state RGB auxiliary cells (111 coded bits).
Same physical footprint and unmodified locator border.
"""
from __future__ import annotations
import math,struct,binascii,time
from functools import lru_cache
import numpy as np
import lab_v01 as old
import robust_code as r

@lru_cache(None)
def check_rows(k:int,n:int=111,degree:int=6):
  rg=np.random.default_rng(309181+k)
  m=n-k
  rows=np.empty((m,degree),dtype=np.int32)
  for j in range(m):
    rows[j]=rg.choice(k,degree,replace=False)
  for _ in range(500):
    deg=np.bincount(rows.ravel(),minlength=k)
    lows=np.flatnonzero(deg<2)
    if not len(lows):return rows
    j=int(lows[0]);upper=np.flatnonzero(deg>2)
    matched=False
    for v in rg.permutation(upper):
      for i in rg.permutation(m):
        if v in rows[i] and j not in rows[i]:
          rows[i,np.where(rows[i]==v)[0][0]]=j;matched=True;break
      if matched:break
    if not matched:break
  return rows

def aux_encoded(payload:bytes)->np.ndarray:
  k=(len(payload)+2)*8
  if not 4<=len(payload)<=8:raise ValueError('extra bytes 4..8')
  source=payload+struct.pack('>H',binascii.crc_hqx(payload,0xffff))
  u=np.unpackbits(np.frombuffer(source,dtype=np.uint8))
  checks=check_rows(k)
  parity=np.bitwise_xor.reduce(u[checks],axis=1)
  encoded=np.concatenate((u,parity))
  assert len(encoded)==111
  return encoded

def aux_decoded(llr:np.ndarray,extra:int,iterations:int=20,alpha:float=.8):
  k=(extra+2)*8;n=111;par=n-k
  edges=np.concatenate((check_rows(k),(np.arange(par)+k)[:,None]),axis=1)
  v_to_c=llr[edges].copy()
  for it in range(iterations):
    vals=np.abs(v_to_c);i_min=np.argmin(vals,axis=1)
    mn1=vals.min(axis=1);mn2=np.partition(vals,1,axis=1)[:,1]
    s=np.where(v_to_c<0,-1.,1.).astype(np.float32)
    prod=s.prod(axis=1)
    m=np.broadcast_to(mn1[:,None],v_to_c.shape).copy()
    m[np.arange(par),i_min]=mn2
    c_to_v=alpha*m*prod[:,None]*s
    t=llr.copy();np.add.at(t,edges.ravel(),c_to_v.ravel())
    hard=(t<0).astype(np.uint8)
    if not np.bitwise_xor.reduce(hard[edges],axis=1).any():break
    v_to_c=np.clip(t[edges]-c_to_v,-30,30)
  bs=np.packbits(hard[:k]).tobytes()
  if binascii.crc_hqx(bs[:-2],0xffff)!=struct.unpack('>H',bs[-2:])[0]:return None
  return bs[:-2]

def encode(payload:bytes,layout:r.Layout=r.LEGACY_LAYOUT):
  extra=payload[138:]
  m=r.encode(payload[:138]+b'\0\0',layout=layout,graph='adversarial')
  bits=aux_encoded(extra).reshape(-1,3)
  syms=(bits[:,0]*4+bits[:,1]*2+bits[:,2]).astype(int)
  for (rr,cc),sym in zip(layout.extra,syms):
    x=(old.QUIET+rr)*6;y=(old.QUIET+cc)*6
    m[x:x+6,y:y+6]=old.PALETTES['rgb8'][sym]
  return m

def decode(img,extra_bytes:int,layout:r.Layout=r.LEGACY_LAYOUT):
  try:g=old.locate_and_calibrate(img)
  except old.DetectionError:return None,'detect'
  llr=old.llr_from_colors(g[layout.core[:,0],layout.core[:,1]],'rgb4')
  info,valid,itt=r.decode_core(llr,'adversarial')
  core=old.extract_payload(info)
  if core is None:return None,'core'
  f_llr=old.llr_from_colors(g[layout.extra[:,0],layout.extra[:,1]],'rgb8',llr_sigma=.22)
  out=aux_decoded(f_llr,extra_bytes)
  if out is None:return None,'aux'
  return core+out,'ok'

if __name__=='__main__':
 import numpy as np
 for ex in (4,6,8):
  p=np.random.default_rng(ex).bytes(138+ex)
  d,why=decode(encode(p),ex)
  print(ex,why,d==p)
