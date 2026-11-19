"""Optical Code Lab v0.1: self-contained binary payload, multi-state optical modulation.

Research prototype, NOT QR compliant and NOT suitable for production storage.
Decoder estimates four fiducials and orientation from the image (no oracle pose).
The LDPC-like systematic sparse binary code is intentionally elementary; no claims
of near-capacity performance. Compare under explicitly held-fixed imaging conditions.
"""
from __future__ import annotations
import dataclasses
import math
import struct
import time
import zlib
from functools import lru_cache
import cv2
import numpy as np
from scipy.special import logsumexp

SIZE = 37
QUIET = 4
GRID = SIZE + 2 * QUIET
NCELLS = 1152
MARKER = 3
CHANNELS = 3

DETECTION_THRESHOLD = 85  # 8-bit max(R,G,B) cut to find dark border
MORPH_KERNEL = 3

# Fixed 3x3 corner pilots. Values stored in RGB order.
PILOTS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)

PALETTES = {
    "binary": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "gray4": np.array([[0, 0, 0], [1/3, 1/3, 1/3], [2/3, 2/3, 2/3], [1, 1, 1]],dtype=np.float32),
    "rgb4": np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],dtype=np.float32),
    "rgb8": np.array([[(s >> 2) & 1, (s >> 1) & 1, s & 1] for s in range(8)],dtype=np.float32),
}

@lru_cache(None)
def available_coords() -> np.ndarray:
    coords=[]
    for r in range(1,SIZE-1):
        for c in range(1,SIZE-1):
            if ((r<=MARKER and c<=MARKER) or (r<=MARKER and c>=SIZE-1-MARKER)
                or (r>=SIZE-1-MARKER and c<=MARKER) or (r>=SIZE-1-MARKER and c>=SIZE-1-MARKER)):
                continue
            coords.append((r,c))
    return np.asarray(coords,dtype=np.int32)

@lru_cache(None)
def cell_coords(mapping: str="interleaved") -> np.ndarray:
    candidates=available_coords()
    if mapping=="interleaved":
        rng=np.random.default_rng(20261115)
        order=rng.permutation(len(candidates))
        return candidates[order[:NCELLS]]
    if mapping=="raster":
        return candidates[:NCELLS]
    raise ValueError("unknown mapping: " + mapping)

@lru_cache(None)
def parity_neighbors(k:int, degree:int=4)->np.ndarray:
    """Each row has 4 unique info vars. Each column visits every info bit."""
    # affinely permute information indices for each check degree
    j=np.arange(k,dtype=np.int64)[:,None]
    offsets=np.array([0,37,151,293],dtype=np.int64)[:degree]
    a=37
    if math.gcd(a,k)!=1: a=5
    if math.gcd(a,k)!=1: a=1
    return ((j*a+offsets[None,:])%k).astype(np.int32)

def get_bpc(mode:str)->int:
    return {"binary":1,"gray4":2,"rgb4":2,"rgb8":3}[mode]

def payload_size(mode:str)->int:
    # Code rate = 1/2, 2 bytes length + 4 bytes CRC32.
    return NCELLS*get_bpc(mode)//16 - 6

def encode_bits(payload:bytes,mode:str)->np.ndarray:
    k=NCELLS*get_bpc(mode)//2
    max_payload=payload_size(mode)
    if len(payload)>max_payload: raise ValueError(f"payload too long: {len(payload)}>{max_payload}")
    # deterministic padding, but keep payload and CRC in one message
    seed=zlib.crc32(payload)
    pad=np.random.default_rng(seed).integers(0,256,size=max_payload-len(payload),dtype=np.uint8).tobytes()
    info=struct.pack('>H',len(payload))+payload+pad+struct.pack('>I',zlib.crc32(payload))
    assert len(info)*8==k
    u=np.unpackbits(np.frombuffer(info,dtype=np.uint8),bitorder='big').astype(np.uint8)
    checks=parity_neighbors(k)
    parity=np.bitwise_xor.reduce(u[checks],axis=1)
    return np.concatenate([u,parity]).astype(np.uint8)

def image_from_bits(bits:np.ndarray,mode:str,mapping:str="interleaved", cell_px:int=6)->np.ndarray:
    palette=PALETTES[mode]
    bpc=get_bpc(mode)
    symbols=bits.reshape(-1,bpc)
    idx=np.zeros(len(symbols),dtype=np.int32)
    for j in range(bpc): idx=(idx<<1)|symbols[:,j]
    codes=np.ones((SIZE,SIZE,3),dtype=np.float32)
    codes[0,:,:]=0; codes[-1,:,:]=0; codes[:,0,:]=0; codes[:,-1,:]=0
    codes[1:4,1:4]=PILOTS[0]
    codes[1:4,SIZE-4:SIZE-1]=PILOTS[1]
    codes[SIZE-4:SIZE-1,SIZE-4:SIZE-1]=PILOTS[2]
    codes[SIZE-4:SIZE-1,1:4]=PILOTS[3]
    coords=cell_coords(mapping)
    codes[coords[:,0],coords[:,1]]=palette[idx]
    canvas=np.ones((GRID,GRID,3),dtype=np.float32)
    canvas[QUIET:QUIET+SIZE,QUIET:QUIET+SIZE]=codes
    return np.repeat(np.repeat(canvas,cell_px,axis=0),cell_px,axis=1)

def encode(payload:bytes,mode:str,mapping:str="interleaved",cell_px:int=6)->np.ndarray:
    return image_from_bits(encode_bits(payload,mode),mode,mapping,cell_px)

@dataclasses.dataclass(frozen=True)
class Channel:
    name:str
    blur:float
    pixel_noise:float
    cell_noise:float
    jitter:float
    rotation:float
    mix:float
    wb:float

CHANNELS = {
    "clean":Channel("clean",.55,.012,.008,0,0,0,0),
    "mild":Channel("mild",1.0,.035,.025,1.0,6.0,.020,.06),
    "medium":Channel("medium",1.6,.055,.045,2.0,9.0,.035,.09),
    "hard":Channel("hard",2.4,.085,.070,3.0,12.0,.055,.13),
    "extreme":Channel("extreme",3.2,.12,.09,4.0,15.0,.065,.16),
}

def camera(img:np.ndarray,condition:Channel,rng:np.random.Generator,cell_px:int=6)->np.ndarray:
    """Synthetic scene. Image-space, correlated cell noise, projective pose, blur, WB, RGB mixing."""
    h,w=img.shape[:2]
    if condition.cell_noise:
        # source-coordinate cell-wise smooth noise, fixed within modules
        hmod=h//cell_px; wmod=w//cell_px
        bias=rng.normal(0,condition.cell_noise,size=(hmod,wmod,3)).astype(np.float32)
        img=np.clip(img+np.repeat(np.repeat(bias,cell_px,axis=0),cell_px,axis=1),0,1)
    angle=rng.uniform(-condition.rotation,condition.rotation)
    R=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,1.0)
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dest=cv2.transform(src[None,:,:],R)[0]
    if condition.jitter:
        dest=dest+rng.uniform(-condition.jitter,condition.jitter,size=(4,2)).astype(np.float32)
    H=cv2.getPerspectiveTransform(src,dest.astype(np.float32))
    result=cv2.warpPerspective(img,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(1,1,1))
    if condition.blur:
        result=cv2.GaussianBlur(result,(0,0),condition.blur)
    if condition.mix:
        # rgb optical / device crosstalk; close to identity
        M=np.eye(3,dtype=np.float32)*(1-2*condition.mix)+np.ones((3,3),dtype=np.float32)*condition.mix
        result=np.einsum('hwc,dc->hwd',result,M)
    if condition.wb:
        gains=rng.uniform(1-condition.wb,1+condition.wb,size=(3,)).astype(np.float32)
        offs=rng.uniform(-condition.wb*.12,condition.wb*.12,size=(3,)).astype(np.float32)
        result=result*gains[None,None,:]+offs[None,None,:]
    if condition.pixel_noise:
        result=result+rng.normal(0,condition.pixel_noise,size=result.shape).astype(np.float32)
    return np.clip(result,0,1).astype(np.float32)


def order_quad(pts:np.ndarray)->np.ndarray:
    pts=pts.reshape(-1,2).astype(np.float32)
    s=pts[:,0]+pts[:,1]; d=pts[:,1]-pts[:,0]
    ids=[np.argmin(s),np.argmin(d),np.argmax(s),np.argmax(d)]
    if len(set(ids))!=4:
        raise ValueError("ambiguous quadrilateral")
    return pts[ids]

class DetectionError(Exception): pass

def locate_and_calibrate(img:np.ndarray,cell_px:int=6,border_data:bool=False)->np.ndarray:
    """Detect black 1-cell frame, rectify and calibrate RGB with four 3x3 pilots."""
    rgb=(np.clip(img,0,1)*255).astype(np.uint8)
    black=(np.max(rgb,axis=2)<DETECTION_THRESHOLD).astype(np.uint8)*255
    black=cv2.morphologyEx(black,cv2.MORPH_CLOSE,np.ones((MORPH_KERNEL,MORPH_KERNEL),np.uint8))
    contours,_=cv2.findContours(black,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    quadris=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<(SIZE*cell_px)**2*.35: continue
        hull=cv2.convexHull(cnt)
        peri=cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,.022*peri,True)
        if len(approx)==4:
            quadris.append((area,approx.reshape(4,2)))
    if not quadris: raise DetectionError('frame contour not found')
    _,corners=max(quadris,key=lambda x:x[0])
    try: ordered=order_quad(corners)
    except ValueError as e: raise DetectionError(str(e)) from e
    side=SIZE*cell_px
    dest=np.float32([[0,0],[side-1,0],[side-1,side-1],[0,side-1]])
    H=cv2.getPerspectiveTransform(ordered,dest)
    rect=cv2.warpPerspective(img,H,(side,side),flags=cv2.INTER_LINEAR,borderValue=(1,1,1))
    grid=cv2.resize(rect,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
    def pilots(g):
        return np.stack([g[1:4,1:4].mean(axis=(0,1)),g[1:4,-4:-1].mean(axis=(0,1)),
                         g[-4:-1,-4:-1].mean(axis=(0,1)),g[-4:-1,1:4].mean(axis=(0,1))])
    choices=[]
    for rotation in range(4):
        g=np.rot90(grid,rotation)
        p=pilots(g)
        score=float(np.sum((p-PILOTS)**2))
        choices.append((score,rotation,p))
    score,rotation,p=min(choices,key=lambda item:item[0])
    if score>4.0: raise DetectionError(f'orientation score too large {score:.2f}')
    g=np.rot90(grid,rotation)
    border=np.concatenate([g[0,5:-5,:],g[-1,5:-5,:],g[5:-5,0,:],g[5:-5,-1,:]],axis=0)
    if border_data:
        # Four black corner regions remain known references even when frame carries data.
        sample=[]
        for idx in list(range(1,5))+list(range(SIZE-5,SIZE-1)):
            sample.extend((g[0,idx],g[-1,idx],g[idx,0],g[idx,-1]))
        black_mean=np.mean(np.stack(sample),axis=0)
    else:
        black_mean=np.mean(border,axis=0)
    cal=np.stack([p[i]-black_mean for i in range(3)],axis=1)
    if abs(np.linalg.det(cal))<.08:
        raise DetectionError('singular color calibration')
    normalized=np.linalg.solve(cal,(g-black_mean).reshape(-1,3).T).T.reshape(SIZE,SIZE,3)
    return normalized.astype(np.float32)

def llr_from_colors(observed:np.ndarray, mode:str, llr_sigma:float=.19)->np.ndarray:
    palette=PALETTES[mode]
    bpc=get_bpc(mode)
    dist2=((observed[:,None,:]-palette[None,:,:])**2).sum(axis=2)
    scores=-dist2/(2*llr_sigma*llr_sigma)
    q=len(palette)
    col=[]
    for j in range(bpc):
        bit=((np.arange(q) >> (bpc-j-1))&1)
        col.append(logsumexp(scores[:,bit==0],axis=1)-logsumexp(scores[:,bit==1],axis=1))
    return np.clip(np.stack(col,axis=1).reshape(-1),-20,20).astype(np.float32)

def ldpc_decode(llr:np.ndarray,iterations:int=18,alpha:float=.82)->tuple[np.ndarray,bool,int]:
    """Normalized min-sum decoder. Outputs systematic info bits and parity pass."""
    n=len(llr)
    assert n%2==0
    k=n//2
    nbr=parity_neighbors(k)
    edges=np.concatenate([nbr,(np.arange(k)+k)[:,None]],axis=1)
    degree=edges.shape[1]
    msg_c2v=np.zeros((k,degree),dtype=np.float32)
    msg_v2c=llr[edges].copy()
    for ite in range(iterations):
        a=np.abs(msg_v2c)
        smallest_idx=np.argmin(a,axis=1)
        min1=np.min(a,axis=1)
        min2=np.partition(a,1,axis=1)[:,1]
        sign=np.where(msg_v2c<0,-1.,1.).astype(np.float32)
        prod=np.prod(sign,axis=1)
        scores=np.broadcast_to(min1[:,None],a.shape).copy()
        scores[np.arange(k),smallest_idx]=min2
        msg_c2v=alpha*scores*prod[:,None]*sign
        total=llr.copy()
        np.add.at(total,edges.ravel(),msg_c2v.ravel())
        hard=(total<0).astype(np.uint8)
        if not np.bitwise_xor.reduce(hard[edges],axis=1).any():
            return hard[:k],True,ite+1
        msg_v2c=np.clip(total[edges]-msg_c2v,-40,40)
    return hard[:k],False,iterations

def extract_payload(u:np.ndarray)->bytes|None:
    bits=np.packbits(u.astype(np.uint8),bitorder='big').tobytes()
    if len(bits)<6: return None
    length=struct.unpack('>H',bits[:2])[0]
    if length>len(bits)-6:return None
    payload=bits[2:2+length]
    crc=struct.unpack('>I',bits[-4:])[0]
    if zlib.crc32(payload)!=crc:return None
    return payload

@dataclasses.dataclass
class Decoded:
    payload:bytes|None
    reason:str
    elapsed_ms:float
    iterations:int


def decode(img:np.ndarray,mode:str,mapping:str="interleaved",cell_px:int=6, max_iter:int=18)->Decoded:
    t0=time.perf_counter()
    try:
        cells=locate_and_calibrate(img,cell_px=cell_px)
        coords=cell_coords(mapping)
        samples=cells[coords[:,0],coords[:,1]]
        llr=llr_from_colors(samples,mode)
        bits,checks,ite=ldpc_decode(llr,iterations=max_iter)
        recovered=extract_payload(bits)
        reason='ok' if recovered is not None else ('crc_failure' if checks else 'parity_failure')
        return Decoded(recovered,reason,(time.perf_counter()-t0)*1e3,ite)
    except DetectionError as e:
        return Decoded(None,'detect_failure:' + str(e),(time.perf_counter()-t0)*1e3,0)


def render_png(img:np.ndarray,path:str)->None:
    rgb=np.clip(np.round(img*255),0,255).astype(np.uint8)
    cv2.imwrite(path,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))


# Experimental dual-use frame: 108 perimeter cells carry data and remain dark
# enough to act as a continuous square-locator. Four corners remain unmodified.
# This directly tests the user's data-bearing finder-pattern hypothesis.
import binascii

def frame_coords()->np.ndarray:
    result=[]
    # avoid the first/last 5 positions on every side, leaving strong black corners
    for col in range(5,SIZE-5): result.append((0,col))
    for row in range(5,SIZE-5): result.append((row,SIZE-1))
    for col in range(SIZE-6,4,-1): result.append((SIZE-1,col))
    for row in range(SIZE-6,4,-1): result.append((row,0))
    assert len(result)==108
    return np.asarray(result,dtype=np.int32)

def dark_tetra(contrast:float=.4)->np.ndarray:
    return PALETTES['rgb4'] * contrast

def encode_joint(payload:bytes,contrast:float=.4,cell_px:int=6,extra_bytes:int=7)->np.ndarray:
    core_n=payload_size('rgb4')
    if not 1<=extra_bytes<=7: raise ValueError('extra bytes must be 1..7')
    if len(payload)!=core_n+extra_bytes:
        raise ValueError(f'joint-frame needs exactly {core_n+extra_bytes} data bytes')
    main=encode(payload[:core_n],'rgb4',cell_px=cell_px)
    appendix=payload[core_n:]
    b=binascii.crc_hqx(appendix,0xFFFF)
    frame=appendix+struct.pack('>H',b)
    bits=np.unpackbits(np.frombuffer(frame,dtype=np.uint8),bitorder='big')
    rep=(108*2)//len(bits)
    repeated=np.tile(bits,rep)
    two=repeated.reshape(-1,2)
    sym=two[:,0]*2+two[:,1]
    colors=dark_tetra(contrast)[sym]
    frame_positions=frame_coords()[:len(colors)]
    for (r,c),rgb in zip(frame_positions,colors):
        rr=(QUIET+int(r))*cell_px
        cc=(QUIET+int(c))*cell_px
        main[rr:rr+cell_px,cc:cc+cell_px]=rgb
    return main

def decode_joint(img:np.ndarray,contrast:float=.4,cell_px:int=6,extra_bytes:int=7)->Decoded:
    t0=time.perf_counter()
    try:
        grid=locate_and_calibrate(img,cell_px=cell_px,border_data=True)
        coord=cell_coords()
        llr=llr_from_colors(grid[coord[:,0],coord[:,1]],'rgb4')
        info,valid,ite=ldpc_decode(llr)
        core=extract_payload(info)
        if core is None:
            return Decoded(None,'core_failure',(time.perf_counter()-t0)*1000,ite)
        if not 1<=extra_bytes<=7: raise ValueError('extra bytes must be 1..7')
        total_bits=(extra_bytes+2)*8
        rep=(108*2)//total_bits
        coords=frame_coords()[:(rep*total_bits)//2]
        f_llr=llr_from_colors(grid[coords[:,0],coords[:,1]]/contrast,'rgb4',llr_sigma=.35)
        b=(np.sum(f_llr.reshape(rep,total_bits),axis=0)<0).astype(np.uint8)
        frame=np.packbits(b,bitorder='big').tobytes()
        extra,crc=frame[:extra_bytes],struct.unpack('>H',frame[extra_bytes:])[0]
        if binascii.crc_hqx(extra,0xFFFF)!=crc:
            return Decoded(None,'border_crc_failure',(time.perf_counter()-t0)*1000,ite)
        return Decoded(core+extra,'ok',(time.perf_counter()-t0)*1000,ite)
    except DetectionError as e:
        return Decoded(None,'detect_failure:' + str(e),(time.perf_counter()-t0)*1000,0)
