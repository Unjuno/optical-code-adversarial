"""Black-box adversarial physical-channel search and symbol-layout generator.
This is MIN-MAX design over bounded simulated optical impairments, not a GAN.
No arbitrary pixels are edited: only camera-like channel parameters vary.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
import cv2
import lab_v01 as base
import robust_code as code

# calibrated only to v01's synthetic medium/hard envelope; arbitrary experimental limits
NAMES=('blur','pixel_noise','cell_noise','jitter','rotation','mix','wb')
LOW=np.asarray([1.1,.035,.025,1.0,6.0,.020,.055])
HIGH=np.asarray([2.55,.100,.083,3.4,13.0,.068,.16])

def channel(z:np.ndarray)->base.Channel:
    vals=LOW+np.clip(z,0,1)*(HIGH-LOW)
    return base.Channel('attack',*map(float,vals))

def sample_params(rng:np.random.Generator)->np.ndarray:
    # adversary allowed a severity budget to prevent simultaneous unrealistic maxima
    z=rng.uniform(.55,1,size=7)
    if z.sum()>5.4: z*=5.4/z.sum()
    return z

def attack_loss(image:np.ndarray,payload:bytes,layout:code.Layout,extra:int):
    """Differentiation-free surrogate: decoder failure plus bitwise logistic loss."""
    d=code.decode(image,extra_bytes=extra,layout=layout)
    invalid=d.payload!=payload
    if d.reason.startswith('detection'):return 80., d.reason
    try:
        g=base.locate_and_calibrate(image)
    except base.DetectionError:return 80.,'detection'
    core=base.llr_from_colors(g[layout.core[:,0],layout.core[:,1]],'rgb4')
    core_sign=1-2*base.encode_bits(payload[:138],'rgb4').astype(np.float32)
    ex=base.llr_from_colors(g[layout.extra[:,0],layout.extra[:,1]],'rgb4',llr_sigma=.22)
    ex_sign=1-2*code.encode_aux(payload[138:]).astype(np.float32)
    # softer logistic surrogate, more weight to auxiliary data
    ca=np.logaddexp(0.,-np.clip(core_sign*core,-20,20)).mean()
    ea=np.logaddexp(0.,-np.clip(ex_sign*ex,-20,20)).mean()
    return float(6.0*int(invalid)+ca+1.5*ea),d.reason

def search(payload:bytes,layout:code.Layout,extra:int,rng:np.random.Generator,
           candidates:int=8,steps:int=1):
    """Random starts + coordinate ascent against a known encoder/decoder."""
    symbol=code.encode(payload,layout)
    camera_seed=int(rng.integers(1,2**31-1))
    best=(-1.,None,None,None)
    starts=[np.ones(7)*.72]
    starts += [sample_params(rng) for _ in range(max(1,candidates-1))]
    for z in starts:
        image=base.camera(symbol,channel(z),np.random.default_rng(camera_seed))
        loss,reason=attack_loss(image,payload,layout,extra)
        if loss>best[0]:best=(loss,z.copy(),camera_seed,reason)
    for _ in range(steps):
        z=best[1]
        for ax in rng.choice(7,size=3,replace=False):
            zz=z.copy();zz[ax]=min(1,zz[ax]+.23)
            if zz.sum()>5.4:zz*=5.4/zz.sum()
            image=base.camera(symbol,channel(zz),np.random.default_rng(camera_seed))
            loss,reason=attack_loss(image,payload,layout,extra)
            if loss>best[0]:best=(loss,zz.copy(),camera_seed,reason)
    return {'loss':float(best[0]),'z':best[1].tolist(),'seed':int(best[2]),'reason':best[3]}

def candidate_layouts()->list[code.Layout]:
    """Many structured positions for auxiliary symbols; all use identical #modules."""
    pool=code.POOL.copy()
    present={tuple(x) for x in pool}
    results=[code.LEGACY_LAYOUT]
    center=np.array([18.,18.])
    edge=np.minimum.reduce([pool[:,0],pool[:,1],36-pool[:,0],36-pool[:,1]])
    dist=np.sqrt(((pool-center)**2).sum(axis=1))
    # safety margin from frame / pilots, plus sparse placement for decorrelation
    for kind in ('central','interior_spaced','uniform_spaced','random_wide'):
        for seed in (91,187,236):
            rng=np.random.default_rng(seed)
            selected=[];remaining=np.arange(len(pool))
            if kind=='central':
                scores=dist +rng.uniform(0,1.6,size=len(pool))
                extra=pool[np.argsort(scores)[:37]]
            elif kind=='random_wide':
                weights=np.maximum(edge-1,1).astype(float)
                ids=rng.choice(len(pool),size=37,replace=False,p=weights/weights.sum())
                extra=pool[ids]
            else:
                # greedily maximize nearest-neighbour distance + distance from frame
                # avoid all extra cells concentrated in blur-correlated clusters
                idx=int(rng.integers(0,len(pool)))
                selected=[idx]
                for _ in range(36):
                    dd=((pool[:,None,:]-pool[np.array(selected)][None,:,:])**2).sum(axis=-1).min(axis=1)
                    spread=dd**.5
                    metric=spread + ((.35 if kind=='interior_spaced' else .10)*edge) +rng.uniform(0,.2,len(pool))
                    metric[selected]=-1e6
                    selected.append(int(np.argmax(metric)))
                extra=pool[np.asarray(selected)]
            rem={tuple(x) for x in extra}
            core=np.array([x for x in code.LEGACY_CORE if tuple(x) not in rem] +
                    [x for x in code.LEGACY_EXTRA if tuple(x) not in rem],dtype=np.int32)
            assert len(core)==1152 and len(extra)==37
            results.append(code.Layout(f'{kind}_{seed}',core,extra))
    return results

if __name__=='__main__':
    print('candidate layouts:',len(candidate_layouts()))
    p=np.random.default_rng(333).bytes(142)
    a=search(p,code.LEGACY_LAYOUT,4,np.random.default_rng(777),candidates=4,steps=0)
    print('attack example',a)
