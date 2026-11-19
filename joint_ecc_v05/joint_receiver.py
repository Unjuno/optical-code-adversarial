"""Core LDPC-18 feedback via auxiliary 37 symbol patch, without core/aux data sharing.

The auxiliary decoded 2 B + CRC establish transmitted colors for 37 auxiliary
cells. These act as extra calibration pilots, estimating channel affine/palette
parameters and per-class noise from the *received image*; main bits remain
independent of the auxiliary bits. 18-iteration core NMS remains unchanged.

Candidates are selected only on development seeds. In final validation the
selected receiver parameters are loaded from FROZEN_CONFIG.json.
"""
from __future__ import annotations
import sys,time,binascii,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import aux42_lib as a
base=a.base
robust=a.robust
PALETTE=base.PALETTES['rgb4']


def load_rows():
    a.freeze_settings()
    rows=np.load(ROOT/'source/selected_aux42_rows.npy')
    digest=hashlib.sha256((ROOT/'source/selected_aux42_rows.npy').read_bytes()).hexdigest()
    assert digest=='4d3f700cda39b60b4e1d2aee775f78ddd842a88a33718b98d55c7e5370a5e9bd'
    assert robust.CORE_MAX_ITER==18
    return a.validate_rows(rows)


def get_context(image,rows):
    """Single geometry and color calibration; no secret data used in receiver."""
    try: grid=base.locate_and_calibrate(image)
    except base.DetectionError as e:
        return {'detection':str(e)}
    cobs=grid[robust.LEGACY_CORE[:,0],robust.LEGACY_CORE[:,1]]
    aobs=grid[robust.LEGACY_EXTRA[:,0],robust.LEGACY_EXTRA[:,1]]
    cllr=base.llr_from_colors(cobs,'rgb4')
    alllr=base.llr_from_colors(aobs,'rgb4',llr_sigma=.22)
    # Independent auxiliary first, to allow valid pilots even if main parity fails.
    aux,areason,auxit=a.decode_aux_rows(alllr,rows)
    data,checks,coreit=robust.decode_core(cllr,'adversarial')
    core=base.extract_payload(data)
    breason=('ok' if core is not None and aux is not None else
             'core_crc' if core is None and checks else
             'core_parity' if core is None else 'aux_crc')
    if core is not None and aux is not None:
        return {'reason':breason,'base_output':core+aux,'core_iter':coreit,'aux_iter':auxit}
    return {'reason':breason,'base_output':None,'core':core,'aux':aux,'cobs':cobs,'aobs':aobs,
            'llr_core':cllr,'llr_aux':alllr,'core_iter':coreit,'aux_iter':auxit,
            'aux_reason':areason}


def aux_rgb(aux:bytes,rows):
    ebits=a.encode_aux_rows(aux,rows).reshape((-1,2))
    ids=ebits[:,0]*2+ebits[:,1]
    return PALETTE[ids]


def calibrate(aux_obs:np.ndarray, true_rgb:np.ndarray, ridge:float=5.0, method='affine'):
    if method=='mean':
        counts=np.array([(true_rgb==c).all(axis=1).sum() for c in PALETTE])
        means=[]
        for c in PALETTE:
            mask=(true_rgb==c).all(axis=1)
            means.append((aux_obs[mask].sum(axis=0)+ridge*c)/(mask.sum()+ridge))
        return np.asarray(means,dtype=np.float32)
    # ridge regularized around identity affine transform (xRGB,1) @ W = yRGB
    X=np.concatenate((true_rgb,np.ones((len(true_rgb),1),dtype=np.float32)),axis=1)
    W0=np.concatenate((np.eye(3,dtype=np.float32),np.zeros((1,3),dtype=np.float32)),axis=0)
    regular=np.diag([ridge,ridge,ridge,ridge])
    W=np.linalg.solve(X.T@X+regular,X.T@aux_obs+regular@W0)
    P=np.concatenate((PALETTE,np.ones((4,1),dtype=np.float32)),axis=1)
    return (P@W).astype(np.float32)


def palette_llr(samples:np.ndarray, means:np.ndarray, sigma=.19, cov='identity',
                aux_obs=None,aux_ideal=None, variance_shrink:float=0.5):
    diff=samples[:,None,:]-means[None,:,:]
    if cov=='diag' and aux_obs is not None:
        # learn channel reliability, regularized with minimum variance floor
        true_means_idx=np.argmin(((aux_ideal[:,None,:]-PALETTE[None,:,:])**2).sum(axis=2),axis=1)
        resid=aux_obs-means[true_means_idx]
        v=variance_shrink*sigma**2+(1-variance_shrink)*(resid**2).mean(axis=0)
        v=np.maximum(v,0.01**2)
        dist=((diff**2)/v[None,None,:]).sum(axis=2)
    else:
        dist=(diff**2).sum(axis=2)/(sigma**2)
    scores=-.5*dist
    bit=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)
    col=[logsumexp(scores[:,bit[:,j]==0],axis=1)-logsumexp(scores[:,bit[:,j]==1],axis=1) for j in range(2)]
    return np.clip(np.stack(col,axis=1).reshape(-1),-20,20).astype(np.float32)


def llr_prior(ctx,rows,config):
    """Construct new main LLR from auxiliary-recovered known color references."""
    if ctx.get('aux') is None or 'cobs' not in ctx:return None
    known=aux_rgb(ctx['aux'],rows)
    observed=ctx['aobs']
    means=calibrate(observed,known,ridge=config['ridge'],method=config['method'])
    llr=palette_llr(ctx['cobs'],means,sigma=config['sigma'],cov=config['cov'],
                    aux_obs=observed,aux_ideal=known,variance_shrink=config['variance_shrink'])
    # Confidence-weighted prior; preserves existing channel estimates.
    prior_strength=config['prior_strength']
    out=(1-prior_strength)*ctx['llr_core']+prior_strength*llr
    return np.clip(out,-25,25).astype(np.float32)


def postprocess_context(ctx,rows,config):
    """Original successes are never replaced. CRC-check all new candidates."""
    if 'detection' in ctx:
        return {'reason':'detection:'+ctx['detection'],'payload':None,'rescued':False,'attempted':False}
    if ctx.get('base_output') is not None:
        return {'reason':'ok','payload':ctx['base_output'],'rescued':False,'attempted':False}
    if ctx.get('aux') is None or ctx.get('core') is not None:
        return {'reason':ctx['reason'],'payload':None,'rescued':False,'attempted':False}
    for item in config['calibrations']:
        llr=llr_prior(ctx,rows,item)
        decoded,check,it=robust.decode_core(llr,'adversarial',iterations=18,alpha=.82)
        core=base.extract_payload(decoded)
        if core is not None:
            return {'reason':'ok','payload':core+ctx['aux'],'rescued':True,'attempted':True,'trial':item['name'],'core_iter':it}
    return {'reason':ctx['reason'],'payload':None,'rescued':False,'attempted':True}


def run_one(payload,seed_camera,rows,config):
    t0=time.perf_counter()
    symbol=a.encode140(payload,rows)
    image=base.camera(symbol,base.CHANNELS['hard'],np.random.default_rng(seed_camera))
    tdecode=time.perf_counter()
    ctx=get_context(image,rows)
    result=postprocess_context(ctx,rows,config)
    result['decode_elapsed_ms']=(time.perf_counter()-tdecode)*1000
    result['elapsed_ms']=(time.perf_counter()-t0)*1000
    result['base_reason']=ctx.get('reason', 'detection:'+ctx.get('detection',''))
    return result

if __name__=='__main__':
    rows=load_rows()
    p=np.random.default_rng(551100001).bytes(142)[:140]
    print(run_one(p,662100001,rows,{'calibrations':[]}),flush=True)
