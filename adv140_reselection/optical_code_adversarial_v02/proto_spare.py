import sys,time,struct,binascii
sys.path.insert(0,'/mnt/data/optical_code_adversarial_v02')
import lab_v01 as l
import numpy as np
l.DETECTION_THRESHOLD=160
l.MORPH_KERNEL=3
used=set(map(tuple,l.cell_coords().tolist()))
spare=np.array([p for p in l.available_coords().tolist() if tuple(p) not in used]);print('spare',len(spare),spare[:5])
assert len(spare)==37

def encode_spare(payload:bytes,extra=2,method='rep',contrast=1.):
  m=l.encode(payload[:138],'rgb4')
  v=payload[138:]
  bits=np.unpackbits(np.frombuffer(v+struct.pack('>H',binascii.crc_hqx(v,0xFFFF)),dtype=np.uint8))
  if method=='rep':
   bits=np.tile(bits,2)
   bits=np.pad(bits,(0,74-len(bits)))
  elif method=='single':
   bits=np.pad(bits,(0,74-len(bits)))
  a=bits.reshape(-1,2)
  ind=2*a[:,0]+a[:,1]
  for (r,c),s in zip(spare,ind):
   m[(l.QUIET+r)*6:(l.QUIET+r+1)*6,(l.QUIET+c)*6:(l.QUIET+c+1)*6]=l.PALETTES['rgb4'][s]*contrast
  return m

def decode_spare(img,extra=2,contrast=1.,method='rep'):
  try:
    g=l.locate_and_calibrate(img)
    llr=l.llr_from_colors(g[l.cell_coords()[:,0],l.cell_coords()[:,1]],'rgb4')
    info,valid,ite=l.ldpc_decode(llr)
    core=l.extract_payload(info)
    if core is None:return None,'core'
    extras=l.llr_from_colors(g[spare[:,0],spare[:,1]]/contrast,'rgb4',llr_sigma=.25)
    if method=='rep':
     n=(extra+2)*8
     b=(extras[:n]+extras[n:2*n]<0).astype(np.uint8)
    else:b=(extras[:(extra+2)*8]<0).astype(np.uint8)
    frame=np.packbits(b).tobytes()
    v,crc=frame[:-2],struct.unpack('>H',frame[-2:])[0]
    if binascii.crc_hqx(v,0xFFFF)!=crc:return None,'extra'
    return core+v,'ok'
  except l.DetectionError as e:return None,'detect:'+str(e)

def test(cond='hard',n=100):
  outcomes={name:0 for name in ('base','joint','spare')}
  reasons={name:{} for name in outcomes}
  t0=time.time()
  for i in range(n):
   x=np.random.default_rng(908122+i).bytes(140)
   seed=610022+i+(100000 if cond=='hard' else 0)
   for name in outcomes:
    if name=='base':
      img=l.encode(x[:138],'rgb4'); expected=x[:138]
    elif name=='joint':
      img=l.encode_joint(x,extra_bytes=2,contrast=.44);expected=x
    else:
      img=encode_spare(x);expected=x
    y=l.camera(img,l.CHANNELS[cond],np.random.default_rng(seed))
    if name=='base':
      dec=l.decode(y,'rgb4');v=dec.payload;reason=dec.reason
    elif name=='joint':
      dec=l.decode_joint(y,extra_bytes=2,contrast=.44);v=dec.payload;reason=dec.reason
    else: v,reason=decode_spare(y)
    outcomes[name]+=int(v==expected)
    reasons[name][reason]=reasons[name].get(reason,0)+1
  print(cond,n,outcomes,reasons,'elapsed',round(time.time()-t0,2),flush=True)
for cond in ['medium','hard']:
 test(cond,80)
