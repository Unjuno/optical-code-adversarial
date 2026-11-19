"""Offline multi-symbol exact-file delivery demonstration (no networks/model). 
Robust 140-byte code is a transport packet: 12B header +128B content.
Special manifest symbol index=65535 carries SHA256 and length.
Missing chunks lead to explicit failure, never hallucinated reconstruction.
"""
from pathlib import Path
import hashlib,struct,math
import numpy as np
import cv2
import lab_v01 as old
import robust_code as code

FRAME=140;HEADER=12;BODY=128;MAGIC=b'OC';MANIFEST=65535

def pack(file:bytes)->list[bytes]:
    file_id=hashlib.sha256(file).digest()[:4]
    count=max(1,math.ceil(len(file)/BODY))
    if count>65535:raise ValueError('file too large for this demo framing')
    def frame(i:int,data:bytes):
        head=struct.pack('>2s4sHHH',MAGIC,file_id,i,count,len(data))
        return head+data.ljust(BODY,b'\0')
    meta=hashlib.sha256(file).digest()+struct.pack('>Q',len(file))
    return [frame(MANIFEST,meta)]+[frame(i,file[i*BODY:(i+1)*BODY]) for i in range(count)]

def unpack(frames:list[bytes])->bytes:
    chunks={};meta=None;fid=None;n=None
    for frame in frames:
        if len(frame)!=FRAME:raise ValueError('frame size mismatch')
        magic,session,idx,count,length=struct.unpack('>2s4sHHH',frame[:HEADER])
        if magic!=MAGIC or length>BODY:raise ValueError('bad packet header')
        if fid is not None and (fid!=session or n!=count):raise ValueError('mixed sessions')
        fid=session;n=count
        if idx==MANIFEST:
            if length!=40:raise ValueError('bad manifest length')
            meta=frame[HEADER:HEADER+length]
        else:
            if idx>=count:raise ValueError('packet index out of range')
            data=frame[HEADER:HEADER+length]
            if idx in chunks and chunks[idx]!=data:raise ValueError('conflicting packet')
            chunks[idx]=data
    if meta is None:raise ValueError('manifest missing')
    missing=set(range(n))-set(chunks)
    if missing:raise ValueError('missing packet indexes '+str(sorted(missing)))
    orig=b''.join(chunks[i] for i in range(n))
    expected=struct.unpack('>Q',meta[32:])[0]
    if len(orig)!=expected or hashlib.sha256(orig).digest()!=meta[:32]:raise ValueError('SHA256 verification failed')
    return orig

def demo():
    folder=Path(__file__).resolve().parent/'results'/'offline_demo';folder.mkdir(parents=True,exist_ok=True)
    data=np.random.default_rng(17777).bytes(2048)
    frames=pack(data)
    for idx,b in enumerate(frames):
        old.render_png(code.encode(b,graph='adversarial'),str(folder/f'{idx:03}.png'))
    # shuffled reading order; each symbol must decode from its own raw 270x270 PNG
    rng=np.random.default_rng(90909)
    recovered=[]
    for i in rng.permutation(len(frames)):
        img=cv2.cvtColor(cv2.imread(str(folder/f'{i:03}.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        d=code.decode(img,2,graph='adversarial')
        if d.payload is None:raise RuntimeError(f'frame {i} failed: {d.reason}')
        recovered.append(d.payload)
    final=unpack(recovered)
    assert final==data
    (folder/'original.bin').write_bytes(data)
    (folder/'recovered.bin').write_bytes(final)
    (folder/'hash.txt').write_text(hashlib.sha256(data).hexdigest()+'\n')
    print(f'OFFLINE FILE DEMO PASS {len(data)} bytes, {len(frames)} optically encoded images, SHA256={hashlib.sha256(data).hexdigest()}')
    return final
if __name__=='__main__':demo()
