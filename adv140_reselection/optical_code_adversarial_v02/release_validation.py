"""Release integrity checks and demonstrator. Run: python release_validation.py"""
from pathlib import Path
import hashlib,json,csv
import numpy as np
import cv2
import lab_v01 as old
import robust_code as robust
import adversary
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)


def check():
    assert old.NCELLS==1152 and len(old.available_coords())==1189
    assert len(robust.LEGACY_EXTRA)==37
    assert len(robust.core_graph_neighbors())==1152
    paths=[]
    layouts=[robust.LEGACY_LAYOUT]
    a=np.load(OUT/'optimized_layout.npz')
    layouts.append(robust.Layout(str(a['name']),a['core'],a['extra']))
    for lay in layouts:
        core={tuple(p) for p in lay.core}
        extra={tuple(p) for p in lay.extra}
        assert len(core)==1152 and len(extra)==37
        assert len(core & extra)==0
        assert core | extra =={tuple(p) for p in old.available_coords()}
        for graph in ('legacy','adversarial'):
            for extra_bytes in (2,3,4):
                dat=np.random.default_rng(9912+extra_bytes).bytes(138+extra_bytes)
                pic=robust.encode(dat,layout=lay,graph=graph)
                d=robust.decode(pic,extra_bytes,layout=lay,graph=graph)
                assert d.payload==dat,(lay.name,graph,extra_bytes,d.reason)
    # black contour at the frame is exactly fixed across all payloads for new formats
    imgs=[]
    for seed in (10,19,35):
        p=np.random.default_rng(seed).bytes(140)
        imgs.append(robust.encode(p,graph='adversarial'))
    top=(old.QUIET*6,(old.QUIET+old.SIZE-1)*6)
    # top-left corner is invariant black
    for pic in imgs:
        assert (pic[old.QUIET*6:old.QUIET*6+6,old.QUIET*6:old.QUIET*6+6]==0).all()
    assert np.allclose(imgs[0][old.QUIET*6, (old.QUIET+10)*6], imgs[1][old.QUIET*6,(old.QUIET+10)*6])
    for z in [adversary.sample_params(np.random.default_rng(i)) for i in range(25)]:
        assert np.all(z>=0) and np.all(z<=1.0001) and z.sum()<=5.4001
    # physical color 4-level palette and auxiliary bit budget
    for extra in (2,3,4):
        p=np.random.default_rng(extra).bytes(extra)
        bits=robust.encode_aux(p)
        assert len(bits)==74
        k=(extra+2)*8
        assert len(bits[:k])==k and len(bits[k:])==74-k
    # full PNG round-trip from random user-like opaque bytes
    raw=np.random.default_rng(20261115).bytes(140)
    orig=robust.encode(raw,graph='adversarial')
    img_path=OUT/'example_robust140.png'
    old.render_png(orig,str(img_path))
    im=cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    out=robust.decode(im,2,graph='adversarial')
    assert out.payload==raw,out.reason
    (OUT/'example_payload_140.bin').write_bytes(raw)
    (OUT/'example_recovered_140.bin').write_bytes(out.payload)
    # show one reproducible simulated observation without requiring a successful hard scan
    cam=old.camera(orig,old.CHANNELS['medium'],np.random.default_rng(190101))
    out2=robust.decode(cam,2,graph='adversarial')
    assert out2.payload==raw,out2.reason
    old.render_png(cam,str(OUT/'example_robust140_captured.png'))
    checksum=hashlib.sha256(raw).hexdigest()
    (OUT/'example_sha256.json').write_text(json.dumps({'capacity_bytes':140,'sha256_original':checksum,
                     'sha256_reconstructed':hashlib.sha256(out.payload).hexdigest(),
                     'decode_png_match':out.payload==raw,'decode_camera_match':out2.payload==raw},indent=2))
    # 142 bytes high-capacity image round-trip
    q=np.random.default_rng(20261116).bytes(142)
    im142=robust.encode(q,layout=layouts[1],graph='adversarial')
    old.render_png(im142,str(OUT/'example_capacity142.png'))
    saved=cv2.cvtColor(cv2.imread(str(OUT/'example_capacity142.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    assert robust.decode(saved,4,layout=layouts[1],graph='adversarial').payload==q
    print('RELEASE TESTS PASS. PNG SHA256:',checksum)

if __name__=='__main__':check()
