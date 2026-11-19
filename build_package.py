"""Prepare reproducible package, sha256 manifest, run code syntax and archive integrity checks."""
from pathlib import Path
import hashlib,json,zipfile,platform,subprocess,sys
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
files=[]
for path in ROOT.rglob('*'):
    if path.is_file() and '__pycache__' not in str(path) and path.name not in ('manifest.json','build_package.py'):
        files.append(path)
manifest={
  'project':'Next-generation 2D optical code adversarial simulation v0.2',
  'created_at':'2026-11-15',
  'scope':'synthetic optical channel, exact byte recovery',
  'model_data_location':'optical symbols only; no external model storage',
  'notes':'Fixed design candidates; independent held-out random seeds. Nonstandard experimental code.',
  'python':platform.python_version(),
  'file_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
for script in ROOT.glob('*.py'):
    source=script.read_text()
    compile(source,str(script),'exec')
archive=Path('/mnt/data/optical_code_adversarial_v02.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for file in ROOT.rglob('*'):
        if file.is_file() and '__pycache__' not in str(file):
            z.write(file,arcname=str(Path(ROOT.name)/file.relative_to(ROOT)))
with zipfile.ZipFile(archive,'r') as z:
    err=z.testzip()
    assert err is None,f'corrupt zip entry {err}'
print('ZIP VERIFIED',str(archive),archive.stat().st_size,'bytes,',len(manifest['file_hashes']),'file hashes')
print('ZIP_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
