"""Restore tracked or locally retained gzip evidence without overwriting files."""
import argparse,csv,gzip,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--prefix',default='');p.add_argument('--external-root',type=Path,default=ROOT.parent/'_local-evidence'/ROOT.name);a=p.parse_args()
count=0;missing=0
for r in csv.DictReader((ROOT/'provenance/compressed_tables.csv').open()):
    if not r['path'].startswith(a.prefix):continue
    archive=ROOT/r['compressed_path'] if r['storage']=='tracked' else a.external_root/r['compressed_path']
    if not archive.exists():missing+=1;continue
    dest=ROOT/r['path'];raw=gzip.decompress(archive.read_bytes())
    assert hashlib.sha256(raw).hexdigest()==r['uncompressed_sha256']
    if dest.exists():
        if dest.read_bytes()!=raw:raise SystemExit(f'refusing overwrite: {dest.relative_to(ROOT)}')
        continue
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);count+=1
print(f'Restored {count} files; {missing} optional external evidence files unavailable. Tracked summaries remain readable.')
