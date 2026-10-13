"""Validate sealed exports, source syntax and obvious credential/path leaks."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
bad=[];verified=0;parsed=0
secret=re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
for r in csv.DictReader((ROOT/'provenance/file_manifest.csv').open()):
    p=ROOT/r['export_path']
    if not p.exists():bad.append('missing: '+r['export_path']);continue
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=r['export_sha256']:bad.append('hash mismatch: '+r['export_path'])
    verified+=1
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in p.parts for x in {'.git','__pycache__','.pytest_cache','local-artifacts','new_runs','.bob_env','.venv_generator'}):continue
    if p.suffix not in {'.py','.json','.yaml','.yml','.md','.csv','.jsonl','.log','.txt','.sh','.toml','.patch','.svg','.html'}:continue
    raw=p.read_bytes()
    if secret.search(raw):bad.append('credential pattern: '+str(p.relative_to(ROOT)))
    if re.search(rb'/home/[A-Za-z0-9_.-]+',raw):bad.append('personal path: '+str(p.relative_to(ROOT)))
    if p.suffix=='.py':
        try:ast.parse(raw.decode());parsed+=1
        except (SyntaxError,UnicodeDecodeError) as e:bad.append('syntax: '+str(p.relative_to(ROOT)))
for r in json.loads((ROOT/'experiments/registry.json').read_text()):
    for key in ('analysis_document',):
        if r.get(key) and not (ROOT/r[key]).exists():bad.append('missing registry target: '+r[key])
    for key in ('source_roots','code_entrypoints','config_files','documents'):
        for name in r.get(key,[]):
            if not (ROOT/name).exists():bad.append('missing registry target: '+name)
print(json.dumps({'verified_export_files':verified,'parsed_python':parsed,'issues':bad},indent=2))
sys.exit(bool(bad))
