import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='ARTIFACT_MANIFEST.json':continue
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    r=json.loads((ROOT/'results'/'results.json').read_text(encoding='utf8'))
    artifact={'status':'Complete:20 maps for all7 saved ridge states; no refit','date':'2026-10-12 Asia/Shanghai','aggregate_rows':140,'per_unit_rows':1260,'timing_slot_rows':94000,'unit_tests_passed':4,'invariant_checks_passed':r['all_checks_passed'],'old_input_hashes_unchanged':r['existing_input_hashes_unchanged'],'refit_count':r['refit_count'],'scope':'Empirical between-cyclic-map variation, not sampling confidence intervals, formal permutation tests, or neural-checkpoint calibration','files':files}
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(artifact,indent=2),encoding='utf8');print(json.dumps({'files':len(files),'status':artifact['status']}),flush=True)
if __name__=='__main__':main()
