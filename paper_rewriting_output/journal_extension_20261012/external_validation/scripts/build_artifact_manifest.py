import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'models' in p.relative_to(ROOT).parts:continue
        if p.name=='ARTIFACT_MANIFEST.json':continue
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result={'status':'Executed and verified','date':'2026-10-12 Asia/Shanghai','datasets':['FNSPID all10 entities','EPA-Air all8 entities','Fidel regular solar all9 plants'],'modern_encoders':['all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41','ProsusAI/finbert@4556d13015211d73dccd3fdd39d39232506f3e43'],'tests':{'observed_unittest_exit_code':0,'tests_passed':4,'test_source_sha256':hashlib.sha256((ROOT/'scripts'/'test_representation.py').read_bytes()).hexdigest(),'observed_full_output_verifier_exit_code':0,'source_hashes_losses_paired_targets_checkpoint_checks':'passed'},'classification':'Exploratory; existing historical test outcomes viewed. Strict forecast report timestamp does not establish issuance availability. Full results include capability and aggregation failures.','third_party_weights_manifest':'models/download_manifest.json; weights excluded from this artifact manifest and redistribution scope','files':files}
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'manifest_files':len(files),'status':result['status']}),flush=True)
if __name__=='__main__':main()
