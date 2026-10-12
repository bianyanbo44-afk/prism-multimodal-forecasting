import json,hashlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[2]
DATA=PROJECT/'phase3_prototype'/'data'/'fidel_ts'/'Canada_photovoltaics_plants'
def main():
    metadata=[]
    for p in sorted((DATA/'.cache'/'huggingface'/'download').rglob('*.metadata')):
        lines=p.read_text(encoding='utf8').splitlines()
        rel=p.relative_to(DATA/'.cache'/'huggingface'/'download').as_posix().removesuffix('.metadata')
        source=DATA/rel
        if len(lines)>=2 and source.is_file():
            digest=hashlib.sha256(source.read_bytes()).hexdigest();metadata.append({'file':rel,'revision':lines[0],'etag':lines[1],'sha256':digest,'etag_matches_local_sha256':lines[1]==digest})
    revisions=sorted({x['revision'] for x in metadata})
    remote={}
    for revision in revisions:
        url=f'https://huggingface.co/api/datasets/VEWOXIC/Canada_photovoltaics_plants/revision/{revision}'
        r=requests.get(url,timeout=45);remote[revision]={'url':url,'status':r.status_code,'metadata':r.json() if r.status_code==200 else None}
    result={'huggingface_dataset':'VEWOXIC/Canada_photovoltaics_plants','local_download_metadata':metadata,'revisions':revisions,'remote_revision_check':remote,'code_execution':'Published source repository not executed by the independent residual ridge implementation'}
    (ROOT/'fidel_dataset_provenance.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'files':len(metadata),'revisions':revisions,'licenses':{k:v['metadata'].get('cardData',{}).get('license') if v['metadata'] else None for k,v in remote.items()}}),flush=True)
if __name__=='__main__':main()
