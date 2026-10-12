"""Retrieve pinned published weights/configuration only; execute no repository code."""
import hashlib, json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', ['README.md','config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','1_Pooling/config.json']),
    'finbert': ('ProsusAI/finbert', '4556d13015211d73dccd3fdd39d39232506f3e43', ['README.md','config.json','pytorch_model.bin','tokenizer_config.json','special_tokens_map.json','vocab.txt']),
}

def main():
    records=[]
    for name,(repo,revision,files) in MODELS.items():
        for f in files:
            dest=ROOT/'models'/name/f
            dest.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://huggingface.co/{repo}/resolve/{revision}/{f}'
            if not dest.exists():
                response=requests.get(url,stream=True,timeout=(30,90)); response.raise_for_status()
                temporary=dest.with_suffix(dest.suffix+'.part')
                with temporary.open('wb') as out:
                    for chunk in response.iter_content(1024*1024): out.write(chunk)
                temporary.replace(dest)
            records.append({'repo':repo,'revision':revision,'file':f,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'url':url})
            print(json.dumps(records[-1]),flush=True)
        (ROOT/'models'/'download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')

if __name__=='__main__': main()
