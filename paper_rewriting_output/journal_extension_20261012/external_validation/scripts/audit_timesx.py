"""Inspect public repository metadata/docs without executing downloaded code."""
import json,re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def main():
    r=requests.get('https://api.github.com/search/repositories',params={'q':'TimesX time series'},timeout=45);r.raise_for_status();payload=r.json()
    result=[]
    for x in payload.get('items',[])[:8]:result.append({'full_name':x['full_name'],'url':x['html_url'],'description':x['description'],'default_branch':x['default_branch'],'license':x.get('license')})
    (ROOT/'timesx_repository_search.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
    html=(ROOT/'timesx_arxiv.html').read_text(encoding='utf8')
    links=[x for x in re.findall(r'href="([^"]+)"',html) if x.startswith('http')]
    (ROOT/'timesx_outlinks.json').write_text(json.dumps(links,indent=2),encoding='utf8');print(json.dumps(links[:70]),flush=True)
if __name__=='__main__':main()
