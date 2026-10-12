import requests,re,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
csv.field_size_limit(64*1024*1024)
def main():
    response=requests.get('https://proceedings.mlr.press/v306/liu26go.html',timeout=45)
    (ROOT/'timesx_pmlr.html').write_text(response.text,encoding='utf8')
    print(re.findall(r'https?://[^\s<>\"]*github[^\s<>\"]*',response.text))
    for dataset in ('FNSPID','EPA-Air'):
        print(dataset,[(p.name,len(list(csv.DictReader((p/'text.csv').open(encoding='utf-8-sig',newline=''))))) for p in (PROJECT/'phase3_prototype'/'external'/'Time-IMM'/'data'/dataset/'processed').iterdir() if p.is_dir()])
if __name__=='__main__':main()
