import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from run_representation import encode,ROOT,MODEL_IDS
def main():
    protocol=json.loads((ROOT/'probe_protocol.json').read_text(encoding='utf8'));path=ROOT/'caches'/'probes';path.mkdir(parents=True,exist_ok=True)
    notes=protocol['train_templates']+protocol['test_templates'];results={}
    for name in ('minilm','finbert'):
        vectors,_,_=encode(name,notes,path/(name+'.npz'))
        mod=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',random_state=91801).fit(vectors[:12],protocol['labels'])
        predictions=mod.predict(vectors[12:]);cosines=(vectors[12::2]*vectors[13::2]).sum(1)
        results[name]={'model_id':MODEL_IDS[name][0],'revision':MODEL_IDS[name][1],'test_accuracy':float(np.mean(predictions==protocol['test_labels'])),'test_predictions':predictions.tolist(),'opposite_pair_cosines':cosines.tolist(),'opposite_pair_cosine_median':float(np.median(cosines))}
        print(json.dumps({name:results[name]}),flush=True)
    output={'protocol_sha256':hashlib.sha256((ROOT/'probe_protocol.json').read_bytes()).hexdigest(),'protocol':protocol,'results':results}
    (ROOT/'results'/'financial_direction_probe.json').write_text(json.dumps(output,indent=2),encoding='utf8')
if __name__=='__main__':main()
