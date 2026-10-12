"""Frozen modern encoder and independent train-only residual-ridge audit.

All outputs are exploratory. Original data and caches are read-only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time,platform
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
PROTOTYPE=PROJECT/'phase3_prototype'
sys.path.insert(0,str(PROTOTYPE))
sys.path.insert(0,str(PROTOTYPE/'scripts'))
from eventmamba.timeimm import load_timeimm,DATASET_SPECS,GPT2_TEXT_CACHE
from cache_timeimm_text_embeddings import read_entity
torch.set_num_threads(4)
csv.field_size_limit(64*1024*1024)
NEUTRAL='A report is available. No task-specific information is provided.'
MODEL_IDS={'minilm':('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41'),'finbert':('ProsusAI/finbert','4556d13015211d73dccd3fdd39d39232506f3e43')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def savejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def log(obj):print(json.dumps(obj,ensure_ascii=False),flush=True)
def pool_vectors(v):return np.asarray(v,dtype=np.float64).mean(0)

def prepare_dataset(name):
    spec=DATASET_SPECS[name]
    source=PROTOTYPE/'external'/'Time-IMM'/'data'/name
    data=load_timeimm(source,spec,normalization='train_only')
    entities={p.name:read_entity(p,86400.) for p in sorted((source/'processed').iterdir()) if p.is_dir()}
    samples={}; selected=defaultdict(set); used_by_split=defaultdict(set)
    for split in ('train','val','test'):
        pack=[]
        for sample in getattr(data,split+'_windows'):
            info=data.window_info[sample.sample_id]; e=entities[info.entity]
            tt=np.array(e['times']); indices=np.flatnonzero((tt>=info.start_time)&(tt<info.start_time+spec.history))
            chosen=indices[-4:]
            selected[info.entity].update(chosen.tolist()); used_by_split[(info.entity,split)].update(chosen.tolist())
            d=len(data.feature_names); rows=[]; lasts=[]
            for m in range(d):
                events=[ev for ev in sample.events if ev.modality==m]
                values=np.array([float(ev.value[0]) for ev in events]); age=(spec.history-events[-1].event_time)/spec.history if events else 1.
                last=float(values[-1]) if len(values) else 0.;lasts.append(last)
                rows.extend([last,float(values.mean()) if len(values) else 0.,float(values.std()) if len(values) else 0.,float(np.log1p(len(values))),float(age)])
            numeric=np.array(rows)
            structure=np.array([np.log1p(len(indices)),(spec.history-(tt[indices[-1]]-info.start_time))/spec.history])
            pack.append({'entity':info.entity,'split':split,'window':info.window_index,'start':info.start_time,'sample_id':sample.sample_id,'selected':chosen.tolist(),'all_notes':len(indices),'numeric':numeric,'structure':structure,'last':np.array(lasts),'horizons':sample.query_horizons.numpy()/spec.prediction,'target':sample.targets.numpy(),'mask':sample.target_masks.numpy().astype(bool)})
        samples[split]=pack
    mappings={}
    for i,((entity,split),used) in enumerate(sorted(used_by_split.items())):
        ids=np.array(sorted(used)); rng=np.random.default_rng(91801+i); shift=int(rng.integers(1,len(ids))) if len(ids)>1 else 0
        mappings[(entity,split)]={int(a):int(b) for a,b in zip(ids,np.roll(ids,shift))}
    passport={'dataset':name,'features':list(data.feature_names),'source_commit':'e934cef23d526817dcf0d1a66c6f82274c96c04b','entities':list(entities),'window_counts':data.window_counts,'dropped_no_text':data.dropped_no_text,'selected_text_counts':{e:len(x) for e,x in selected.items()},'full_text_counts':{e:len(v['notes']) for e,v in entities.items()},'source_sha256':{e:{f:sha(source/'processed'/e/f) for f in ('text.csv','time_series.csv')} for e in entities},'all_windows_retained':True,'latest4_cap':True}
    return data,entities,samples,selected,mappings,passport

def encode(modelname, notes, cachepath):
    notes_digest=hashlib.sha256(json.dumps(notes,ensure_ascii=False).encode('utf8')).hexdigest()
    if cachepath.exists():
        z=np.load(cachepath)
        if str(z['notes_sha256'])!=notes_digest:raise ValueError('Selected text cache fingerprint mismatch')
        return z['embeddings'],z['lengths'],z['neutral']
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models'/modelname,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(ROOT/'models'/modelname,local_files_only=True,trust_remote_code=False).eval()
    for p in model.parameters():p.requires_grad_(False)
    fullnotes=list(notes)+[NEUTRAL]
    lengths=np.array([len(x) for x in tokenizer(fullnotes,truncation=False,padding=False)['input_ids']])
    output=[];started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(fullnotes),16):
            batch=tokenizer(fullnotes[start:start+16],padding=True,truncation=True,max_length=256,return_tensors='pt')
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            v=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1);v=torch.nn.functional.normalize(v,dim=-1)
            output.append(v.numpy())
            if start%160==0:log({'encoder':modelname,'encoded':start+len(v),'total':len(fullnotes),'seconds':round(time.perf_counter()-started,1)})
    z=np.concatenate(output)
    np.savez_compressed(cachepath,embeddings=z[:-1],lengths=lengths[:-1],neutral=z[-1],notes_sha256=notes_digest)
    return z[:-1],lengths[:-1],z[-1]

def geometry(vectors,trainmean,replacements):
    x=np.asarray(vectors,dtype=np.float64);y=np.asarray(replacements,dtype=np.float64)
    def cosine(a,b):return np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    centered_x=x-trainmean;centered_y=y-trainmean
    return {'n':len(x),'replacement_raw_cosine_median':float(np.median(cosine(x,y))),'replacement_train_centered_cosine_median':float(np.median(cosine(centered_x,centered_y))),'centroid_norm_ratio':float(np.linalg.norm(x.mean(0))/np.mean(np.linalg.norm(x,axis=1))),'exact_vector_unchanged_fraction':float(np.all(x==y,axis=1).mean())}

def design(samples,vectors,mappings,variant,pca=None):
    result={}
    all_texts={}
    if vectors is not None:
        dimension=len(next(iter(next(iter(vectors.values())).values())))
        for split,pack in samples.items():
            all_texts[split]=np.stack([pool_vectors([vectors[s['entity']][mappings[(s['entity'],split)][i] if variant=='semantic_derangement' else i] for i in s['selected']]) if variant not in ('zero','neutral') else (np.zeros(dimension) if variant=='zero' else vectors['_neutral']) for s in pack])
        if pca is None:
            pca=PCA(n_components=min(32,len(all_texts['train'])-1,all_texts['train'].shape[1]),svd_solver='full').fit(all_texts['train'])
    for split,pack in samples.items():
        xx=[];yy=[];mm=[];baseline=[];records=[]
        tx=pca.transform(all_texts[split]) if vectors is not None else None
        for k,s in enumerate(pack):
            base=s['numeric']
            if variant!='numeric_only':base=np.r_[base,s['structure']]
            if tx is not None:base=np.r_[base,tx[k]]
            for j,h in enumerate(s['horizons']):
                xx.append(np.r_[base,h,h*h]); yy.append(s['target'][j]);mm.append(s['mask'][j]);baseline.append(s['last'])
                records.append((s['entity'],s['window'],s['start'],float(h),s['sample_id']))
        result[split]=(np.stack(xx),np.stack(yy),np.stack(mm),np.stack(baseline),records)
    return result,pca

def metric(pred,y,mask,records):
    err=pred-y; count=int(mask.sum()); by=defaultdict(lambda:[0.,0.,0])
    for row,e,m in zip(records,err,mask):
        by[row[0]][0]+=float((e[m]**2).sum());by[row[0]][1]+=float(np.abs(e[m]).sum());by[row[0]][2]+=int(m.sum())
    entities={e:{'mse':a/n,'mae':b/n,'count':n} for e,(a,b,n) in by.items()}
    return {'mse':float((err[mask]**2).mean()),'mae':float(np.abs(err[mask]).mean()),'observations':count,'macro_entity_mse':float(np.mean([v['mse'] for v in entities.values()])),'entities':entities}

def train(designset):
    x,y,mask,base,_=designset['train'];scaler=StandardScaler().fit(x);tx=scaler.transform(x);vx,vy,vm,vb,vr=designset['val'];vx=scaler.transform(vx)
    candidates=[]
    for alpha in (1.,10.,100.,1000.):
        models=[Ridge(alpha=alpha).fit(tx[mask[:,j]],(y-base)[mask[:,j],j]) for j in range(y.shape[1])]
        vp=vb+np.stack([mod.predict(vx) for mod in models],axis=1)
        candidates.append((metric(vp,vy,vm,vr)['mse'],alpha,models))
    _,alpha,models=min(candidates,key=lambda z:z[0]);return scaler,models,alpha,[{'alpha':a,'val_mse':m} for m,a,_ in candidates]

def predict(fit,pack):
    scaler,models,*_=fit;x,y,mask,base,records=pack
    return base+np.stack([mod.predict(scaler.transform(x)) for mod in models],axis=1)

def write_records(path,pack,pred,condition,representation):
    x,y,mask,base,records=pack
    with path.open('w',encoding='utf8',newline='') as handle:
        out=csv.writer(handle);out.writerow(['entity','window','start_time_days','normalized_forecast_horizon','sample_id','feature','target','prediction','squared_error','condition','representation'])
        for i,record in enumerate(records):
            for j in np.flatnonzero(mask[i]):out.writerow([*record,int(j),float(y[i,j]),float(pred[i,j]),float((pred[i,j]-y[i,j])**2),condition,representation])

def run(dataset,representations):
    data,entities,samples,selected,mappings,passport=prepare_dataset(dataset)
    cache=ROOT/'caches'/dataset;cache.mkdir(parents=True,exist_ok=True)
    target=ROOT/'results'/dataset;target.mkdir(parents=True,exist_ok=True)
    savejson(target/'material_passport.json',passport)
    allslots=[(e,i) for e in sorted(selected) for i in sorted(selected[e])]
    texts=[entities[e]['notes'][i] for e,i in allslots]
    savejson(cache/'selected_rows.json',[{'entity':e,'sorted_text_index':i,'source_row':entities[e]['row_indices'][i],'time_days':entities[e]['times'][i],'text_sha256':hashlib.sha256(entities[e]['notes'][i].encode()).hexdigest()} for e,i in allslots])
    summary={'protocol_sha256':sha(ROOT/'protocol.json'),'script_sha256':sha(__file__),'material':passport,'results':[],'representation_audit':{},'runtime':{'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform(),'threads':4,'device':'cpu'}}
    for condition in ('numeric_only','event_structure'):
        dd,_=design(samples,None,mappings,condition);fit=train(dd);pr=predict(fit,dd['test'])
        summary['results'].append({'representation':'none','training_condition':condition,'evaluation_condition':condition,'selected_alpha':fit[2],'validation_grid':fit[3],'val':metric(predict(fit,dd['val']),dd['val'][1],dd['val'][2],dd['val'][4]),'test':metric(pr,dd['test'][1],dd['test'][2],dd['test'][4])})
        write_records(target/(condition+'.csv'),dd['test'],pr,condition,'none')
    for reprname in representations:
        vectors=defaultdict(dict)
        if reprname=='gpt2':
            for e in selected:
                g=torch.load(PROTOTYPE/'external'/'Time-IMM'/'data'/dataset/'processed'/e/GPT2_TEXT_CACHE,weights_only=True,map_location='cpu')
                for i in selected[e]:vectors[e][i]=g['embeddings'][i].float().numpy()
            lengths=None
        else:
            z,lengths,neutral=encode(reprname,texts,cache/(reprname+'.npz'))
            for (e,i),v in zip(allslots,z):vectors[e][i]=v
            vectors['_neutral']=neutral
        audits={};trainvec=np.stack([vectors[s['entity']][i] for s in samples['train'] for i in s['selected']]);trainmean=trainvec.mean(0)
        for split,pack in samples.items():
            ids=[(s['entity'],i) for s in pack for i in s['selected']]
            original=np.stack([vectors[e][i] for e,i in ids]);replacement=np.stack([vectors[e][mappings[(e,split)][i]] for e,i in ids])
            audits[split]=geometry(original,trainmean,replacement)
            audits[split]['raw_text_unchanged_fraction']=float(np.mean([entities[e]['notes'][i]==entities[e]['notes'][mappings[(e,split)][i]] for e,i in ids]))
            audits[split]['unique_text_fraction']=len({entities[e]['notes'][i] for e,i in ids})/len(ids)
        audits['tokens']={'selected_notes':len(texts),'max_length':256,'truncated_fraction':float((lengths>256).mean()),'median_length':float(np.median(lengths)),'max_length_observed':int(lengths.max())} if lengths is not None else {'cache':GPT2_TEXT_CACHE,'no_new_encoding':True}
        summary['representation_audit'][reprname]=audits
        for training_condition in ('aligned','semantic_derangement'):
            dd,pca=design(samples,vectors,mappings,training_condition);fit=train(dd)
            checkpoint=target/f'{reprname}_{training_condition}_checkpoint.npz'
            np.savez_compressed(checkpoint,scaler_mean=fit[0].mean_,scaler_scale=fit[0].scale_,coef=np.stack([m.coef_ for m in fit[1]]),intercept=np.array([m.intercept_ for m in fit[1]]),pca_mean=pca.mean_,pca_components=pca.components_)
            for eval_condition in (['aligned','semantic_derangement','zero']+(['neutral'] if reprname!='gpt2' else [])) if training_condition=='aligned' else ['semantic_derangement']:
                ev,_=design(samples,vectors,mappings,eval_condition,pca=pca);pr=predict(fit,ev['test'])
                record={'representation':reprname,'training_condition':training_condition,'evaluation_condition':eval_condition,'checkpoint_sha256':sha(checkpoint),'selected_alpha':fit[2],'validation_grid':fit[3],'val':metric(predict(fit,ev['val']),ev['val'][1],ev['val'][2],ev['val'][4]),'test':metric(pr,ev['test'][1],ev['test'][2],ev['test'][4])}
                summary['results'].append(record);write_records(target/f'{reprname}_trained_{training_condition}_eval_{eval_condition}.csv',ev['test'],pr,eval_condition,reprname)
                log({'dataset':dataset,'representation':reprname,'training':training_condition,'evaluation':eval_condition,'test_mse':record['test']['mse']})
                savejson(target/'results.json',summary)
    savejson(target/'results.json',summary)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['FNSPID','EPA-Air']);parser.add_argument('--representations',nargs='+',default=['gpt2','minilm','finbert']);args=parser.parse_args()
    for name in args.datasets:run(name,args.representations)
if __name__=='__main__':main()
