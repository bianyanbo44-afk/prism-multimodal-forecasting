"""Independent CPU ridge on regular Fidel solar data, historical reports only."""
from __future__ import annotations
import csv,hashlib,json,sys,time
from pathlib import Path
from collections import defaultdict
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[2]
DATA=PROJECT/'phase3_prototype'/'data'/'fidel_ts'/'Canada_photovoltaics_plants'
OUT=ROOT/'results'/'Fidel';OUT.mkdir(parents=True,exist_ok=True)
SPLITS={'train':('2018-01-01','2021-01-01'),'val':('2021-01-01','2022-01-01'),'test':('2022-01-01','2023-01-01')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def prepare():
    reports={};files=[]
    for y in range(2018,2023):
        path=DATA/'weather'/'calgary'/'report_embedding'/'formal_report'/f'fast_general_formal_embeddings_{y}.pkl'
        reports.update(joblib.load(path));files.append(path)
    keys=sorted(reports);times=pd.to_datetime(keys,format='%Y%m%d%H%M').to_numpy(dtype='datetime64[ns]')
    original=np.stack([reports[k] for k in keys]).astype(np.float64)
    weather=original.mean(1)
    mappings=np.arange(len(times))
    for k,(split,(start,end)) in enumerate(SPLITS.items()):
        ii=np.flatnonzero((times>=np.datetime64(start))&(times<np.datetime64(end)))
        shift=int(np.random.default_rng(91801+k).integers(1,len(ii)))
        mappings[ii]=np.roll(ii,shift)
    samples={k:[] for k in SPLITS};scales={};plants=[]
    for path in sorted((DATA/'time_series').glob('id_*.parquet')):
        files.append(path);plant=path.stem.removeprefix('id_');plants.append(plant)
        frame=pd.read_parquet(path).sort_values('date');tt=pd.to_datetime(frame['date']);raw=frame['kWh'].to_numpy(dtype=np.float64)
        trainmask=(tt>=pd.Timestamp('2018-01-01'))&(tt<pd.Timestamp('2021-01-01'));mean=float(raw[trainmask].mean());std=float(raw[trainmask].std());scales[plant]={'mean':mean,'std':std}
        values=(raw-mean)/std
        for split,(start,end) in SPLITS.items():
            start=max(pd.Timestamp(start),tt.iloc[0]+pd.Timedelta(hours=360));end=min(pd.Timestamp(end),tt.iloc[-1]+pd.Timedelta(hours=1))-pd.Timedelta(hours=24)
            for cutoff in pd.date_range(start.ceil('24h'),end,freq='24h'):
                pos=int(tt.searchsorted(cutoff))
                if pos<360 or pos+24>len(tt):continue
                segment=pd.DatetimeIndex(tt.iloc[pos-360:pos+24]);expected=pd.date_range(cutoff-pd.Timedelta(hours=360),periods=384,freq='h')
                if not segment.equals(expected):continue
                report=int(np.searchsorted(times,np.datetime64(cutoff),side='right')-1)
                if report<0:continue
                assert times[report]<=np.datetime64(cutoff)
                h=values[pos-360:pos];days=h[-7*24:].reshape(7,24)
                numeric=np.r_[h[-24:],h[-48:-24],days.mean(0),days.std(0),np.sin(2*np.pi*cutoff.dayofyear/365.25),np.cos(2*np.pi*cutoff.dayofyear/365.25)]
                samples[split].append({'plant':plant,'cutoff':cutoff.isoformat(),'report_index':report,'report_time':str(times[report]),'weather_age_hours':float((np.datetime64(cutoff)-times[report])/np.timedelta64(1,'h')),'numeric':numeric,'baseline':h[-24:],'target':values[pos:pos+24]})
    passport={'dataset':'VEWOXIC/Canada_photovoltaics_plants','data_geometry':'Regular hourly targets, regular daily forecast starts; not irregular-history calibration','entities':plants,'sample_counts':{s:len(v) for s,v in samples.items()},'plant_counts':{s:{p:sum(v['plant']==p for v in rows) for p in plants} for s,rows in samples.items()},'normalization':scales,'reports':len(times),'published_representation_shape':list(original.shape),'availability':'Report timestamp <= forecast cutoff; no arrival/issuance metadata; timestamp availability assumed','all_files_sha256':{str(f.relative_to(DATA)):sha(f) for f in files},'protocol_sha256':sha(ROOT/'protocol.json'),'script_sha256':sha(__file__)}
    save(OUT/'material_passport.json',passport)
    return samples,weather,mappings,passport
def design(samples,weather,mapping,variant,pca=None):
    result={}
    if variant!='numeric_only':
        tx=np.stack([weather[mapping[x['report_index']] if variant=='semantic_derangement' else x['report_index']] for x in samples['train']])
        if pca is None:pca=PCA(n_components=32,svd_solver='full').fit(tx)
    for split,pack in samples.items():
        nx=np.stack([x['numeric'] for x in pack]);target=np.stack([x['target'] for x in pack]);baseline=np.stack([x['baseline'] for x in pack])
        if variant!='numeric_only':
            wx=np.stack([weather[mapping[x['report_index']] if variant=='semantic_derangement' else x['report_index']] if variant!='zero' else np.zeros(weather.shape[1]) for x in pack])
            nx=np.c_[nx,pca.transform(wx)]
        result[split]=(nx,target,baseline,pack)
    return result,pca
def metric(pred,pack):
    _,target,_,records=pack;err=pred-target;by={}
    for plant in sorted({x['plant'] for x in records}):
        selected=np.array([x['plant']==plant for x in records]);e=err[selected];by[plant]={'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'observations':int(e.size)}
    return {'mse':float(np.mean(err*err)),'mae':float(np.mean(np.abs(err))),'observations':int(err.size),'macro_plant_mse':float(np.mean([x['mse'] for x in by.values()])),'plants':by}
def fit(dd):
    x,y,b,_=dd['train'];scaler=StandardScaler().fit(x);xx=scaler.transform(x);vx,vy,vb,_=dd['val'];grid=[]
    for a in (1.,10.,100.,1000.):
        mod=Ridge(alpha=a).fit(xx,y-b);pr=vb+mod.predict(scaler.transform(vx));grid.append((metric(pr,dd['val'])['mse'],a,mod))
    _,a,mod=min(grid,key=lambda z:z[0]);return scaler,mod,a,[{'alpha':a,'val_mse':m} for m,a,_ in grid]
def predict(fit,pack):return pack[2]+fit[1].predict(fit[0].transform(pack[0]))
def export(name,pred,pack):
    with (OUT/(name+'.csv')).open('w',encoding='utf8',newline='') as h:
        out=csv.writer(h);out.writerow(['plant','cutoff','hour_ahead','report_time','report_age_hours','target','prediction','squared_error'])
        for p,y,r in zip(pred,pack[1],pack[3]):
            for i in range(24):out.writerow([r['plant'],r['cutoff'],i,r['report_time'],r['weather_age_hours'],y[i],p[i],(p[i]-y[i])**2])
def main():
    samples,weather,mapping,passport=prepare();summary={'material':passport,'results':[]}
    base,_=design(samples,weather,mapping,'numeric_only')
    summary['results'].append({'training_condition':'persistence','evaluation_condition':'persistence','val':metric(base['val'][2],base['val']),'test':metric(base['test'][2],base['test'])})
    for training in ('numeric_only','aligned','semantic_derangement'):
        dd,pca=design(samples,weather,mapping,training);f=fit(dd)
        checkpoint=OUT/(training+'_checkpoint.npz');args={'scaler_mean':f[0].mean_,'scaler_scale':f[0].scale_,'coef':f[1].coef_,'intercept':f[1].intercept_}
        if pca is not None:args.update({'pca_mean':pca.mean_,'pca_components':pca.components_})
        np.savez_compressed(checkpoint,**args)
        for evaluation in ('aligned','semantic_derangement','zero') if training=='aligned' else (training,):
            ev,_=design(samples,weather,mapping,evaluation,pca=pca);pr=predict(f,ev['test'])
            record={'training_condition':training,'evaluation_condition':evaluation,'selected_alpha':f[2],'validation_grid':f[3],'checkpoint_sha256':sha(checkpoint),'val':metric(predict(f,ev['val']),ev['val']),'test':metric(pr,ev['test'])};summary['results'].append(record)
            export('trained_'+training+'_eval_'+evaluation,pr,ev['test']);save(OUT/'results.json',summary);print(json.dumps({'training':training,'evaluation':evaluation,'test':record['test']['mse']}),flush=True)
    test_ids=np.array([x['report_index'] for x in samples['test']]);old=weather[test_ids];new=weather[mapping[test_ids]]
    trainmean=weather[[x['report_index'] for x in samples['train']]].mean(0)
    def cos(a,b):return np.sum(a*b,axis=1)/(np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12))
    summary['intervention_audit']={'test_report_timestamps_never_after_cutoff':True,'report_embedding_unchanged_fraction':float(np.all(old==new,axis=1).mean()),'raw_cosine_median':float(np.median(cos(old,new))),'train_centered_cosine_median':float(np.median(cos(old-trainmean,new-trainmean))),'raw_weather_text_unavailable':True,'mean_test_report_age_hours':float(np.mean([x['weather_age_hours'] for x in samples['test']]))}
    save(OUT/'results.json',summary)
if __name__=='__main__':main()
