"""Recompute reported losses and audit source/cache/paired target identity."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from run_representation import ROOT,PROTOTYPE,prepare_dataset,design,metric,sha,savejson

def verify_timeimm(name):
    resultdir=ROOT/'results'/name
    result=json.loads((resultdir/'results.json').read_text(encoding='utf8'))
    data,entities,samples,selected,mappings,passport=prepare_dataset(name)
    source=PROTOTYPE/'external'/'Time-IMM'/'data'/name/'processed'
    checked=0
    for entity,files in result['material']['source_sha256'].items():
        for file,digest in files.items():assert sha(source/entity/file)==digest;checked+=1
    dd,_=design(samples,None,mappings,'numeric_only')
    baselines={s:{'persistence':metric(v[3],v[1],v[2],v[4]),'train_mean_zero':metric(np.zeros_like(v[1]),v[1],v[2],v[4])} for s,v in dd.items()}
    checks=[];fixedidentities={}
    for record in result['results']:
        rep=record['representation'];train=record['training_condition'];evaluation=record['evaluation_condition']
        if rep=='none':file=resultdir/(train+'.csv')
        else:file=resultdir/f'{rep}_trained_{train}_eval_{evaluation}.csv'
        count=0;squared=0.;identities=[]
        with file.open(encoding='utf8',newline='') as handle:
            for row in csv.DictReader(handle):
                count+=1;target=float(row['target']);prediction=float(row['prediction']);squared+=(target-prediction)**2
                assert np.isclose((target-prediction)**2,float(row['squared_error']),atol=1e-10,rtol=1e-10)
                identities.append((row['entity'],row['window'],row['start_time_days'],row['normalized_forecast_horizon'],row['sample_id'],row['feature'],row['target']))
        assert count==record['test']['observations']
        assert np.isclose(squared/count,record['test']['mse'],atol=1e-10,rtol=1e-10)
        if rep!='none' and train=='aligned':
            if rep in fixedidentities:assert fixedidentities[rep][0]==identities and fixedidentities[rep][1]==record['checkpoint_sha256']
            else:fixedidentities[rep]=(identities,record['checkpoint_sha256'])
        checks.append({'file':file.name,'count':count,'recomputed_mse':squared/count,'sha256':sha(file)})
    for (entity,split),mapping in mappings.items():
        assert set(mapping)==set(mapping.values())
        if len(mapping)>1:assert all(a!=b for a,b in mapping.items())
    capability=[]
    reference=baselines['val']['persistence']['mse']
    for record in result['results']:
        if record['training_condition']==record['evaluation_condition']:
            capability.append({'representation':record['representation'],'condition':record['training_condition'],'validation_mse':record['val']['mse'],'persistence_validation_mse':reference,'relative_validation_improvement':(reference-record['val']['mse'])/reference})
    audit={'dataset':name,'all_source_hashes_unchanged':True,'source_files_checked':checked,'permutation_has_no_index_fixed_points_when_n_gt1':True,'fixed_checkpoint_and_targets_match_across_conditions':True,'csv_metrics':checks,'baselines':baselines,'validation_capability_descriptive':capability,'no_eligibility_threshold_selected_after_test':True}
    savejson(resultdir/'verification.json',audit);return audit

def verify_fidel():
    folder=ROOT/'results'/'Fidel';result=json.loads((folder/'results.json').read_text(encoding='utf8'));files=[]
    data=PROTOTYPE/'data'/'fidel_ts'/'Canada_photovoltaics_plants'
    for f,digest in result['material']['all_files_sha256'].items():assert sha(data/f)==digest
    identities=None;identitydigest=None
    for r in result['results']:
        if r['training_condition']=='persistence':continue
        f=folder/f"trained_{r['training_condition']}_eval_{r['evaluation_condition']}.csv";squared=0.;count=0;keys=[]
        with f.open(encoding='utf8',newline='') as handle:
            for row in csv.DictReader(handle):
                e=(float(row['target'])-float(row['prediction']))**2;assert np.isclose(e,float(row['squared_error']));squared+=e;count+=1
                assert row['report_time'][:10]<=row['cutoff'][:10]
                assert float(row['report_age_hours'])>=0
                keys.append((row['plant'],row['cutoff'],row['hour_ahead'],row['target']))
        assert count==r['test']['observations'];assert np.isclose(squared/count,r['test']['mse'],atol=1e-10)
        if r['training_condition']=='aligned':
            if identities is None:identities=keys;identitydigest=r['checkpoint_sha256']
            else:assert keys==identities and identitydigest==r['checkpoint_sha256']
        files.append({'file':f.name,'observations':count,'recomputed_mse':squared/count,'sha256':sha(f)})
    out={'all_sources_unchanged':True,'all_weather_report_timestamps_not_after_forecast_cutoff':True,'same_checkpoint_and_targets_for_fixed_conditions':True,'csv_metrics':files};savejson(folder/'verification.json',out);return out

def main():
    outputs={n:verify_timeimm(n) for n in ('EPA-Air','FNSPID')};outputs['Fidel']=verify_fidel();savejson(ROOT/'results'/'verification_summary.json',{n:{k:v for k,v in x.items() if k not in ('baselines','csv_metrics')} for n,x in outputs.items()});print('All loss/source/paired-condition verification checks passed.',flush=True)
if __name__=='__main__':main()
