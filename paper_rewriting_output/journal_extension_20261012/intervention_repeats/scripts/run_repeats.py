"""Between-map sensitivity, replaying saved aligned linear states without fitting."""
from __future__ import annotations
import csv,hashlib,json,sys,time,platform
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT.parent/'external_validation'
sys.path.insert(0,str(EXT/'scripts'))
import run_representation as rr
import run_fidel_independent as rf
SEEDS=list(range(120001,120021))
torch.set_num_threads(4)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def digest_arrays(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        arr=np.ascontiguousarray(a);h.update(str(arr.shape).encode());h.update(str(arr.dtype).encode());h.update(arr.tobytes())
    return h.hexdigest()

class FrozenPCA:
    def __init__(self,state):
        self.mean_=state['pca_mean'];self.components_=state['pca_components']
    def transform(self,x):return (np.asarray(x)-self.mean_)@self.components_.T

def load_state(path):
    z=np.load(path);state={k:z[k] for k in z.files}
    for v in state.values():v.setflags(write=False)
    return state

def replay(state,x,baseline):
    return baseline+((x-state['scaler_mean'])/state['scaler_scale'])@state['coef'].T+state['intercept']

def check_state_hash(path,expected):assert sha(path)==expected

def metrics(pred,target,mask,units):
    errors=(pred-target)**2;count=int(mask.sum());by={}
    labels=np.array(units)
    for unit in sorted(set(units)):
        rows=labels==unit;mm=mask[rows];e=errors[rows]
        by[unit]={'mse':float(e[mm].mean()),'count':int(mm.sum())}
    return {'mse':float(errors[mask].mean()),'count':count,'macro_mse':float(np.mean([x['mse'] for x in by.values()])),'units':by}

def cyclic_mapping(groups,seed):
    mappings={};shifts={}
    for group_index,(group,ids) in enumerate(sorted(groups.items())):
        ordered=np.array(sorted(ids));rng=np.random.default_rng(np.random.SeedSequence([seed,group_index]))
        shift=int(rng.integers(1,len(ordered))) if len(ordered)>=2 else 0
        mapping={int(a):int(b) for a,b in zip(ordered,np.roll(ordered,shift))}
        assert set(mapping)==set(mapping.values())
        assert all(a!=b for a,b in mapping.items()) if len(mapping)>1 else True
        mappings[group]=mapping;shifts[str(group)]=shift
    return mappings,shifts

def timing_rows_timeimm(dataset,samples,entities,mapping,seed):
    rows=[]
    for sample in samples['test']:
        entity=sample['entity'];cutoff=sample['start']+rr.DATASET_SPECS[dataset].history
        for i in sample['selected']:
            j=mapping[(entity,'test')][i];times=entities[entity]['times']
            rows.append({'dataset':dataset,'base_seed':seed,'entity':entity,'window':sample['window'],'recipient_index':i,'donor_index':j,'recipient_time':times[i],'recipient_cutoff':cutoff,'donor_time':times[j],'donor_after_cutoff':int(times[j]>cutoff),'time_unit':'days from first numeric timestamp','raw_text_changed':int(entities[entity]['notes'][i]!=entities[entity]['notes'][j])})
    return rows

def add_records(aggregates,unitrows,dataset,representation,seed,baseline,loss,dose,checkpoint,target_hash):
    relative=(loss['mse']-baseline['mse'])/loss['mse']
    aggregates.append({'dataset':dataset,'representation':representation,'base_seed':seed,'aligned_mse':baseline['mse'],'map_mse':loss['mse'],'loss_difference':loss['mse']-baseline['mse'],'relative_aligned_gain':relative,'aligned_macro_mse':baseline['macro_mse'],'map_macro_mse':loss['macro_mse'],'macro_loss_difference':loss['macro_mse']-baseline['macro_mse'],'observed_scalar_targets':loss['count'],'checkpoint_sha256':checkpoint,'target_multiset_sha256':target_hash,**dose})
    for unit,value in loss['units'].items():
        original=baseline['units'][unit]
        unitrows.append({'dataset':dataset,'representation':representation,'base_seed':seed,'unit':unit,'aligned_mse':original['mse'],'map_mse':value['mse'],'loss_difference':value['mse']-original['mse'],'relative_aligned_gain':(value['mse']-original['mse'])/value['mse'],'observed_scalar_targets':value['count'],'checkpoint_sha256':checkpoint,'target_multiset_sha256':target_hash})

def run_timeimm(name,aggregates,unitrows,timing,checks):
    _,entities,samples,selected,oldmaps,passport=rr.prepare_dataset(name)
    groups=defaultdict(set)
    for split,pack in samples.items():
        for sample in pack:groups[(sample['entity'],split)].update(sample['selected'])
    maps={seed:cyclic_mapping(groups,seed) for seed in SEEDS}
    save(ROOT/'results'/f'{name}_map_shifts.json',{str(seed):shifts for seed,(_,shifts) in maps.items()})
    for seed,(mapping,_) in maps.items():timing.extend(timing_rows_timeimm(name,samples,entities,mapping,seed))
    old=json.loads((EXT/'results'/name/'results.json').read_text(encoding='utf8'))
    for entity,files in old['material']['source_sha256'].items():
        for filename,expected in files.items():assert sha(rr.PROTOTYPE/'external'/'Time-IMM'/'data'/name/'processed'/entity/filename)==expected
    chosen=[(e,i) for e in sorted(selected) for i in sorted(selected[e])]
    test_samples=samples['test'];repeat_idx=np.repeat(np.arange(len(test_samples)),[len(x['horizons']) for x in test_samples])
    selected_slots=[(x['entity'],i) for x in test_samples for i in x['selected']]
    test_groups={k:v for k,v in groups.items() if k[1]=='test'}
    eligible_slots=sum(len(v) for v in test_groups.values() if len(v)>=2)
    for representation in ('gpt2','minilm','finbert'):
        vectors=defaultdict(dict)
        if representation=='gpt2':
            for entity in selected:
                raw=torch.load(rr.PROTOTYPE/'external'/'Time-IMM'/'data'/name/'processed'/entity/rr.GPT2_TEXT_CACHE,weights_only=True,map_location='cpu')
                for i in selected[entity]:vectors[entity][i]=raw['embeddings'][i].float().numpy()
        else:
            cache=np.load(EXT/'caches'/name/(representation+'.npz'));encoded=cache['embeddings']
            for (entity,i),v in zip(chosen,encoded):vectors[entity][i]=v
        checkpoint=EXT/'results'/name/f'{representation}_aligned_checkpoint.npz';checkpoint_hash=sha(checkpoint)
        expected=next(r for r in old['results'] if r['representation']==representation and r['training_condition']=='aligned' and r['evaluation_condition']=='aligned')
        assert checkpoint_hash==expected['checkpoint_sha256']
        state=load_state(checkpoint);pca=FrozenPCA(state)
        prepared,_=rr.design(samples,vectors,oldmaps,'aligned',pca=pca)
        x,target,mask,base,records=prepared['test']
        baseline_pred=replay(state,x,base);baseline=metrics(baseline_pred,target,mask,[r[0] for r in records])
        assert np.isclose(baseline['mse'],expected['test']['mse'],atol=1e-10,rtol=1e-10)
        target_hash=digest_arrays(target,mask);ids_hash=hashlib.sha256(json.dumps(records).encode()).hexdigest()
        aligned_pool=np.stack([np.mean([vectors[s['entity']][i] for i in s['selected']],axis=0,dtype=np.float64) for s in test_samples])
        content_width=len(state['pca_components']);start=x.shape[1]-2-content_width;stop=start+content_width
        pooled_baseline=pca.transform(aligned_pool)
        assert np.allclose(x[:,start:stop],pooled_baseline[repeat_idx],atol=1e-12)
        for seed,(mapping,_) in maps.items():
            pool=np.stack([np.mean([vectors[s['entity']][mapping[(s['entity'],'test')][i]] for i in s['selected']],axis=0,dtype=np.float64) for s in test_samples])
            component_delta=pca.transform(pool)-pooled_baseline
            prediction=baseline_pred+((component_delta/state['scaler_scale'][start:stop])@state['coef'][:,start:stop].T)[repeat_idx]
            if seed==SEEDS[0]:
                perturbed=x.copy();perturbed[:,start:stop]=pca.transform(pool)[repeat_idx]
                np.testing.assert_array_equal(perturbed[:,:start],x[:,:start]);np.testing.assert_array_equal(perturbed[:,stop:],x[:,stop:])
                assert np.allclose(prediction,replay(state,perturbed,base),atol=1e-10,rtol=1e-10)
            assert digest_arrays(target,mask)==target_hash
            assert hashlib.sha256(json.dumps(records).encode()).hexdigest()==ids_hash
            changed_indices=sum(mapping[(entity,'test')][i]!=i for entity,i in selected_slots)
            changed_text=sum(entities[e]['notes'][i]!=entities[e]['notes'][mapping[(e,'test')][i]] for e,i in selected_slots)
            changed_vectors=sum(not np.array_equal(vectors[e][i],vectors[e][mapping[(e,'test')][i]]) for e,i in selected_slots)
            changed_windows=int(np.sum(np.linalg.norm(pool-aligned_pool,axis=1)>1e-10))
            dose={'selected_recipient_slots':len(selected_slots),'eligible_recipient_slots':eligible_slots,'changed_source_index_slots':changed_indices,'changed_raw_text_slots':changed_text,'changed_vector_slots':changed_vectors,'test_windows':len(test_samples),'changed_pooled_windows':changed_windows,'changed_projected_windows':int(np.sum(np.linalg.norm(component_delta,axis=1)>1e-10)),'donor_after_cutoff_fraction':float(np.mean([entities[s['entity']]['times'][mapping[(s['entity'],'test')][i]]>s['start']+rr.DATASET_SPECS[name].history for s in test_samples for i in s['selected']]))}
            loss=metrics(prediction,target,mask,[r[0] for r in records])
            add_records(aggregates,unitrows,name,representation,seed,baseline,loss,dose,checkpoint_hash,target_hash)
        check_state_hash(checkpoint,checkpoint_hash)
        checks.append({'dataset':name,'representation':representation,'aligned_replay_mse':baseline['mse'],'prior_aligned_mse':expected['test']['mse'],'checkpoint_sha256':checkpoint_hash,'target_sha256':target_hash,'identifier_multiset_sha256':ids_hash,'baseline_replay_matches':True,'fixed_targets_and_noncontent_inputs_pass':True,'original_source_hashes_pass':True,'no_model_refit':True})
        print(json.dumps({'dataset':name,'representation':representation,'maps_completed':len(SEEDS)}),flush=True)

def run_fidel(aggregates,unitrows,timing,checks):
    rf.OUT=ROOT/'reconstruction'
    samples,weather,original_map,passport=rf.prepare()
    old=json.loads((EXT/'results'/'Fidel'/'results.json').read_text(encoding='utf8'))
    expected=next(r for r in old['results'] if r['training_condition']=='aligned' and r['evaluation_condition']=='aligned')
    for file,digest in old['material']['all_files_sha256'].items():assert sha(rf.DATA/file)==digest
    # Reconstruct report timestamp identity from the original ordered published keys.
    import joblib,pandas as pd
    reports={}
    for y in range(2018,2023):reports.update(joblib.load(rf.DATA/'weather'/'calgary'/'report_embedding'/'formal_report'/f'fast_general_formal_embeddings_{y}.pkl'))
    times=pd.to_datetime(sorted(reports),format='%Y%m%d%H%M').to_numpy(dtype='datetime64[ns]')
    checkpoint=EXT/'results'/'Fidel'/'aligned_checkpoint.npz';checkpoint_hash=sha(checkpoint);assert checkpoint_hash==expected['checkpoint_sha256']
    state=load_state(checkpoint);pca=FrozenPCA(state);prepared,_=rf.design(samples,weather,original_map,'aligned',pca=pca)
    x,target,base,records=prepared['test'];mask=np.ones_like(target,dtype=bool);units=[r['plant'] for r in records]
    baseline_pred=replay(state,x,base);baseline=metrics(baseline_pred,target,mask,units)
    assert np.isclose(baseline['mse'],expected['test']['mse'],atol=1e-10,rtol=1e-10)
    target_hash=digest_arrays(target,mask);ids_hash=hashlib.sha256(json.dumps([(x['plant'],x['cutoff']) for x in records]).encode()).hexdigest()
    recipient_indices=np.array([r['report_index'] for r in records]);aligned_pool=weather[recipient_indices]
    content_width=len(state['pca_components']);start=x.shape[1]-content_width;stop=x.shape[1];pcabase=pca.transform(aligned_pool)
    assert np.allclose(x[:,start:stop],pcabase,atol=1e-12)
    shifts={}
    for seed in SEEDS:
        mapping=np.arange(len(times));shift_record={}
        for split_index,(split,(begin,end)) in enumerate(rf.SPLITS.items()):
            ids=np.flatnonzero((times>=np.datetime64(begin))&(times<np.datetime64(end)))
            rng=np.random.default_rng(np.random.SeedSequence([seed,split_index,91801]));shift=int(rng.integers(1,len(ids)))
            mapping[ids]=np.roll(ids,shift);shift_record[split]=shift
            assert set(ids)==set(mapping[ids]);assert np.all(ids!=mapping[ids])
        shifts[str(seed)]=shift_record
        donor_indices=mapping[recipient_indices];pool=weather[donor_indices];delta=pca.transform(pool)-pcabase
        prediction=baseline_pred+(delta/state['scaler_scale'][start:stop])@state['coef'][:,start:stop].T
        if seed==SEEDS[0]:
            perturbed=x.copy();perturbed[:,start:stop]=pca.transform(pool);np.testing.assert_array_equal(perturbed[:,:start],x[:,:start])
            assert np.allclose(prediction,replay(state,perturbed,base),atol=1e-10,rtol=1e-10)
        assert digest_arrays(target,mask)==target_hash
        assert hashlib.sha256(json.dumps([(x['plant'],x['cutoff']) for x in records]).encode()).hexdigest()==ids_hash
        future=[]
        for r,recipient,donor in zip(records,recipient_indices,donor_indices):
            assert times[recipient]<=np.datetime64(r['cutoff'])
            after=int(times[donor]>np.datetime64(r['cutoff']));future.append(after)
            timing.append({'dataset':'Fidel','base_seed':seed,'entity':r['plant'],'window':r['cutoff'],'recipient_index':int(recipient),'donor_index':int(donor),'recipient_time':str(times[recipient]),'recipient_cutoff':r['cutoff'],'donor_time':str(times[donor]),'donor_after_cutoff':after,'time_unit':'ISO timestamp','raw_text_changed':None})
        changed=int(np.sum(np.linalg.norm(pool-aligned_pool,axis=1)>1e-10))
        dose={'selected_recipient_slots':len(records),'eligible_recipient_slots':len(records),'changed_source_index_slots':int(np.sum(recipient_indices!=donor_indices)),'changed_raw_text_slots':None,'changed_vector_slots':changed,'test_windows':len(records),'changed_pooled_windows':changed,'changed_projected_windows':int(np.sum(np.linalg.norm(delta,axis=1)>1e-10)),'donor_after_cutoff_fraction':float(np.mean(future))}
        loss=metrics(prediction,target,mask,units);add_records(aggregates,unitrows,'Fidel','published_weather',seed,baseline,loss,dose,checkpoint_hash,target_hash)
    check_state_hash(checkpoint,checkpoint_hash);save(ROOT/'results'/'Fidel_map_shifts.json',shifts)
    checks.append({'dataset':'Fidel','representation':'published_weather','aligned_replay_mse':baseline['mse'],'prior_aligned_mse':expected['test']['mse'],'checkpoint_sha256':checkpoint_hash,'target_sha256':target_hash,'identifier_multiset_sha256':ids_hash,'baseline_replay_matches':True,'fixed_targets_and_noncontent_inputs_pass':True,'original_source_hashes_pass':True,'no_model_refit':True,'recipient_report_times_not_after_cutoff':True,'donor_content_is_offline_intervention':True})
    print(json.dumps({'dataset':'Fidel','maps_completed':len(SEEDS)}),flush=True)

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf8',newline='') as handle:
        out=csv.DictWriter(handle,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)

def summary(rows,unit=False):
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['representation'],r['unit'] if unit else '')].append(r)
    result=[]
    for (dataset,representation,name),rrs in groups.items():
        delta=np.array([x['loss_difference'] for x in rrs]);relative=np.array([x['relative_aligned_gain'] for x in rrs])
        row={'dataset':dataset,'representation':representation,'unit':name,'map_draws':len(delta),'aligned_mse':rrs[0]['aligned_mse'],'mean_map_mse':float(np.mean([x['map_mse'] for x in rrs])),'mean_loss_difference':float(delta.mean()),'empirical_q025_loss_difference':float(np.quantile(delta,.025)),'empirical_q975_loss_difference':float(np.quantile(delta,.975)),'min_loss_difference':float(delta.min()),'max_loss_difference':float(delta.max()),'positive_map_count':int((delta>0).sum()),'zero_map_count':int((delta==0).sum()),'negative_map_count':int((delta<0).sum()),'mean_relative_aligned_gain':float(relative.mean()),'empirical_q025_relative_gain':float(np.quantile(relative,.025)),'empirical_q975_relative_gain':float(np.quantile(relative,.975)),'observed_scalar_targets':rrs[0]['observed_scalar_targets']}
        if not unit:
            macro=np.array([r['macro_loss_difference'] for r in rrs]);row.update({'mean_macro_loss_difference':float(macro.mean()),'empirical_q025_macro_difference':float(np.quantile(macro,.025)),'empirical_q975_macro_difference':float(np.quantile(macro,.975)),'macro_positive_map_count':int((macro>0).sum()),'eligible_recipient_slots':rrs[0]['eligible_recipient_slots'],'selected_recipient_slots':rrs[0]['selected_recipient_slots'],'min_changed_vector_slots':min(r['changed_vector_slots'] for r in rrs),'max_changed_vector_slots':max(r['changed_vector_slots'] for r in rrs),'min_changed_pooled_windows':min(r['changed_pooled_windows'] for r in rrs),'max_changed_pooled_windows':max(r['changed_pooled_windows'] for r in rrs),'min_donor_after_cutoff_fraction':min(r['donor_after_cutoff_fraction'] for r in rrs),'max_donor_after_cutoff_fraction':max(r['donor_after_cutoff_fraction'] for r in rrs)})
        result.append(row)
    return result

def main():
    started=time.perf_counter()
    inputs=[EXT/'scripts'/'run_representation.py',EXT/'scripts'/'run_fidel_independent.py',EXT/'protocol.json',EXT/'representation_manifest.json']
    inputs += [EXT/'results'/name/f'{rep}_aligned_checkpoint.npz' for name in ('FNSPID','EPA-Air') for rep in ('gpt2','minilm','finbert')]
    inputs += [EXT/'results'/'Fidel'/'aligned_checkpoint.npz']
    frozen={'protocol_sha256':sha(ROOT/'protocol.json'),'script_sha256':sha(__file__),'existing_input_hashes':{str(p):sha(p) for p in inputs},'frozen_before_scoring':True,'base_seeds':SEEDS}
    freeze_path=ROOT/'FROZEN_PROTOCOL_MANIFEST.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text(encoding='utf8'))==frozen,'Changed freeze; preserve prior run rather than silently reuse protocol'
    else:save(freeze_path,frozen)
    aggregates=[];unitrows=[];timing=[];checks=[]
    for name in ('FNSPID','EPA-Air'):run_timeimm(name,aggregates,unitrows,timing,checks)
    run_fidel(aggregates,unitrows,timing,checks)
    aggregate_summary=summary(aggregates);unit_summary=summary(unitrows,unit=True)
    write_csv(ROOT/'results'/'map_aggregate.csv',aggregates);write_csv(ROOT/'results'/'map_unit_effects.csv',unitrows);write_csv(ROOT/'results'/'map_slot_timing.csv',timing);write_csv(ROOT/'results'/'between_map_summary.csv',aggregate_summary);write_csv(ROOT/'results'/'between_map_unit_summary.csv',unit_summary)
    for p,expected in frozen['existing_input_hashes'].items():assert sha(p)==expected
    save(ROOT/'results'/'results.json',{'protocol':json.loads((ROOT/'protocol.json').read_text(encoding='utf8')),'frozen_manifest':frozen,'summaries':aggregate_summary,'unit_summaries':unit_summary,'invariant_checks':checks,'runtime':{'duration_seconds':time.perf_counter()-started,'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),'platform':platform.platform(),'device':'cpu'},'all_checks_passed':True,'existing_input_hashes_unchanged':True,'refit_count':0,'scope':'Between cyclic-map draws; NOT sampling confidence interval or neural-checkpoint calibration'})
    print(json.dumps({'complete':True,'maps_scored':len(aggregates),'unit_records':len(unitrows),'timing_slots':len(timing),'seconds':round(time.perf_counter()-started,1)}),flush=True)
if __name__=='__main__':main()
