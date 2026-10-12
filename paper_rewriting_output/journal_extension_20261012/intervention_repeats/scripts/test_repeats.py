import ast,hashlib,unittest
from pathlib import Path
import numpy as np
from run_repeats import ROOT,EXT,cyclic_mapping,load_state,replay,digest_arrays

class RepeatInvariantTests(unittest.TestCase):
    def test_group_multiset_and_singleton_invariance(self):
        groups={('A','test'):{1,2,3,4},('B','test'):{10},('A','val'):{20,21}}
        for seed in range(120001,120021):
            maps,_=cyclic_mapping(groups,seed)
            for group,ids in groups.items():
                self.assertEqual(set(maps[group]),ids);self.assertEqual(set(maps[group].values()),ids)
                self.assertEqual(sum(i!=j for i,j in maps[group].items()),len(ids) if len(ids)>1 else 0)
            self.assertEqual(maps[('B','test')][10],10)
    def test_saved_checkpoint_readonly_and_hash_unchanged(self):
        file=EXT/'results'/'Fidel'/'aligned_checkpoint.npz';before=hashlib.sha256(file.read_bytes()).hexdigest();state=load_state(file)
        with self.assertRaises(ValueError):state['coef'][0,0]=1e99
        self.assertEqual(before,hashlib.sha256(file.read_bytes()).hexdigest())
    def test_closed_form_content_delta_matches_direct_replay(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(9,7));base=rng.normal(size=(9,2))
        state={'scaler_mean':rng.normal(size=7),'scaler_scale':np.arange(1,8,dtype=float),'coef':rng.normal(size=(2,7)),'intercept':rng.normal(size=2)}
        changed=x.copy();changed[:,3:5]+=rng.normal(size=(9,2));delta=changed[:,3:5]-x[:,3:5]
        fast=replay(state,x,base)+(delta/state['scaler_scale'][3:5])@state['coef'][:,3:5].T
        np.testing.assert_allclose(fast,replay(state,changed,base),atol=1e-12)
        np.testing.assert_array_equal(changed[:,:3],x[:,:3]);np.testing.assert_array_equal(changed[:,5:],x[:,5:])
    def test_target_hash_and_no_fit_call(self):
        target=np.arange(12,dtype=float).reshape(6,2);mask=target%2==0;before=digest_arrays(target,mask)
        copied=target.copy();self.assertEqual(before,digest_arrays(copied,mask));copied[0,0]=99.;self.assertNotEqual(before,digest_arrays(copied,mask))
        tree=ast.parse((ROOT/'scripts'/'run_repeats.py').read_text(encoding='utf8'))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in ast.walk(tree)))

if __name__=='__main__':
    with (ROOT/'results'/'test_results.txt').open('w',encoding='utf8') as handle:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(RepeatInvariantTests)
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    print(f'Invariant unit tests: {result.testsRun} run; success={result.wasSuccessful()}',flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
