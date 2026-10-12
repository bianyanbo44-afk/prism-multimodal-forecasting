import copy,unittest
import numpy as np
from run_representation import design,train,predict

def fixtures():
    records={};vectors={'A':{i:np.array([i,1.,float(i%2),.5]) for i in range(12)}};mappings={}
    for split,offset in (('train',0),('val',4),('test',8)):
        records[split]=[];ids=list(range(offset,offset+4));mappings[('A',split)]={a:b for a,b in zip(ids,np.roll(ids,1))}
        for k,i in enumerate(ids):
            records[split].append({'entity':'A','window':i,'start':float(i),'sample_id':i,'selected':[i],'structure':np.array([1.,.1]),'numeric':np.array([i*.1,1.]),'last':np.array([i*.1]),'target':np.array([[i*.12]]),'mask':np.array([[True]]),'horizons':np.array([.5])})
    return records,vectors,mappings

class Tests(unittest.TestCase):
    def test_permutation_preserves_numeric_time_and_targets(self):
        samples,vectors,mapping=fixtures();aligned,pca=design(samples,vectors,mapping,'aligned');shuffled,_=design(samples,vectors,mapping,'semantic_derangement',pca)
        for split in samples:
            np.testing.assert_array_equal(aligned[split][0][:,:4],shuffled[split][0][:,:4]);np.testing.assert_array_equal(aligned[split][0][:,-2:],shuffled[split][0][:,-2:])
            for i in (1,2,3):np.testing.assert_array_equal(aligned[split][i],shuffled[split][i])
            self.assertEqual(aligned[split][4],shuffled[split][4])
    def test_zero_keeps_feature_dimension_and_finiteness(self):
        samples,vectors,mapping=fixtures();aligned,pca=design(samples,vectors,mapping,'aligned');zero,_=design(samples,vectors,mapping,'zero',pca)
        self.assertEqual(zero['test'][0].shape,aligned['test'][0].shape);self.assertTrue(np.isfinite(zero['test'][0]).all())
    def test_changed_test_targets_do_not_change_training_or_selection(self):
        samples,vectors,mapping=fixtures();dd,_=design(samples,None,mapping,'numeric_only');first=train(dd);changed=copy.deepcopy(dd);changed['test'][1][:]=99999.;second=train(changed)
        self.assertEqual(first[2],second[2]);np.testing.assert_array_equal(first[0].mean_,second[0].mean_);np.testing.assert_array_equal(first[1][0].coef_,second[1][0].coef_)
    def test_changed_test_embedding_does_not_change_pca_or_train(self):
        samples,vectors,mapping=fixtures();_,first=design(samples,vectors,mapping,'aligned');changed=copy.deepcopy(vectors)
        for i in range(8,12):changed['A'][i]*=1000.
        _,second=design(samples,changed,mapping,'aligned');np.testing.assert_array_equal(first.mean_,second.mean_);np.testing.assert_array_equal(first.components_,second.components_)

if __name__=='__main__':unittest.main()
