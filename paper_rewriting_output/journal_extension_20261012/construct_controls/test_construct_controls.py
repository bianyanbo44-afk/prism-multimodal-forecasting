"""Generator-defined semantics/proxy/availability tests; no fitted test scores."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from run_construct_controls import sentences, phrase_index, generate, history_indices

HERE = Path(__file__).resolve().parent


class ConstructTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.cfg["entities_per_split"] = {"train": 8, "val": 8, "test": 8}
        self.schedules = [{"name": str(i), "numeric": np.arange(0,30,.125),
                           "text": np.array([1.,2.2,4.3,6.8,7.3,10.,12.,16.,19.,22.,24.]),
                           "cutoffs": np.array([7.,14.,21.,28.])} for i in range(8)]
        self.phrases, self.d, self.p = sentences(self.cfg)
        # Structural algebra only. The experiment encodes actual prose.
        self.enc = np.column_stack([self.d,self.p,np.arange(48)/48])

    def test_all_direction_proxy_combinations_are_balanced(self):
        self.assertEqual(len(self.phrases),48)
        for block in [slice(0,16),slice(16,32),slice(32,48)]:
            self.assertEqual(np.mean(self.d[block]),0)
            self.assertEqual(np.mean(self.p[block]),0)
            self.assertEqual(np.mean(self.d[block]*self.p[block]),0)

    def test_meaning_flip_preserves_exact_proxy_suffix(self):
        for idx in range(48):
            self.assertEqual(self.p[idx],self.p[idx^2])
            self.assertEqual(self.d[idx],-self.d[idx^2])
            self.assertEqual(self.phrases[idx].split(" Bulletin source:")[1], self.phrases[idx^2].split(" Bulletin source:")[1])

    def test_proxy_flip_preserves_exact_directional_proposition(self):
        for idx in range(48):
            self.assertEqual(self.d[idx],self.d[idx^1])
            self.assertEqual(self.p[idx],-self.p[idx^1])
            self.assertEqual(self.phrases[idx].split(" Bulletin source:")[0], self.phrases[idx^1].split(" Bulletin source:")[0])

    def test_paraphrase_preserves_constructs_but_changes_actual_words(self):
        for idx in range(16,32):
            other = idx+16
            self.assertEqual(self.d[idx],self.d[other])
            self.assertEqual(self.p[idx],self.p[other])
            self.assertNotEqual(self.phrases[idx],self.phrases[other])

    def test_oracle_flip_and_paraphrase_invariance(self):
        data,_,_ = generate(self.cfg,self.schedules,self.enc,123)
        for z in data.values():
            np.testing.assert_allclose(z["meaning_flip"][:,0],-z["direction_signal"])
            np.testing.assert_allclose(z["meaning_flip"][:,1],z["proxy_signal"])
            np.testing.assert_allclose(z["proxy_flip"][:,0],z["direction_signal"])
            np.testing.assert_allclose(z["proxy_flip"][:,1],-z["proxy_signal"])
            np.testing.assert_allclose(z["meaning_preserving_paraphrase"][:,:2],z["aligned"][:,:2])
            np.testing.assert_array_equal(z["zero"],np.zeros_like(z["aligned"]))

    def test_target_mechanisms_have_distinct_truth_without_mutating_data(self):
        data,notes,identity = generate(self.cfg,self.schedules,self.enc,123)
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["mechanisms"] = list(reversed(cfg2["mechanisms"]))
        other,notes2,identity2 = generate(cfg2,self.schedules,self.enc,123)
        self.assertEqual(notes,notes2)
        self.assertEqual(identity,identity2)
        for split,z in data.items():
            for key in z:
                np.testing.assert_array_equal(z[key],other[split][key])
            for field in ["direction_signal","proxy_signal","timing_signal"]:
                y = z["baseline"]+z[field]+z["noise"]
                np.testing.assert_allclose(y-z["baseline"]-z["noise"],z[field],atol=1e-12)

    def test_split_random_streams_and_holdout(self):
        data,notes,_ = generate(self.cfg,self.schedules,self.enc,123)
        self.assertTrue(all(n["phrase_index"]<16 for n in notes if n["split"]=="train"))
        self.assertTrue(all(16<=n["phrase_index"]<32 for n in notes if n["split"]!="train"))
        self.assertFalse(np.array_equal(data["train"]["direction_signal"],data["test"]["direction_signal"]))
        self.assertFalse(np.array_equal(data["test"]["direction_signal"],data["test"]["proxy_signal"]))

    def test_timing_standardization_fitted_only_in_training(self):
        data,_,identity = generate(self.cfg,self.schedules,self.enc,123)
        self.assertAlmostEqual(data["train"]["timing_signal"].mean(),0)
        self.assertAlmostEqual(data["train"]["timing_signal"].std(),1)
        for split,z in data.items():
            np.testing.assert_allclose(z["timing_signal"],(z["common"][:,5]-identity["timing_center_train_only"])/identity["timing_scale_train_only"])

    def test_availability_rejects_future_and_late_notes(self):
        event=np.array([1.,4.,4.5,6.])
        arrival=np.array([1.,4.1,7.,6.])
        np.testing.assert_array_equal(history_indices(event,arrival,5.,7.),[0,1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
