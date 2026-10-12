"""Boundary tests for intervention validity and paired dependence analysis."""
from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from run_fixed_model import (derangement, dose_source_maps, groups_for, make_variants,
                             paired_bootstrap, train_centroid, verify_invariants)
from eventmamba.data import AsyncEvent
from eventmamba.timeimm import TimeIMMWindowInfo, TimeIMMWindowSample


def window(sample_id, text_values):
    events = [AsyncEvent(event_time=0.0, arrival_time=0.0, modality=0,
                         value=torch.tensor([3.0]), quality=1.0, event_id=0)]
    events += [AsyncEvent(event_time=float(i + 1), arrival_time=float(i + 1),
                          modality=1, value=torch.tensor(value, dtype=torch.float32),
                          quality=1.0, event_id=i + 1) for i, value in enumerate(text_values)]
    return TimeIMMWindowSample(tuple(events), 10.0, torch.tensor([1.0]),
                               torch.tensor([[2.0]]), torch.tensor([[1.0]]), sample_id)


class DummyData:
    text_modality = 1
    test_windows = (window(1, [[1, 2]]), window(2, [[3, 4], [5, 6], [7, 8], [9, 10]]))
    train_windows = (window(3, [[2, 4], [4, 6]]),)
    window_info = {1: TimeIMMWindowInfo("singleton", 0, 0.0),
                   2: TimeIMMWindowInfo("multi", 0, 0.0),
                   3: TimeIMMWindowInfo("train", 0, 0.0)}


class InterventionTests(unittest.TestCase):
    def test_derangement_has_no_fixed_point_and_preserves_multiset(self):
        for n in range(2, 50):
            p = derangement(n, random.Random(42))
            self.assertEqual(sorted(p), list(range(n)))
            self.assertTrue(all(i != j for i, j in enumerate(p)))
        self.assertEqual(derangement(1, random.Random(42)), [0])

    def test_dose_reports_impossible_subsets_and_nested_changed_locations(self):
        groups = {"a": [(0, i) for i in range(8)], "singleton": [(1, 0)]}
        last = set()
        for dose in (0.25, 0.5, 0.75, 1.0):
            maps = dose_source_maps(groups, dose, 9103)
            changed = {i for i, j in enumerate(maps["a"]) if i != j}
            self.assertTrue(last <= changed)
            last = changed
            self.assertEqual(maps["singleton"], [0])
            self.assertEqual(sorted(maps["a"]), list(range(8)))
        self.assertEqual(dose_source_maps({"a": [(0, 0), (0, 1)]}, 0.5, 4)["a"], [0, 1])

    def test_controls_keep_numeric_event_and_targets(self):
        variants, coverage, _, _ = make_variants(DummyData())
        self.assertEqual(len(variants), 8)
        for altered in variants.values():
            verify_invariants(DummyData.test_windows, altered, 1)
        self.assertEqual(coverage["derangement_100"]["changed_source_positions"], 4)
        self.assertEqual(coverage["derangement_100"]["singleton_positions"], 1)
        self.assertEqual(coverage["derangement_100"]["achieved_value_change_fraction"], 0.8)

    def test_centroid_train_only_not_test(self):
        center, count = train_centroid(DummyData.train_windows, 1)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(center, torch.tensor([3.0, 5.0])))

    def test_invariant_checker_catches_target_and_numeric_mutation(self):
        original = DummyData.test_windows
        bad = (replace(original[0], targets=torch.tensor([[99.0]])), original[1])
        with self.assertRaises(AssertionError):
            verify_invariants(original, bad, 1)
        bad_events = (replace(original[0].events[0], value=torch.tensor([99.0])),) + original[0].events[1:]
        bad = (replace(original[0], events=bad_events), original[1])
        with self.assertRaises(AssertionError):
            verify_invariants(original, bad, 1)


class PairedAnalysisTests(unittest.TestCase):
    def arrays(self):
        return {"sample_ids": np.arange(6), "entity": np.array(["a"] * 3 + ["b"] * 3),
                "start_time": np.array([0, 1, 2, 0, 1, 2]),
                "squared": np.ones((6, 2)), "counts": np.ones((6, 2))}

    def test_identical_pair_has_exact_zero_interval(self):
        a = self.arrays()
        r = paired_bootstrap(a, dict(a), repetitions=100)
        self.assertEqual(r["delta_mse"], 0)
        self.assertEqual(r["ci95"], [0, 0])

    def test_constant_increase_remains_constant_under_block_resampling(self):
        a = self.arrays()
        b = dict(a, squared=a["squared"] + 0.7)
        r = paired_bootstrap(a, b, repetitions=100, block_length=4)
        self.assertAlmostEqual(r["delta_mse"], 0.7)
        np.testing.assert_allclose(r["ci95"], [0.7, 0.7])

    def test_unpaired_windows_rejected(self):
        a = self.arrays()
        b = dict(a, sample_ids=np.arange(6) + 1)
        with self.assertRaises(AssertionError):
            paired_bootstrap(a, b, repetitions=10)


if __name__ == "__main__":
    unittest.main()
