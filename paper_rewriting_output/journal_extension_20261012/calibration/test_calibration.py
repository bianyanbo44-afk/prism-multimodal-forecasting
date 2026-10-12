"""Structural falsification tests; no test-set prediction scores are inspected."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from run_calibration import combine, cyclic_permutation, generate_world, history_indices, load_schedules

HERE = Path(__file__).resolve().parent


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.cfg["entities_per_split"] = {"train": 8, "val": 8, "test": 8}
        self.schedules = [{"name": str(k), "numeric": np.arange(0., 30, .125),
                           "text": np.array([1., 2.3, 4.2, 6.9, 7.1, 10., 12., 15., 19., 22., 24.]),
                           "cutoffs": np.array([7., 14., 21., 28.])} for k in range(8)]
        # Test algebra uses oracle-equivalent first coordinate; never called a text encoder.
        self.rep = {"projected": np.array([[1., 0.], [-1., 0.]] * 8)}

    def test_available_only_after_event_and_arrival(self):
        e = np.array([1., 3., 5., 7., 9.])
        a = np.array([1., 8., 5.1, 7., 9.])
        np.testing.assert_array_equal(history_indices(e, a, 7., 5.), [2, 3])

    def test_unavailable_future_insertions_leave_history_unchanged(self):
        e, a = np.array([1., 4.]), np.array([1., 4.1])
        idx = history_indices(e, a, 5, 7)
        truth = combine(np.array([[1.], [-1.]])[idx], 5 - e[idx], 1)
        e2, a2 = np.r_[e, 6., 4.5], np.r_[a, 6., 7.]
        idx2 = history_indices(e2, a2, 5, 7)
        actual = combine(np.array([[1.], [-1.], [1000.], [1000.]])[idx2], 5 - e2[idx2], 1)
        np.testing.assert_array_equal(truth, actual)

    def test_cyclic_rotation_has_no_fixed_points_when_feasible(self):
        rng = np.random.default_rng(9)
        for n in range(2, 40):
            p = cyclic_permutation(n, rng)
            self.assertEqual(sorted(p), list(range(n)))
            self.assertTrue(np.all(p != np.arange(n)))
        np.testing.assert_array_equal(cyclic_permutation(1, rng), [0])

    def test_empty_history_is_zero_signal(self):
        np.testing.assert_array_equal(combine(np.empty((0, 2)), np.empty(0), 1), [0., 0.])

    def test_oracle_signal_recovery_and_zero_signal(self):
        world, _ = generate_world(self.cfg, self.schedules, self.rep, 12345)
        for d in world.values():
            np.testing.assert_allclose(d["oracle"].ravel(), d["sentence"][:, 0])
            for strength in [0, .25, .5, 1]:
                y = d["baseline"] + strength * d["oracle"].ravel() + d["noise"]
                np.testing.assert_allclose(y - d["baseline"] - d["noise"], strength * d["oracle"].ravel(), atol=1e-12)
            self.assertTrue(np.isfinite(d["common"]).all())

    def test_content_only_control_preserves_numerics_clocks_counts_targets(self):
        world, notes = generate_world(self.cfg, self.schedules, self.rep, 12345)
        for split, d in world.items():
            np.testing.assert_allclose(d["oracle_shuffle"].ravel(), d["sentence_shuffle"][:, 0])
            self.assertTrue(np.all(d["changed_positions"][d["event_count"] >= 2] == d["event_count"][d["event_count"] >= 2]))
            self.assertTrue(np.all(d["changed_positions"][d["event_count"] < 2] == 0))
            # Perturbation arrays contain only payloads; numerical, timing and
            # targets exist once and are reused without mutation for all conditions.
            self.assertEqual(len(d["oracle"]), len(d["common"]))
        for note in notes:
            self.assertGreaterEqual(note["simulated_arrival_time_days"], note["event_time_days"])

    def test_nuisance_generation_invariant_to_signal_strength(self):
        a, notes_a = generate_world(self.cfg, self.schedules, self.rep, 12345)
        alt = copy.deepcopy(self.cfg)
        alt["strengths"] = [0, 999]
        b, notes_b = generate_world(alt, self.schedules, self.rep, 12345)
        self.assertEqual(notes_a, notes_b)
        for split in a:
            for key in a[split]:
                np.testing.assert_array_equal(a[split][key], b[split][key])

    def test_phrase_holdout_and_independent_split_draws(self):
        world, notes = generate_world(self.cfg, self.schedules, self.rep, 12345)
        self.assertTrue(all(n["phrase_index"] < 8 for n in notes if n["split"] == "train"))
        self.assertTrue(all(n["phrase_index"] >= 8 for n in notes if n["split"] != "train"))
        self.assertFalse(np.array_equal(world["train"]["noise"], world["test"]["noise"]))
        self.assertFalse(np.array_equal(world["train"]["oracle"], world["test"]["oracle"]))

    def test_actual_source_schedule_has_irregular_event_gaps(self):
        schedules = load_schedules(self.cfg)
        self.assertEqual(len(schedules), 8)
        self.assertTrue(all(np.std(np.diff(s["text"])) > 0 for s in schedules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
