from __future__ import annotations

import unittest

import torch

from eventmamba.data import INPUT_DIMS, SyntheticAsyncDataset


class SyntheticDataTests(unittest.TestCase):
    def test_deterministic_generation(self) -> None:
        first = SyntheticAsyncDataset(3, seed=91)
        second = SyntheticAsyncDataset(3, seed=91)
        self.assertEqual(len(first), len(second))
        for sample_a, sample_b in zip(first, second):
            self.assertEqual(len(sample_a.events), len(sample_b.events))
            self.assertTrue(torch.equal(sample_a.target, sample_b.target))
            for event_a, event_b in zip(sample_a.events, sample_b.events):
                self.assertEqual(event_a.event_time, event_b.event_time)
                self.assertEqual(event_a.arrival_time, event_b.arrival_time)
                self.assertEqual(event_a.modality, event_b.modality)
                self.assertTrue(torch.equal(event_a.value, event_b.value))

    def test_modality_dimensions_and_sorted_arrivals(self) -> None:
        dataset = SyntheticAsyncDataset(8, seed=17)
        for sample in dataset:
            arrivals = [event.arrival_time for event in sample.events]
            self.assertEqual(arrivals, sorted(arrivals))
            for event in sample.events:
                self.assertEqual(event.value.numel(), INPUT_DIMS[event.modality])
                self.assertGreaterEqual(event.arrival_time, event.event_time - 1e-7)
                self.assertGreaterEqual(event.quality, 0.0)
                self.assertLessEqual(event.quality, 1.0)

    def test_noise_conditions_preserve_targets_and_reduce_quality(self) -> None:
        clean = SyntheticAsyncDataset(4, seed=33, noise_multipliers=(1.0, 1.0, 1.0))
        noisy = SyntheticAsyncDataset(4, seed=33, noise_multipliers=(1.0, 1.0, 4.0))
        clean_quality = []
        noisy_quality = []
        for clean_sample, noisy_sample in zip(clean, noisy):
            self.assertTrue(torch.equal(clean_sample.target, noisy_sample.target))
            clean_quality.extend(
                event.quality for event in clean_sample.events if event.modality == 2
            )
            noisy_quality.extend(
                event.quality for event in noisy_sample.events if event.modality == 2
            )
        self.assertEqual(len(clean_quality), len(noisy_quality))
        self.assertLess(sum(noisy_quality) / len(noisy_quality), sum(clean_quality) / len(clean_quality))


if __name__ == "__main__":
    unittest.main()
