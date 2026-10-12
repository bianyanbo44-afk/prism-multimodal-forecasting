from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import numpy as np
import torch

from eventmamba.multimodal import (
    shuffle_text_values_within_entities,
    zero_text_values,
)
from eventmamba.timeimm import (
    DATASET_SPECS,
    GPT2_TEXT_CACHE,
    GPT2_TEXT_CACHE_1024,
    TimeIMMData,
    load_timeimm,
)


DATA_ROOT = Path(__file__).resolve().parents[1] / "external" / "Time-IMM" / "data"


@unittest.skipUnless(
    (DATA_ROOT / "ClusterTrace").exists(), "Time-IMM external sparse checkouts unavailable"
)
class TimeIMMIntegrationTests(unittest.TestCase):
    datasets: dict[str, TimeIMMData]

    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = {
            name: load_timeimm(DATA_ROOT / name, DATASET_SPECS[name])
            for name in (
                "ClusterTrace",
                "CESNET",
                "EPA-Air",
                "FNSPID",
                "RepoHealth",
            )
        }

    def test_official_window_counts_and_text_drops(self) -> None:
        expected = {
            "ClusterTrace": ({"train": 204, "val": 69, "test": 69}, 4, 11),
            "CESNET": ({"train": 23, "val": 11, "test": 12}, 64, 10),
            "EPA-Air": ({"train": 118, "val": 41, "test": 43}, 46, 4),
            "FNSPID": ({"train": 948, "val": 317, "test": 321}, 36, 6),
            "RepoHealth": ({"train": 234, "val": 79, "test": 80}, 0, 10),
        }
        for name, (window_counts, dropped, feature_count) in expected.items():
            data = self.datasets[name]
            totals = {
                split: sum(counts[split] for counts in data.window_counts.values())
                for split in ("train", "val", "test")
            }
            self.assertEqual(totals, window_counts)
            self.assertEqual(sum(data.dropped_no_text.values()), dropped)
            self.assertEqual(len(data.feature_names), feature_count)

    def test_sparse_targets_are_finite_and_masked(self) -> None:
        saw_missing = False
        for data in self.datasets.values():
            for window in itertools.chain(
                data.train_windows, data.val_windows, data.test_windows
            ):
                self.assertEqual(window.targets.shape, window.target_masks.shape)
                self.assertTrue(torch.isfinite(window.targets).all())
                self.assertTrue((window.target_masks.sum(dim=1) > 0).all())
                self.assertTrue((window.query_horizons >= 0).all())
                self.assertTrue(
                    (window.query_horizons < data.spec.prediction + 1e-6).all()
                )
                saw_missing = saw_missing or bool((window.target_masks == 0).any())
        self.assertTrue(saw_missing)

    def test_history_has_no_future_events(self) -> None:
        for data in self.datasets.values():
            for window in itertools.chain(data.train_windows, data.val_windows):
                self.assertTrue(window.events)
                self.assertTrue(
                    all(event.arrival_time < window.cutoff for event in window.events)
                )

    def test_clustertrace_fallback_scales_are_finite(self) -> None:
        data = self.datasets["ClusterTrace"]
        first = data.scales["1c6afd5c227e89b8a29589f4"]
        self.assertTrue((first.observed_counts[3:] == 0).all())
        self.assertTrue(np.isfinite(first.mean).all())
        self.assertTrue(np.isfinite(first.std).all())
        self.assertTrue((first.std > 0).all())

    def test_target_count_manifest_matches_masks(self) -> None:
        for data in self.datasets.values():
            split_windows = {
                "train": data.train_windows,
                "val": data.val_windows,
                "test": data.test_windows,
            }
            for split, windows in split_windows.items():
                manifest = sum(counts[split] for counts in data.target_counts.values())
                observed = sum(int(window.target_masks.sum().item()) for window in windows)
                self.assertEqual(manifest, observed)

    def test_frozen_gpt2_cache_alignment_and_text_events(self) -> None:
        expected_text_occurrences = {
            "ClusterTrace": {"train": 397, "val": 131, "test": 132},
            "CESNET": {"train": 27, "val": 15, "test": 16},
            "EPA-Air": {"train": 574, "val": 191, "test": 176},
            "FNSPID": {"train": 10391, "val": 4192, "test": 5944},
            "RepoHealth": {"train": 7223, "val": 2428, "test": 2479},
        }
        for name, expected in expected_text_occurrences.items():
            data = load_timeimm(
                DATA_ROOT / name,
                DATASET_SPECS[name],
                text_cache_name=(
                    GPT2_TEXT_CACHE_1024 if name == "RepoHealth" else GPT2_TEXT_CACHE
                ),
            )
            self.assertEqual(data.text_embedding_dim, 768)
            self.assertEqual(data.input_dims[-1], 768)
            self.assertEqual(data.text_modality, len(data.feature_names))
            revisions = {
                metadata["resolved_revision"]
                for metadata in data.text_metadata.values()
            }
            self.assertEqual(
                revisions, {"607a30d783dfa663caf39e06633721c8d4cfcd7e"}
            )
            for split, windows in {
                "train": data.train_windows,
                "val": data.val_windows,
                "test": data.test_windows,
            }.items():
                text_events = [
                    event
                    for window in windows
                    for event in window.events
                    if event.modality == data.text_modality
                ]
                self.assertEqual(len(text_events), expected[split])
                self.assertTrue(
                    all(event.value.shape == (768,) for event in text_events)
                )
                self.assertTrue(
                    all(torch.isfinite(event.value).all() for event in text_events)
                )
                self.assertTrue(
                    all(0.0 <= event.arrival_time < windows[0].cutoff for event in text_events)
                )
                self.assertTrue(
                    all(
                        window.targets.shape[1] == len(data.feature_names)
                        for window in windows
                    )
                )

    def test_text_layer_norm_is_per_event_and_finite(self) -> None:
        data = load_timeimm(
            DATA_ROOT / "EPA-Air",
            DATASET_SPECS["EPA-Air"],
            text_cache_name=GPT2_TEXT_CACHE,
            text_preprocessing="layer_norm",
        )
        text_event = next(
            event
            for window in data.train_windows
            for event in window.events
            if event.modality == data.text_modality
        )
        self.assertTrue(torch.isfinite(text_event.value).all())
        self.assertAlmostEqual(float(text_event.value.mean()), 0.0, places=5)
        self.assertAlmostEqual(
            float(text_event.value.square().mean()), 1.0, places=4
        )

    def test_semantic_controls_preserve_text_timing_and_marginals(self) -> None:
        data = load_timeimm(
            DATA_ROOT / "EPA-Air",
            DATASET_SPECS["EPA-Air"],
            text_cache_name=GPT2_TEXT_CACHE,
            text_preprocessing="layer_norm",
        )
        shuffled = shuffle_text_values_within_entities(
            data.train_windows,
            data.window_info,
            int(data.text_modality),
            seed=9101,
        )
        changed = 0
        original_sums: dict[str, torch.Tensor] = {}
        shuffled_sums: dict[str, torch.Tensor] = {}
        for original_window, shuffled_window in zip(
            data.train_windows, shuffled, strict=True
        ):
            entity = data.window_info[original_window.sample_id].entity
            self.assertEqual(len(original_window.events), len(shuffled_window.events))
            for original, permuted in zip(
                original_window.events, shuffled_window.events, strict=True
            ):
                self.assertEqual(original.arrival_time, permuted.arrival_time)
                self.assertEqual(original.modality, permuted.modality)
                if original.modality == data.text_modality:
                    changed += int(not torch.equal(original.value, permuted.value))
                    original_sums[entity] = original_sums.get(
                        entity, torch.zeros_like(original.value, dtype=torch.float64)
                    ) + original.value.double()
                    shuffled_sums[entity] = shuffled_sums.get(
                        entity, torch.zeros_like(permuted.value, dtype=torch.float64)
                    ) + permuted.value.double()
                else:
                    self.assertTrue(torch.equal(original.value, permuted.value))
        self.assertGreater(changed, 0)
        for entity in original_sums:
            self.assertTrue(
                torch.allclose(original_sums[entity], shuffled_sums[entity])
            )
        time_only = zero_text_values(data.train_windows, int(data.text_modality))
        self.assertTrue(
            all(
                torch.count_nonzero(event.value) == 0
                for window in time_only
                for event in window.events
                if event.modality == data.text_modality
            )
        )


if __name__ == "__main__":
    unittest.main()
