from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import torch

from eventmamba.baselines import LastValueBaseline, SetDecayGRUBaseline
from eventmamba.forecasting import ResidualEventForecast
from eventmamba.model import EventMambaConfig
from eventmamba.repohealth import FEATURE_NAMES, RepoHealthData, load_repohealth
from eventmamba.repohealth_training import prepare_repohealth_split
from eventmamba.selective import SelectiveEventMamba


REPOHEALTH_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "Time-IMM"
    / "data"
    / "RepoHealth"
)


@unittest.skipUnless(REPOHEALTH_ROOT.exists(), "RepoHealth sparse checkout is unavailable")
class RepoHealthIntegrationTests(unittest.TestCase):
    data: RepoHealthData

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_repohealth(REPOHEALTH_ROOT, normalization="train_only")

    def test_official_window_and_query_counts(self) -> None:
        window_totals = {
            split: sum(counts[split] for counts in self.data.window_counts.values())
            for split in ("train", "val", "test")
        }
        self.assertEqual(window_totals, {"train": 234, "val": 79, "test": 80})
        self.assertEqual(
            {
                "train": len(self.data.train_samples),
                "val": len(self.data.val_samples),
                "test": len(self.data.test_samples),
            },
            {"train": 4044, "val": 1182, "test": 1471},
        )

    def test_future_information_is_excluded_from_events(self) -> None:
        for sample in itertools.chain(
            self.data.train_samples, self.data.val_samples, self.data.test_samples
        ):
            self.assertTrue(all(event.arrival_time < sample.cutoff for event in sample.events))
            self.assertGreaterEqual(sample.forecast_horizon, 0.0)
            self.assertLess(sample.forecast_horizon, 31.0)
            self.assertEqual(sample.target.numel(), len(FEATURE_NAMES))
            self.assertTrue(torch.isfinite(sample.target).all())

    def test_numeric_variables_remain_simultaneous_event_channels(self) -> None:
        sample = self.data.train_samples[0]
        by_time: dict[float, set[int]] = {}
        for event in sample.events:
            by_time.setdefault(event.arrival_time, set()).add(event.modality)
        expected = set(range(len(FEATURE_NAMES)))
        self.assertTrue(by_time)
        self.assertTrue(all(modalities == expected for modalities in by_time.values()))

    def test_sample_ids_are_disjoint_across_splits(self) -> None:
        split_ids = [
            {sample.sample_id for sample in samples}
            for samples in (
                self.data.train_samples,
                self.data.val_samples,
                self.data.test_samples,
            )
        ]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])

    def test_last_value_baseline_returns_latest_channel_values(self) -> None:
        sample = self.data.train_samples[0]
        baseline = LastValueBaseline(self.data.input_dims)
        prediction = baseline((sample,))[0]
        expected = torch.empty(len(FEATURE_NAMES))
        for modality in range(len(FEATURE_NAMES)):
            latest = max(
                (event for event in sample.events if event.modality == modality),
                key=lambda event: event.arrival_time,
            )
            expected[modality] = latest.value.item()
        self.assertTrue(torch.equal(prediction, expected))

    def test_prepacked_path_matches_direct_models(self) -> None:
        samples = self.data.train_samples[:8]
        models = (
            SelectiveEventMamba(
                EventMambaConfig(
                    input_dims=self.data.input_dims,
                    event_width=8,
                    state_dim=6,
                    readout_width=12,
                    target_dim=len(FEATURE_NAMES),
                    use_reliability=False,
                )
            ),
            SetDecayGRUBaseline(
                input_dims=self.data.input_dims,
                event_width=8,
                hidden_dim=6,
                readout_width=12,
                target_dim=len(FEATURE_NAMES),
            ),
        )
        for model in models:
            model.eval()
            prepared = prepare_repohealth_split(model, samples, self.data.input_dims)
            with torch.no_grad():
                direct = model(samples)
                cached = model.forward_packed(prepared.packed)
            self.assertTrue(torch.allclose(direct, cached, atol=1e-6, rtol=1e-6))

    def test_zero_initialized_residual_equals_last_value(self) -> None:
        samples = self.data.train_samples[:8]
        base = SelectiveEventMamba(
            EventMambaConfig(
                input_dims=self.data.input_dims,
                event_width=8,
                state_dim=6,
                readout_width=12,
                target_dim=len(FEATURE_NAMES),
                use_reliability=False,
            )
        )
        residual = ResidualEventForecast(base, self.data.input_dims).eval()
        persistence = LastValueBaseline(self.data.input_dims).eval()
        with torch.no_grad():
            expected = persistence(samples)
            direct = residual(samples)
            prepared = prepare_repohealth_split(
                residual, samples, self.data.input_dims
            )
            cached = residual.forward_packed(prepared.packed)
        self.assertTrue(torch.equal(direct, expected))
        self.assertTrue(torch.equal(cached, expected))

    def test_train_only_normalization_excludes_late_entity_rows(self) -> None:
        official = load_repohealth(
            REPOHEALTH_ROOT, normalization="official_full_entity"
        )
        expected_full_rows = {
            "facebook_react": 573,
            "huggingface_transformers": 1534,
            "kubernetes_kubernetes": 2068,
            "pytorch_pytorch": 2608,
        }
        for entity, train_scale in self.data.scales.items():
            full_scale = official.scales[entity]
            self.assertLess(train_scale.fitted_rows, full_scale.fitted_rows)
            self.assertLess(train_scale.fit_end_day, full_scale.fit_end_day)
            self.assertEqual(full_scale.fitted_rows, expected_full_rows[entity])


if __name__ == "__main__":
    unittest.main()
