from __future__ import annotations

import itertools
import unittest

import torch

from eventmamba.data import AsyncEvent, AsyncSample
from eventmamba.model import EventMamba, EventMambaConfig
from eventmamba.selective import SelectiveEventMamba


def make_event(
    modality: int,
    value: list[float],
    *,
    event_time: float,
    arrival_time: float,
    event_id: int,
    quality: float = 0.9,
) -> AsyncEvent:
    return AsyncEvent(
        event_time=event_time,
        arrival_time=arrival_time,
        modality=modality,
        value=torch.tensor(value, dtype=torch.float32),
        quality=quality,
        event_id=event_id,
    )


class EventMambaPropertyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.model = EventMamba(EventMambaConfig()).eval()

    def test_semigroup_consistency(self) -> None:
        shared = torch.randn(self.model.config.state_dim)
        private = torch.randn(self.model.num_modalities, self.model.config.state_dim)
        direct_shared = self.model.evolve_shared(shared, 1.75)
        split_shared = self.model.evolve_shared(self.model.evolve_shared(shared, 0.6), 1.15)
        direct_private = self.model.evolve_private(private, 1.75)
        split_private = self.model.evolve_private(self.model.evolve_private(private, 0.6), 1.15)
        self.assertTrue(torch.allclose(direct_shared, split_shared, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(direct_private, split_private, atol=1e-6, rtol=1e-6))

    def test_simultaneous_event_permutation_invariance(self) -> None:
        base_events = [
            make_event(0, [0.2, -0.3], event_time=0.8, arrival_time=1.0, event_id=0),
            make_event(1, [0.7], event_time=0.9, arrival_time=1.0, event_id=1),
            make_event(2, [0.1, 0.4, -0.2], event_time=0.7, arrival_time=1.0, event_id=2),
        ]
        outputs = []
        for permutation in itertools.permutations(base_events):
            renumbered = tuple(
                AsyncEvent(
                    event_time=event.event_time,
                    arrival_time=event.arrival_time,
                    modality=event.modality,
                    value=event.value,
                    quality=event.quality,
                    event_id=index,
                )
                for index, event in enumerate(permutation)
            )
            sample = AsyncSample(
                events=renumbered,
                cutoff=1.0,
                forecast_horizon=0.5,
                target=torch.zeros(2),
                sample_id=0,
            )
            outputs.append(self.model.forward_sample(sample))
        reference = outputs[0]
        for output in outputs[1:]:
            self.assertTrue(torch.allclose(reference, output, atol=1e-6, rtol=1e-6))

    def test_future_arrival_does_not_change_prediction(self) -> None:
        observed = make_event(0, [0.2, -0.3], event_time=0.4, arrival_time=0.5, event_id=0)
        late = make_event(1, [9.0], event_time=0.7, arrival_time=1.5, event_id=1)
        base = AsyncSample(
            events=(observed,), cutoff=1.0, forecast_horizon=0.5, target=torch.zeros(2), sample_id=0
        )
        augmented = AsyncSample(
            events=(observed, late),
            cutoff=1.0,
            forecast_horizon=0.5,
            target=torch.zeros(2),
            sample_id=0,
        )
        self.assertTrue(torch.equal(self.model.forward_sample(base), self.model.forward_sample(augmented)))

    def test_reliability_is_monotone_in_staleness(self) -> None:
        encoded = torch.randn(self.model.config.state_dim)
        fresh = self.model.event_reliability(0, encoded, quality=0.8, age=0.0)
        stale = self.model.event_reliability(0, encoded, quality=0.8, age=2.0)
        self.assertLessEqual(stale.item(), fresh.item())

    def test_selective_model_preserves_simultaneous_permutation_invariance(self) -> None:
        model = SelectiveEventMamba(EventMambaConfig()).eval()
        base_events = [
            make_event(0, [0.2, -0.3], event_time=0.8, arrival_time=1.0, event_id=0),
            make_event(1, [0.7], event_time=0.9, arrival_time=1.0, event_id=1),
            make_event(2, [0.1, 0.4, -0.2], event_time=0.7, arrival_time=1.0, event_id=2),
        ]
        outputs = []
        for permutation in itertools.permutations(base_events):
            sample = AsyncSample(
                events=tuple(
                    AsyncEvent(
                        event_time=event.event_time,
                        arrival_time=event.arrival_time,
                        modality=event.modality,
                        value=event.value,
                        quality=event.quality,
                        event_id=index,
                    )
                    for index, event in enumerate(permutation)
                ),
                cutoff=1.0,
                forecast_horizon=0.5,
                target=torch.zeros(2),
                sample_id=0,
            )
            outputs.append(model.forward_sample(sample))
        for output in outputs[1:]:
            self.assertTrue(torch.allclose(outputs[0], output, atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
