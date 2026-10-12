from __future__ import annotations

import unittest

import torch

from eventmamba.baselines import (
    DecayEventGRUBaseline,
    EventGRUBaseline,
    SetDecayGRUBaseline,
)
from eventmamba.batching import pack_grouped_samples
from eventmamba.data import AsyncEvent, AsyncSample, SyntheticAsyncDataset
from eventmamba.forecasting import (
    PackedShrinkagePersistenceForecast,
    PackedTrainMeanForecast,
    packed_last_values,
)
from eventmamba.immtsf_baseline import IMMTSFRecAvgGRAddBaseline
from eventmamba.model import EventMamba, EventMambaConfig
from eventmamba.multimodal import MeanTextFusionBaseline
from eventmamba.selective import SelectiveEventMamba


class BatchedForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.samples = SyntheticAsyncDataset(7, seed=41).samples

    def test_eventmamba_matches_samplewise_reference(self) -> None:
        model = EventMamba(EventMambaConfig()).eval()
        with torch.no_grad():
            reference = torch.stack([model.forward_sample(sample) for sample in self.samples])
            batched = model(self.samples)
        self.assertTrue(torch.allclose(reference, batched, atol=2e-6, rtol=2e-6))

    def test_eventgru_matches_samplewise_reference(self) -> None:
        model = EventGRUBaseline().eval()
        with torch.no_grad():
            reference = torch.stack([model.forward_sample(sample) for sample in self.samples])
            batched = model(self.samples)
        self.assertTrue(torch.allclose(reference, batched, atol=2e-6, rtol=2e-6))

    def test_decay_eventgru_matches_samplewise_reference(self) -> None:
        model = DecayEventGRUBaseline().eval()
        with torch.no_grad():
            reference = torch.stack([model.forward_sample(sample) for sample in self.samples])
            batched = model(self.samples)
        self.assertTrue(torch.allclose(reference, batched, atol=2e-6, rtol=2e-6))

    def test_batched_eventmamba_backpropagates(self) -> None:
        model = EventMamba(EventMambaConfig())
        prediction = model(self.samples[:4])
        target = torch.stack([sample.target for sample in self.samples[:4]])
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_eventmamba_cuda_batch_matches_cpu_reference(self) -> None:
        cpu_model = EventMamba(EventMambaConfig()).eval()
        gpu_model = EventMamba(EventMambaConfig()).eval().cuda()
        gpu_model.load_state_dict(cpu_model.state_dict())
        with torch.no_grad():
            reference = torch.stack(
                [cpu_model.forward_sample(sample) for sample in self.samples]
            )
            gpu_output = gpu_model(self.samples).cpu()
        self.assertTrue(torch.allclose(reference, gpu_output, atol=3e-5, rtol=3e-5))

    def test_selective_eventmamba_backpropagates(self) -> None:
        model = SelectiveEventMamba(EventMambaConfig())
        prediction = model(self.samples[:4])
        target = torch.stack([sample.target for sample in self.samples[:4]])
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_selective_decay_initialization_spans_time_scales(self) -> None:
        model = SelectiveEventMamba(EventMambaConfig())
        decay = model.shared_decay().detach()
        self.assertGreater(decay.max().item() / decay.min().item(), 50.0)

    def test_vectorized_private_update_matches_modality_loop(self) -> None:
        model = SelectiveEventMamba(EventMambaConfig())
        batch_size = 5
        private = torch.randn(
            batch_size, model.num_modalities, model.config.state_dim
        )
        message = torch.randn_like(private)
        active = torch.rand(batch_size, model.num_modalities) > 0.35
        gate_weight = torch.stack(
            [layer.weight for layer in model.private_update_gates], dim=0
        )
        gate_bias = torch.stack(
            [layer.bias for layer in model.private_update_gates], dim=0
        )
        candidate_weight = torch.stack(
            [layer.weight for layer in model.private_update_candidates], dim=0
        )
        candidate_bias = torch.stack(
            [layer.bias for layer in model.private_update_candidates], dim=0
        )
        vectorized = model._vectorized_private_update(
            private,
            message,
            active,
            gate_weight,
            gate_bias,
            candidate_weight,
            candidate_bias,
        )
        reference = []
        for modality in range(model.num_modalities):
            state = private[:, modality]
            update_input = torch.cat((state, message[:, modality]), dim=1)
            gate = torch.sigmoid(model.private_update_gates[modality](update_input))
            candidate = torch.tanh(
                model.private_update_candidates[modality](update_input)
            )
            proposal = state + gate * (candidate - state)
            reference.append(
                torch.where(active[:, modality].unsqueeze(1), proposal, state)
            )
        reference_tensor = torch.stack(reference, dim=1)
        self.assertTrue(
            torch.allclose(vectorized, reference_tensor, atol=1e-6, rtol=1e-6)
        )

    def test_set_decay_gru_is_simultaneous_permutation_invariant(self) -> None:
        model = SetDecayGRUBaseline(
            input_dims=(2, 1, 3),
            event_width=16,
            hidden_dim=12,
            readout_width=24,
            target_dim=2,
        ).eval()
        sample = self.samples[0]
        simultaneous = {}
        for event in sample.events:
            simultaneous.setdefault(event.arrival_time, []).append(event)
        anchor, group = next(
            (time, events) for time, events in simultaneous.items() if len(events) > 1
        )
        group_ids = {event.event_id for event in group}
        reversed_events = tuple(
            list(reversed(group))
            + [event for event in sample.events if event.event_id not in group_ids]
        )
        permuted = AsyncSample(
            events=reversed_events,
            cutoff=sample.cutoff,
            forecast_horizon=sample.forecast_horizon,
            target=sample.target,
            sample_id=sample.sample_id,
        )
        with torch.no_grad():
            reference = model((sample,))
            output = model((permuted,))
        self.assertTrue(torch.allclose(reference, output, atol=2e-6, rtol=2e-6))

    def test_set_decay_gru_backpropagates(self) -> None:
        model = SetDecayGRUBaseline(
            input_dims=(2, 1, 3),
            event_width=16,
            hidden_dim=12,
            readout_width=24,
            target_dim=2,
        )
        prediction = model(self.samples[:4])
        target = torch.stack([sample.target for sample in self.samples[:4]])
        torch.mean((prediction - target) ** 2).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_multiquery_path_matches_duplicated_histories(self) -> None:
        sample = self.samples[0]
        horizons = (0.4, 0.9, 1.4)
        duplicated = tuple(
            AsyncSample(
                events=sample.events,
                cutoff=sample.cutoff,
                forecast_horizon=horizon,
                target=sample.target,
                sample_id=index,
            )
            for index, horizon in enumerate(horizons)
        )
        models = (
            SelectiveEventMamba(EventMambaConfig()).eval(),
            SetDecayGRUBaseline(
                input_dims=(2, 1, 3),
                event_width=16,
                hidden_dim=12,
                readout_width=24,
                target_dim=2,
            ).eval(),
        )
        query_times = torch.tensor(
            [[sample.cutoff + horizon for horizon in horizons]], dtype=torch.float32
        )
        for model in models:
            packed = pack_grouped_samples(
                (sample,),
                (2, 1, 3),
                device=model.device,
                dtype=next(model.parameters()).dtype,
            )
            with torch.no_grad():
                multiquery = model.forward_packed_queries(packed, query_times)[0]
                repeated = model(duplicated)
            self.assertTrue(
                torch.allclose(multiquery, repeated, atol=2e-6, rtol=2e-6)
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_selective_eventmamba_cuda_forward_backward(self) -> None:
        model = SelectiveEventMamba(EventMambaConfig()).cuda()
        prediction = model(self.samples[:4])
        target = torch.stack([sample.target for sample in self.samples[:4]]).cuda()
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_persistence_can_ignore_vector_text_modality(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([1.0]), 1.0, 0),
                AsyncEvent(0.5, 0.5, 1, torch.tensor([2.0, 3.0, 4.0]), 1.0, 1),
                AsyncEvent(1.0, 1.0, 0, torch.tensor([5.0]), 1.0, 2),
            ),
            cutoff=2.0,
            forecast_horizon=1.0,
            target=torch.tensor([0.0]),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 3), device=torch.device("cpu"), dtype=torch.float32
        )
        latest = packed_last_values(packed, (1, 3), modalities=(0,))
        self.assertTrue(torch.equal(latest, torch.tensor([[5.0]])))

    def test_zero_initialized_mean_text_fusion_equals_numeric_persistence(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([1.0]), 1.0, 0),
                AsyncEvent(0.0, 0.0, 1, torch.tensor([2.0]), 1.0, 1),
                AsyncEvent(0.5, 0.5, 2, torch.tensor([1.0, -1.0, 0.5]), 1.0, 2),
                AsyncEvent(1.0, 1.0, 0, torch.tensor([5.0]), 1.0, 3),
            ),
            cutoff=2.0,
            forecast_horizon=1.0,
            target=torch.zeros(2),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 1, 3), device=torch.device("cpu"), dtype=torch.float32
        )
        model = MeanTextFusionBaseline(
            input_dims=(1, 1, 3),
            numeric_modalities=(0, 1),
            text_modality=2,
            event_width=4,
            text_width=4,
            readout_width=8,
            target_dim=2,
        ).eval()
        prediction = model.forward_packed_queries(packed, torch.tensor([[3.0, 4.0]]))
        expected = torch.tensor([[[5.0, 2.0], [5.0, 2.0]]])
        self.assertTrue(torch.equal(prediction, expected))

    def test_immtsf_explicit_mask_matches_official_recavg_for_real_notes(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([1.0]), 1.0, 0),
                AsyncEvent(0.0, 0.0, 1, torch.tensor([2.0]), 1.0, 1),
                AsyncEvent(0.5, 0.5, 2, torch.randn(768), 1.0, 2),
                AsyncEvent(1.5, 1.5, 2, torch.randn(768), 1.0, 3),
            ),
            cutoff=2.0,
            forecast_horizon=1.0,
            target=torch.zeros(2),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 1, 768), device=torch.device("cpu"), dtype=torch.float32
        )
        model = IMMTSFRecAvgGRAddBaseline(
            input_dims=(1, 1, 768),
            numeric_modalities=(0, 1),
            text_modality=2,
            target_dim=2,
            total_window=4.0,
        ).eval()
        queries = torch.tensor([[2.5, 3.0]])
        notes, tau, note_mask = model._packed_text(packed)
        scaled_tau, scaled_queries = model._scale_times(tau, queries)
        with torch.no_grad():
            adapted, adapted_mask = model._recency_text(
                notes, note_mask, scaled_tau, scaled_queries
            )
            official, official_mask = model.ttf(notes, scaled_tau, scaled_queries)
        self.assertTrue(torch.allclose(adapted, official, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(adapted_mask, official_mask))

    def test_immtsf_zero_content_retains_explicit_text_presence(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([1.0]), 1.0, 0),
                AsyncEvent(0.5, 0.5, 1, torch.zeros(768), 1.0, 1),
            ),
            cutoff=1.0,
            forecast_horizon=1.0,
            target=torch.zeros(1),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 768), device=torch.device("cpu"), dtype=torch.float32
        )
        model = IMMTSFRecAvgGRAddBaseline(
            input_dims=(1, 768),
            numeric_modalities=(0,),
            text_modality=1,
            target_dim=1,
            total_window=2.0,
        ).eval()
        notes, tau, note_mask = model._packed_text(packed)
        scaled_tau, scaled_queries = model._scale_times(tau, torch.tensor([[1.5]]))
        _, presence = model._recency_text(
            notes, note_mask, scaled_tau, scaled_queries
        )
        self.assertTrue(bool(presence.item()))
        self.assertEqual(int(note_mask.sum()), 1)

    def test_immtsf_baseline_backpropagates(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([1.0]), 1.0, 0),
                AsyncEvent(0.0, 0.0, 1, torch.tensor([2.0]), 1.0, 1),
                AsyncEvent(0.5, 0.5, 2, torch.randn(768), 1.0, 2),
            ),
            cutoff=1.0,
            forecast_horizon=1.0,
            target=torch.zeros(2),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 1, 768), device=torch.device("cpu"), dtype=torch.float32
        )
        model = IMMTSFRecAvgGRAddBaseline(
            input_dims=(1, 1, 768),
            numeric_modalities=(0, 1),
            text_modality=2,
            target_dim=2,
            total_window=2.0,
        )
        prediction = model.forward_packed_queries(packed, torch.tensor([[1.5, 2.0]]))
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_train_mean_forecast_is_zero_in_normalized_space(self) -> None:
        model = PackedTrainMeanForecast(target_dim=2)
        sample = self.samples[0]
        packed = pack_grouped_samples(
            (sample,), (2, 1, 3), device=torch.device("cpu"), dtype=torch.float32
        )
        prediction = model.forward_packed_queries(
            packed, torch.tensor([[1.0, 2.0, 3.0]])
        )
        self.assertTrue(torch.equal(prediction, torch.zeros(1, 3, 2)))

    def test_shrinkage_persistence_interpolates_mean_and_latest(self) -> None:
        sample = AsyncSample(
            events=(
                AsyncEvent(0.0, 0.0, 0, torch.tensor([4.0]), 1.0, 0),
                AsyncEvent(0.0, 0.0, 1, torch.tensor([8.0]), 1.0, 1),
            ),
            cutoff=1.0,
            forecast_horizon=1.0,
            target=torch.zeros(2),
            sample_id=0,
        )
        packed = pack_grouped_samples(
            (sample,), (1, 1), device=torch.device("cpu"), dtype=torch.float32
        )
        model = PackedShrinkagePersistenceForecast(
            (1, 1), torch.tensor([0.0, 0.25])
        )
        prediction = model.forward_packed_queries(packed, torch.tensor([[2.0]]))
        self.assertTrue(torch.equal(prediction, torch.tensor([[[0.0, 2.0]]])))


if __name__ == "__main__":
    unittest.main()
