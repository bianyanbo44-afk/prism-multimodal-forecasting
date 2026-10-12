"""Selective physical-time shared/private event state-space model."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .batching import PackedGroupedBatch
from .data import AsyncSample
from .model import EventMamba, EventMambaConfig


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


class SelectiveEventMamba(EventMamba):
    """Event SSM with set-symmetric, state-dependent updates at arrival groups."""

    def __init__(self, config: EventMambaConfig) -> None:
        super().__init__(config)
        state_dim = config.state_dim
        update_dim = 2 * state_dim
        self.shared_update_gate = nn.Linear(update_dim, state_dim)
        self.shared_update_candidate = nn.Linear(update_dim, state_dim)
        self.private_update_gates = nn.ModuleList(
            [nn.Linear(update_dim, state_dim) for _ in range(self.num_modalities)]
        )
        self.private_update_candidates = nn.ModuleList(
            [nn.Linear(update_dim, state_dim) for _ in range(self.num_modalities)]
        )
        self.raw_reliability_strength = nn.Parameter(
            torch.full((self.num_modalities,), -1.0986123)
        )
        nn.init.constant_(self.shared_update_gate.bias, -0.5)
        for gate in self.private_update_gates:
            nn.init.constant_(gate.bias, -0.5)
        self._initialize_multiscale_decay()

    def _initialize_multiscale_decay(self) -> None:
        minimum = self.config.minimum_decay
        rates = torch.logspace(-2.0, 0.0, self.config.state_dim)
        rates = rates.clamp_min(minimum + 1e-6)
        raw_shared = _inverse_softplus(rates - minimum)
        modality_scale = torch.linspace(0.8, 1.2, self.num_modalities).unsqueeze(1)
        private_rates = (modality_scale * rates.unsqueeze(0)).clamp_min(minimum + 1e-6)
        raw_private = _inverse_softplus(private_rates - minimum)
        with torch.no_grad():
            self.raw_shared_decay.copy_(raw_shared)
            self.raw_private_decay.copy_(raw_private)

    def _calibrate_packed_reliability(
        self, modality: int, reliability: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.use_reliability:
            return reliability
        strength = torch.sigmoid(self.raw_reliability_strength[modality])
        return reliability.clamp_min(1e-6).pow(strength)

    @staticmethod
    def _vectorized_private_update(
        private: torch.Tensor,
        message: torch.Tensor,
        active: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor,
        candidate_weight: torch.Tensor,
        candidate_bias: torch.Tensor,
    ) -> torch.Tensor:
        update_input = torch.cat((private, message), dim=2)
        gate = torch.sigmoid(
            torch.einsum("bmi,moi->bmo", update_input, gate_weight)
            + gate_bias.unsqueeze(0)
        )
        candidate = torch.tanh(
            torch.einsum("bmi,moi->bmo", update_input, candidate_weight)
            + candidate_bias.unsqueeze(0)
        )
        proposal = private + gate * (candidate - private)
        return torch.where(active.unsqueeze(2), proposal, private)

    def _encode_packed_states(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_groups = packed.arrival_times.shape
        shared_messages, private_messages = self._packed_messages(packed)
        modality_active = torch.stack(
            [event_mask.any(dim=2) for event_mask in packed.event_masks], dim=2
        )

        shared = self.initial_shared.unsqueeze(0).expand(batch_size, -1)
        private = self.initial_private.unsqueeze(0).expand(batch_size, -1, -1)
        current_time = torch.zeros(
            batch_size, device=self.device, dtype=self.initial_shared.dtype
        )
        private_gate_weight = torch.stack(
            [layer.weight for layer in self.private_update_gates], dim=0
        )
        private_gate_bias = torch.stack(
            [layer.bias for layer in self.private_update_gates], dim=0
        )
        private_candidate_weight = torch.stack(
            [layer.weight for layer in self.private_update_candidates], dim=0
        )
        private_candidate_bias = torch.stack(
            [layer.bias for layer in self.private_update_candidates], dim=0
        )

        for group_index in range(max_groups):
            active = packed.group_mask[:, group_index]
            arrival_time = packed.arrival_times[:, group_index]
            delta_t = torch.where(
                active,
                (arrival_time - current_time).clamp_min(0.0),
                torch.zeros_like(current_time),
            )
            shared = self.evolve_shared(shared, delta_t)
            private = self.evolve_private(private, delta_t)

            if self.config.use_shared_state:
                shared_input = torch.cat(
                    (shared, shared_messages[:, group_index]), dim=1
                )
                shared_gate = torch.sigmoid(self.shared_update_gate(shared_input))
                shared_candidate = torch.tanh(
                    self.shared_update_candidate(shared_input)
                )
                updated_shared = shared + shared_gate * (shared_candidate - shared)
                shared = torch.where(active.unsqueeze(1), updated_shared, shared)

            if self.config.use_private_states:
                private = self._vectorized_private_update(
                    private,
                    private_messages[:, group_index],
                    modality_active[:, group_index],
                    private_gate_weight,
                    private_gate_bias,
                    private_candidate_weight,
                    private_candidate_bias,
                )

            current_time = torch.where(active, arrival_time, current_time)

        return shared, private, current_time

    def _readout_at_queries(
        self,
        shared: torch.Tensor,
        private: torch.Tensor,
        current_time: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        if query_times.ndim != 2 or query_times.shape[0] != shared.shape[0]:
            raise ValueError("query_times must have shape [batch, queries]")
        final_delta = (query_times - current_time.unsqueeze(1)).clamp_min(0.0)
        shared = self.evolve_shared(shared.unsqueeze(1), final_delta)
        private = self.evolve_private(private.unsqueeze(1), final_delta)
        batch_size, query_count = query_times.shape
        if not self.config.use_shared_state:
            shared = torch.zeros_like(shared)
        if not self.config.use_private_states:
            private = torch.zeros_like(private)
        features = torch.cat(
            (shared, private.reshape(batch_size, query_count, -1)), dim=2
        )
        return self.readout(features)

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        shared, private, current_time = self._encode_packed_states(packed)
        return self._readout_at_queries(
            shared, private, current_time, query_times.to(shared)
        )

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        shared, private, current_time = self._encode_packed_states(packed)
        output = self._readout_at_queries(
            shared, private, current_time, packed.query_times.unsqueeze(1)
        )
        return output[:, 0]

    def forward_sample(self, sample: AsyncSample) -> torch.Tensor:
        return EventMamba.forward(self, (sample,))[0]

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        return EventMamba.forward(self, samples)
