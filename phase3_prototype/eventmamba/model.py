"""Physical-time shared/private event state-space model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .batching import PackedGroupedBatch, pack_grouped_samples
from .data import AsyncEvent, AsyncSample, INPUT_DIMS


@dataclass(frozen=True)
class EventMambaConfig:
    input_dims: tuple[int, ...] = INPUT_DIMS
    event_width: int = 24
    state_dim: int = 16
    readout_width: int = 32
    target_dim: int = 2
    minimum_decay: float = 1e-4
    use_shared_state: bool = True
    use_private_states: bool = True
    use_reliability: bool = True


class EventMamba(nn.Module):
    def __init__(self, config: EventMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.num_modalities = len(config.input_dims)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, config.event_width),
                    nn.SiLU(),
                    nn.Linear(config.event_width, config.state_dim),
                    nn.Tanh(),
                )
                for input_dim in config.input_dims
            ]
        )
        self.shared_writes = nn.ModuleList(
            [nn.Linear(config.state_dim, config.state_dim, bias=False) for _ in config.input_dims]
        )
        self.private_writes = nn.ModuleList(
            [nn.Linear(config.state_dim, config.state_dim, bias=False) for _ in config.input_dims]
        )
        self.content_gates = nn.ModuleList(
            [nn.Linear(config.state_dim + 1, 1) for _ in config.input_dims]
        )

        self.raw_shared_decay = nn.Parameter(torch.full((config.state_dim,), -2.0))
        self.raw_private_decay = nn.Parameter(
            torch.full((self.num_modalities, config.state_dim), -2.0)
        )
        self.raw_staleness = nn.Parameter(torch.zeros(self.num_modalities))
        self.initial_shared = nn.Parameter(torch.zeros(config.state_dim))
        self.initial_private = nn.Parameter(torch.zeros(self.num_modalities, config.state_dim))

        readout_dim = config.state_dim * (1 + self.num_modalities)
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, config.readout_width),
            nn.SiLU(),
            nn.Linear(config.readout_width, config.target_dim),
        )

    @property
    def device(self) -> torch.device:
        return self.initial_shared.device

    def shared_decay(self) -> torch.Tensor:
        return F.softplus(self.raw_shared_decay) + self.config.minimum_decay

    def private_decay(self) -> torch.Tensor:
        return F.softplus(self.raw_private_decay) + self.config.minimum_decay

    def evolve_shared(self, state: torch.Tensor, delta_t: float | torch.Tensor) -> torch.Tensor:
        dt = torch.as_tensor(delta_t, dtype=state.dtype, device=state.device).clamp_min(0.0)
        while dt.ndim < state.ndim:
            dt = dt.unsqueeze(-1)
        return state * torch.exp(-self.shared_decay().to(state) * dt)

    def evolve_private(self, states: torch.Tensor, delta_t: float | torch.Tensor) -> torch.Tensor:
        dt = torch.as_tensor(delta_t, dtype=states.dtype, device=states.device).clamp_min(0.0)
        while dt.ndim < states.ndim:
            dt = dt.unsqueeze(-1)
        return states * torch.exp(-self.private_decay().to(states) * dt)

    def event_reliability(
        self,
        modality: int,
        encoded: torch.Tensor,
        quality: float | torch.Tensor,
        age: float | torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_reliability:
            return torch.ones((), dtype=encoded.dtype, device=encoded.device)
        quality_tensor = torch.as_tensor(
            quality, dtype=encoded.dtype, device=encoded.device
        ).clamp(0.0, 1.0)
        age_tensor = torch.as_tensor(age, dtype=encoded.dtype, device=encoded.device).clamp_min(0.0)
        gate_input = torch.cat((encoded, quality_tensor.reshape(1)), dim=0)
        content_gate = torch.sigmoid(self.content_gates[modality](gate_input)).squeeze(0)
        stale_rate = F.softplus(self.raw_staleness[modality])
        return content_gate * quality_tensor * torch.exp(-stale_rate * age_tensor)

    def _event_messages(self, event: AsyncEvent) -> tuple[torch.Tensor, torch.Tensor]:
        modality = event.modality
        expected_dim = self.config.input_dims[modality]
        if event.value.numel() != expected_dim:
            raise ValueError(
                f"Modality {modality} expects {expected_dim} values, got {event.value.numel()}"
            )
        value = event.value.to(device=self.device, dtype=self.initial_shared.dtype).reshape(-1)
        encoded = self.encoders[modality](value)
        age = max(0.0, event.arrival_time - event.event_time)
        reliability = self.event_reliability(modality, encoded, event.quality, age)
        return (
            reliability * self.shared_writes[modality](encoded),
            reliability * self.private_writes[modality](encoded),
        )

    @staticmethod
    def _available_events(sample: AsyncSample) -> list[AsyncEvent]:
        return sorted(
            (event for event in sample.events if event.arrival_time <= sample.cutoff + 1e-9),
            key=lambda event: (event.arrival_time, event.event_id),
        )

    def forward_sample(self, sample: AsyncSample) -> torch.Tensor:
        shared = self.initial_shared
        private = self.initial_private
        current_time = 0.0

        events = self._available_events(sample)
        for arrival_time, grouped in groupby(events, key=lambda event: event.arrival_time):
            delta_t = max(0.0, float(arrival_time) - current_time)
            shared = self.evolve_shared(shared, delta_t)
            private = self.evolve_private(private, delta_t)

            shared_message = torch.zeros_like(shared)
            private_message = torch.zeros_like(private)
            group = sorted(grouped, key=lambda event: (event.modality, event.event_id))
            for event in group:
                event_shared, event_private = self._event_messages(event)
                shared_message = shared_message + event_shared
                update = torch.zeros_like(private_message)
                update[event.modality] = event_private
                private_message = private_message + update

            if self.config.use_shared_state:
                shared = shared + shared_message
            if self.config.use_private_states:
                private = private + private_message
            current_time = float(arrival_time)

        query_time = sample.cutoff + sample.forecast_horizon
        final_delta = max(0.0, query_time - current_time)
        shared = self.evolve_shared(shared, final_delta)
        private = self.evolve_private(private, final_delta)

        if not self.config.use_shared_state:
            shared = torch.zeros_like(shared)
        if not self.config.use_private_states:
            private = torch.zeros_like(private)
        features = torch.cat((shared, private.reshape(-1)), dim=0)
        return self.readout(features)

    def _calibrate_packed_reliability(
        self, modality: int, reliability: torch.Tensor
    ) -> torch.Tensor:
        return reliability

    def _packed_messages(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_groups = packed.arrival_times.shape
        state_dim = self.config.state_dim
        shared_flat = torch.zeros(
            batch_size * max_groups,
            state_dim,
            device=self.device,
            dtype=self.initial_shared.dtype,
        )
        private_flat = torch.zeros(
            batch_size * max_groups * self.num_modalities,
            state_dim,
            device=self.device,
            dtype=self.initial_shared.dtype,
        )
        for modality in range(self.num_modalities):
            mask = packed.event_masks[modality]
            indices = torch.nonzero(mask, as_tuple=False)
            if indices.numel() == 0:
                continue
            encoded = self.encoders[modality](
                packed.values[modality][mask].reshape(
                    -1, self.config.input_dims[modality]
                )
            )
            if self.config.use_reliability:
                quality = packed.qualities[modality][mask].clamp(0.0, 1.0)
                age = packed.ages[modality][mask].clamp_min(0.0)
                gate_input = torch.cat((encoded, quality.unsqueeze(1)), dim=1)
                content_gate = torch.sigmoid(self.content_gates[modality](gate_input)).squeeze(-1)
                stale_rate = F.softplus(self.raw_staleness[modality])
                reliability = content_gate * quality * torch.exp(-stale_rate * age)
            else:
                reliability = torch.ones(
                    encoded.shape[0], device=encoded.device, dtype=encoded.dtype
                )
            reliability = self._calibrate_packed_reliability(modality, reliability)
            flat_group = indices[:, 0] * max_groups + indices[:, 1]
            shared_flat = shared_flat.index_add(
                0,
                flat_group,
                self.shared_writes[modality](encoded) * reliability.unsqueeze(1),
            )
            private_flat = private_flat.index_add(
                0,
                flat_group * self.num_modalities + modality,
                self.private_writes[modality](encoded) * reliability.unsqueeze(1),
            )

        shared_messages = shared_flat.reshape(batch_size, max_groups, state_dim)
        private_messages = private_flat.reshape(
            batch_size, max_groups, self.num_modalities, state_dim
        )
        return shared_messages, private_messages

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        batch_size, _ = packed.arrival_times.shape
        state_dim = self.config.state_dim
        shared_messages, private_messages = self._packed_messages(packed)
        group_age = (
            packed.query_times.unsqueeze(1) - packed.arrival_times
        ).clamp_min(0.0)
        group_mask = packed.group_mask.to(self.initial_shared.dtype)

        shared_kernel = torch.exp(
            -group_age.unsqueeze(-1) * self.shared_decay().reshape(1, 1, state_dim)
        )
        shared = self.initial_shared.unsqueeze(0) * torch.exp(
            -packed.query_times.unsqueeze(1) * self.shared_decay().reshape(1, state_dim)
        )
        if self.config.use_shared_state:
            shared = shared + (
                shared_messages * shared_kernel * group_mask.unsqueeze(-1)
            ).sum(dim=1)

        private_decay = self.private_decay().reshape(
            1, 1, self.num_modalities, state_dim
        )
        private_kernel = torch.exp(-group_age[:, :, None, None] * private_decay)
        private = self.initial_private.unsqueeze(0) * torch.exp(
            -packed.query_times[:, None, None]
            * self.private_decay().reshape(1, self.num_modalities, state_dim)
        )
        if self.config.use_private_states:
            private = private + (
                private_messages * private_kernel * group_mask[:, :, None, None]
            ).sum(dim=1)

        if not self.config.use_shared_state:
            shared = torch.zeros_like(shared)
        if not self.config.use_private_states:
            private = torch.zeros_like(private)
        features = torch.cat((shared, private.reshape(batch_size, -1)), dim=1)
        return self.readout(features)

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        if not samples:
            raise ValueError("At least one sample is required")
        packed = pack_grouped_samples(
            samples,
            self.config.input_dims,
            device=self.device,
            dtype=self.initial_shared.dtype,
        )
        return self.forward_packed(packed)
