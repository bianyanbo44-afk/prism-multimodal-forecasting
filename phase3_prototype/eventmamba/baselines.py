"""Compact baselines for the first falsification gate."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .batching import PackedGroupedBatch, pack_grouped_samples, pack_sequential_samples
from .data import AsyncSample, INPUT_DIMS


class EventGRUBaseline(nn.Module):
    """Sequential event GRU with timestamp and staleness features.

    Unlike EventMamba, simultaneous events are consumed sequentially. The
    baseline therefore has no set-update or semigroup guarantee.
    """

    def __init__(
        self,
        input_dims: tuple[int, ...] = INPUT_DIMS,
        event_width: int = 24,
        hidden_dim: int = 16,
        readout_width: int = 32,
        target_dim: int = 2,
        continuous_decay: bool = False,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.num_modalities = len(input_dims)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, event_width),
                    nn.SiLU(),
                    nn.Linear(event_width, hidden_dim),
                    nn.Tanh(),
                )
                for input_dim in input_dims
            ]
        )
        self.event_projection = nn.Linear(hidden_dim + self.num_modalities + 3, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.initial_hidden = nn.Parameter(torch.zeros(hidden_dim))
        if continuous_decay:
            self.raw_hidden_decay = nn.Parameter(torch.full((hidden_dim,), -2.0))
        else:
            self.register_parameter("raw_hidden_decay", None)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + 1, readout_width),
            nn.SiLU(),
            nn.Linear(readout_width, target_dim),
        )

    @property
    def device(self) -> torch.device:
        return self.initial_hidden.device

    def decay_hidden(
        self, hidden: torch.Tensor, delta_t: float | torch.Tensor
    ) -> torch.Tensor:
        if self.raw_hidden_decay is None:
            return hidden
        dt = torch.as_tensor(delta_t, dtype=hidden.dtype, device=hidden.device).clamp_min(0.0)
        while dt.ndim < hidden.ndim:
            dt = dt.unsqueeze(-1)
        decay = torch.nn.functional.softplus(self.raw_hidden_decay).to(hidden)
        return hidden * torch.exp(-decay * dt)

    def forward_sample(self, sample: AsyncSample) -> torch.Tensor:
        hidden = self.initial_hidden
        previous_arrival = 0.0
        events = sorted(
            (event for event in sample.events if event.arrival_time <= sample.cutoff + 1e-9),
            key=lambda event: (event.arrival_time, event.event_id),
        )
        for event in events:
            value = event.value.to(self.device, dtype=hidden.dtype).reshape(-1)
            encoded = self.encoders[event.modality](value)
            one_hot = torch.zeros(self.num_modalities, device=self.device, dtype=hidden.dtype)
            one_hot[event.modality] = 1.0
            gap = max(0.0, event.arrival_time - previous_arrival)
            age = max(0.0, event.arrival_time - event.event_time)
            extras = torch.tensor(
                [gap, age, event.quality], device=self.device, dtype=hidden.dtype
            )
            projected = torch.tanh(self.event_projection(torch.cat((encoded, one_hot, extras))))
            hidden = self.decay_hidden(hidden, gap)
            hidden = self.cell(projected, hidden)
            previous_arrival = event.arrival_time
        horizon = torch.tensor(
            [sample.cutoff + sample.forecast_horizon - previous_arrival],
            device=self.device,
            dtype=hidden.dtype,
        )
        hidden = self.decay_hidden(hidden, horizon)
        return self.readout(torch.cat((hidden, horizon)))

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        if not samples:
            raise ValueError("At least one sample is required")
        packed = pack_sequential_samples(
            samples,
            self.input_dims,
            device=self.device,
            dtype=self.initial_hidden.dtype,
        )
        batch_size, max_events = packed.arrival_times.shape
        hidden = self.initial_hidden.unsqueeze(0).expand(batch_size, -1)
        previous_arrival = torch.zeros(
            batch_size, device=self.device, dtype=self.initial_hidden.dtype
        )

        for event_index in range(max_events):
            active = packed.event_mask[:, event_index]
            modality = packed.modalities[:, event_index]
            encoded = torch.zeros(
                batch_size, hidden.shape[1], device=self.device, dtype=hidden.dtype
            )
            for modality_index, input_dim in enumerate(self.input_dims):
                indices = torch.nonzero(
                    active & (modality == modality_index), as_tuple=False
                ).squeeze(1)
                if indices.numel() == 0:
                    continue
                modality_values = packed.values[indices, event_index, :input_dim]
                modality_encoded = self.encoders[modality_index](modality_values)
                encoded = encoded.index_add(0, indices, modality_encoded)

            active_float = active.to(hidden.dtype)
            one_hot = F.one_hot(modality, num_classes=self.num_modalities).to(hidden.dtype)
            one_hot = one_hot * active_float.unsqueeze(1)
            arrival_time = packed.arrival_times[:, event_index]
            gap = (arrival_time - previous_arrival).clamp_min(0.0) * active_float
            age = (
                arrival_time - packed.event_times[:, event_index]
            ).clamp_min(0.0) * active_float
            quality = packed.qualities[:, event_index] * active_float
            extras = torch.stack((gap, age, quality), dim=1)
            projected = torch.tanh(
                self.event_projection(torch.cat((encoded, one_hot, extras), dim=1))
            )
            hidden = self.decay_hidden(hidden, gap)
            updated_hidden = self.cell(projected, hidden)
            hidden = torch.where(active.unsqueeze(1), updated_hidden, hidden)
            previous_arrival = torch.where(active, arrival_time, previous_arrival)

        horizon = (packed.query_times - previous_arrival).clamp_min(0.0).unsqueeze(1)
        hidden = self.decay_hidden(hidden, horizon)
        return self.readout(torch.cat((hidden, horizon), dim=1))


class DecayEventGRUBaseline(EventGRUBaseline):
    """Event-GRU with learned semigroup-consistent hidden-state decay."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(continuous_decay=True, **kwargs)


class LastValueBaseline(nn.Module):
    """Return the most recent observed value of each event channel."""

    def __init__(self, input_dims: tuple[int, ...]) -> None:
        super().__init__()
        if any(input_dim != 1 for input_dim in input_dims):
            raise ValueError("LastValueBaseline currently expects scalar event channels")
        self.input_dims = input_dims
        self.register_buffer("device_anchor", torch.zeros(()))

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for sample in samples:
            latest = torch.zeros(
                len(self.input_dims),
                dtype=self.device_anchor.dtype,
                device=self.device_anchor.device,
            )
            latest_time = torch.full_like(latest, -torch.inf)
            for event in sample.events:
                if event.arrival_time > sample.cutoff + 1e-9:
                    continue
                if event.arrival_time >= float(latest_time[event.modality]):
                    latest[event.modality] = event.value.reshape(-1)[0].to(latest)
                    latest_time[event.modality] = event.arrival_time
            outputs.append(latest)
        return torch.stack(outputs)


class SetDecayGRUBaseline(nn.Module):
    """Set-symmetric grouped-event GRU with physical-time hidden decay.

    All events that arrive at the same timestamp are encoded and pooled before
    a single GRU update, so results do not depend on arbitrary channel order.
    """

    def __init__(
        self,
        *,
        input_dims: tuple[int, ...],
        event_width: int,
        hidden_dim: int,
        readout_width: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.num_modalities = len(input_dims)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, event_width),
                    nn.SiLU(),
                    nn.Linear(event_width, hidden_dim),
                    nn.Tanh(),
                )
                for input_dim in input_dims
            ]
        )
        self.modality_writes = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in input_dims]
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.group_projection = nn.Linear(hidden_dim + 2, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.initial_hidden = nn.Parameter(torch.zeros(hidden_dim))
        self.raw_hidden_decay = nn.Parameter(torch.full((hidden_dim,), -2.0))
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + 1, readout_width),
            nn.SiLU(),
            nn.Linear(readout_width, target_dim),
        )

    @property
    def device(self) -> torch.device:
        return self.initial_hidden.device

    def decay_hidden(
        self, hidden: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        dt = delta_t.clamp_min(0.0)
        while dt.ndim < hidden.ndim:
            dt = dt.unsqueeze(-1)
        decay = F.softplus(self.raw_hidden_decay).to(hidden)
        return hidden * torch.exp(-decay * dt)

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        packed = pack_grouped_samples(
            samples,
            self.input_dims,
            device=self.device,
            dtype=self.initial_hidden.dtype,
        )
        return self.forward_packed(packed)

    def _encode_packed_hidden(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_groups = packed.arrival_times.shape
        messages = torch.zeros(
            batch_size,
            max_groups,
            self.initial_hidden.numel(),
            device=self.device,
            dtype=self.initial_hidden.dtype,
        )
        active_modalities = torch.zeros(
            batch_size,
            max_groups,
            device=self.device,
            dtype=self.initial_hidden.dtype,
        )
        for modality, input_dim in enumerate(self.input_dims):
            mask = packed.event_masks[modality]
            group_active = mask.any(dim=2)
            active_modalities = active_modalities + group_active.to(
                active_modalities.dtype
            )
            indices = torch.nonzero(mask, as_tuple=False)
            if indices.numel() == 0:
                continue
            encoded = self.encoders[modality](packed.values[modality][mask])
            flat_group = indices[:, 0] * max_groups + indices[:, 1]
            contribution = torch.zeros_like(messages).reshape(
                batch_size * max_groups, -1
            )
            contribution = contribution.index_add(
                0, flat_group, self.modality_writes[modality](encoded)
            )
            messages = messages + contribution.reshape_as(messages)
        divisor = active_modalities.clamp_min(1.0).sqrt().unsqueeze(-1)
        messages = self.message_norm(messages / divisor)

        hidden = self.initial_hidden.unsqueeze(0).expand(batch_size, -1)
        current_time = torch.zeros(
            batch_size, device=self.device, dtype=self.initial_hidden.dtype
        )
        for group_index in range(max_groups):
            active = packed.group_mask[:, group_index]
            arrival = packed.arrival_times[:, group_index]
            gap = torch.where(
                active,
                (arrival - current_time).clamp_min(0.0),
                torch.zeros_like(arrival),
            )
            hidden = self.decay_hidden(hidden, gap)
            coverage = active_modalities[:, group_index] / max(1, self.num_modalities)
            update = torch.tanh(
                self.group_projection(
                    torch.cat(
                        (messages[:, group_index], gap.unsqueeze(1), coverage.unsqueeze(1)),
                        dim=1,
                    )
                )
            )
            proposal = self.cell(update, hidden)
            hidden = torch.where(active.unsqueeze(1), proposal, hidden)
            current_time = torch.where(active, arrival, current_time)

        return hidden, current_time

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        hidden, current_time = self._encode_packed_hidden(packed)
        if query_times.ndim != 2 or query_times.shape[0] != hidden.shape[0]:
            raise ValueError("query_times must have shape [batch, queries]")
        horizon = (
            query_times.to(hidden) - current_time.unsqueeze(1)
        ).clamp_min(0.0)
        query_hidden = self.decay_hidden(hidden.unsqueeze(1), horizon)
        return self.readout(torch.cat((query_hidden, horizon.unsqueeze(2)), dim=2))

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        return self.forward_packed_queries(
            packed, packed.query_times.unsqueeze(1)
        )[:, 0]
