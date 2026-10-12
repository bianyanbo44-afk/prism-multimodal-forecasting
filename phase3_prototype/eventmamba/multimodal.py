"""Multimodal controls for asynchronous numeric-text forecasting."""

from __future__ import annotations

from collections.abc import Sequence
import random

import torch
from torch import nn

from .batching import PackedGroupedBatch
from .data import AsyncEvent
from .forecasting import packed_last_values
from .timeimm import TimeIMMWindowInfo, TimeIMMWindowSample


def select_window_modalities(
    windows: Sequence[TimeIMMWindowSample],
    modalities: tuple[int, ...],
) -> tuple[TimeIMMWindowSample, ...]:
    """Filter and densely remap event modalities while preserving targets."""

    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("Modalities must be a nonempty unique tuple")
    mapping = {source: target for target, source in enumerate(modalities)}
    output: list[TimeIMMWindowSample] = []
    for window in windows:
        selected = [event for event in window.events if event.modality in mapping]
        if not selected:
            raise ValueError(
                f"Window {window.sample_id} has no selected modality events"
            )
        events = tuple(
            AsyncEvent(
                event_time=event.event_time,
                arrival_time=event.arrival_time,
                modality=mapping[event.modality],
                value=event.value,
                quality=event.quality,
                event_id=event_id,
            )
            for event_id, event in enumerate(selected)
        )
        output.append(
            TimeIMMWindowSample(
                events=events,
                cutoff=window.cutoff,
                query_horizons=window.query_horizons,
                targets=window.targets,
                target_masks=window.target_masks,
                sample_id=window.sample_id,
            )
        )
    return tuple(output)


def zero_initialize_readout(model: nn.Module) -> None:
    readout = getattr(model, "readout", None)
    if not isinstance(readout, nn.Sequential) or not isinstance(
        readout[-1], nn.Linear
    ):
        raise TypeError("Model must expose a Sequential readout ending in Linear")
    nn.init.zeros_(readout[-1].weight)
    nn.init.zeros_(readout[-1].bias)


def shuffle_text_values_within_entities(
    windows: Sequence[TimeIMMWindowSample],
    window_info: dict[int, TimeIMMWindowInfo],
    text_modality: int,
    *,
    seed: int,
) -> tuple[TimeIMMWindowSample, ...]:
    """Break text-content alignment but preserve entity, split, time, and count."""

    locations: dict[str, list[tuple[int, int]]] = {}
    values: dict[str, list[torch.Tensor]] = {}
    for window_index, window in enumerate(windows):
        entity = window_info[window.sample_id].entity
        for event_index, event in enumerate(window.events):
            if event.modality == text_modality:
                locations.setdefault(entity, []).append((window_index, event_index))
                values.setdefault(entity, []).append(event.value)
    replacements: dict[tuple[int, int], torch.Tensor] = {}
    generator = random.Random(seed)
    for entity in sorted(locations):
        permutation = list(range(len(values[entity])))
        generator.shuffle(permutation)
        for location, source_index in zip(locations[entity], permutation, strict=True):
            replacements[location] = values[entity][source_index]
    output: list[TimeIMMWindowSample] = []
    for window_index, window in enumerate(windows):
        events = tuple(
            AsyncEvent(
                event_time=event.event_time,
                arrival_time=event.arrival_time,
                modality=event.modality,
                value=replacements.get((window_index, event_index), event.value),
                quality=event.quality,
                event_id=event.event_id,
            )
            for event_index, event in enumerate(window.events)
        )
        output.append(
            TimeIMMWindowSample(
                events=events,
                cutoff=window.cutoff,
                query_horizons=window.query_horizons,
                targets=window.targets,
                target_masks=window.target_masks,
                sample_id=window.sample_id,
            )
        )
    return tuple(output)


def zero_text_values(
    windows: Sequence[TimeIMMWindowSample], text_modality: int
) -> tuple[TimeIMMWindowSample, ...]:
    """Preserve text timestamps/counts while removing all semantic content."""

    output: list[TimeIMMWindowSample] = []
    for window in windows:
        events = tuple(
            AsyncEvent(
                event_time=event.event_time,
                arrival_time=event.arrival_time,
                modality=event.modality,
                value=(
                    torch.zeros_like(event.value)
                    if event.modality == text_modality
                    else event.value
                ),
                quality=event.quality,
                event_id=event.event_id,
            )
            for event in window.events
        )
        output.append(
            TimeIMMWindowSample(
                events=events,
                cutoff=window.cutoff,
                query_horizons=window.query_horizons,
                targets=window.targets,
                target_masks=window.target_masks,
                sample_id=window.sample_id,
            )
        )
    return tuple(output)


class MeanTextFusionBaseline(nn.Module):
    """Naive late fusion of persistence with unordered mean-pooled text.

    This control sees the latest numeric values, the mean text representation,
    log text count, and numeric/text recency. It deliberately does not model the
    order of text events or cross-modal state transitions.
    """

    def __init__(
        self,
        *,
        input_dims: tuple[int, ...],
        numeric_modalities: tuple[int, ...],
        text_modality: int,
        event_width: int,
        text_width: int,
        readout_width: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        if text_modality in numeric_modalities:
            raise ValueError("Text modality cannot be a numeric persistence modality")
        if input_dims[text_modality] <= 1:
            raise ValueError("Text modality must be vector-valued")
        if len(numeric_modalities) != target_dim:
            raise ValueError("Numeric persistence count must match target dimension")
        self.input_dims = input_dims
        self.numeric_modalities = numeric_modalities
        self.text_modality = text_modality
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dims[text_modality], event_width),
            nn.SiLU(),
            nn.Linear(event_width, text_width),
            nn.Tanh(),
        )
        self.text_norm = nn.LayerNorm(text_width)
        self.readout = nn.Sequential(
            nn.Linear(target_dim + text_width + 3, readout_width),
            nn.SiLU(),
            nn.Linear(readout_width, target_dim),
        )
        zero_initialize_readout(self)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _latest_time(
        packed: PackedGroupedBatch, group_active: torch.Tensor
    ) -> torch.Tensor:
        negative = torch.full_like(packed.arrival_times, -torch.inf)
        latest = torch.where(group_active, packed.arrival_times, negative).max(dim=1).values
        return torch.where(torch.isfinite(latest), latest, torch.zeros_like(latest))

    def _text_summary(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = packed.event_masks[self.text_modality]
        batch_size, max_groups, _ = mask.shape
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() == 0:
            width = self.text_norm.normalized_shape[0]
            zeros = torch.zeros(
                batch_size, width, device=self.device, dtype=packed.arrival_times.dtype
            )
            counts = torch.zeros(batch_size, device=self.device, dtype=zeros.dtype)
            return zeros, counts, counts
        encoded = self.text_encoder(packed.values[self.text_modality][mask])
        sums = torch.zeros(
            batch_size, encoded.shape[1], device=encoded.device, dtype=encoded.dtype
        ).index_add(0, indices[:, 0], encoded)
        counts = torch.zeros(
            batch_size, device=encoded.device, dtype=encoded.dtype
        ).index_add(0, indices[:, 0], torch.ones_like(indices[:, 0], dtype=encoded.dtype))
        mean = sums / counts.clamp_min(1.0).unsqueeze(1)
        mean = self.text_norm(mean)
        latest = self._latest_time(packed, mask.any(dim=2))
        return mean, counts, latest

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        query_times = query_times.to(packed.arrival_times)
        persistence = packed_last_values(
            packed, self.input_dims, self.numeric_modalities
        )
        text, counts, latest_text = self._text_summary(packed)
        numeric_active = torch.stack(
            [packed.event_masks[index].any(dim=2) for index in self.numeric_modalities],
            dim=0,
        ).any(dim=0)
        latest_numeric = self._latest_time(packed, numeric_active)
        query_count = query_times.shape[1]
        features = torch.cat(
            (
                persistence.unsqueeze(1).expand(-1, query_count, -1),
                text.unsqueeze(1).expand(-1, query_count, -1),
                counts.log1p().reshape(-1, 1, 1).expand(-1, query_count, -1),
                (query_times - latest_text.unsqueeze(1)).clamp_min(0.0).unsqueeze(2),
                (query_times - latest_numeric.unsqueeze(1)).clamp_min(0.0).unsqueeze(2),
            ),
            dim=2,
        )
        return persistence.unsqueeze(1) + self.readout(features)

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        return self.forward_packed_queries(
            packed, packed.query_times.unsqueeze(1)
        )[:, 0]
