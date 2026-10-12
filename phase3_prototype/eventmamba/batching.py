"""Packing utilities for vectorized asynchronous event batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .data import AsyncEvent, AsyncSample


def _available_events(sample: AsyncSample) -> list[AsyncEvent]:
    return sorted(
        (event for event in sample.events if event.arrival_time <= sample.cutoff + 1e-9),
        key=lambda event: (event.arrival_time, event.event_id),
    )


@dataclass(frozen=True)
class PackedGroupedBatch:
    arrival_times: torch.Tensor
    group_mask: torch.Tensor
    values: tuple[torch.Tensor, ...]
    qualities: tuple[torch.Tensor, ...]
    ages: tuple[torch.Tensor, ...]
    event_masks: tuple[torch.Tensor, ...]
    query_times: torch.Tensor


@dataclass(frozen=True)
class PackedSequentialBatch:
    arrival_times: torch.Tensor
    event_times: torch.Tensor
    modalities: torch.Tensor
    values: torch.Tensor
    qualities: torch.Tensor
    event_mask: torch.Tensor
    query_times: torch.Tensor


def index_packed_grouped(
    packed: PackedGroupedBatch, indices: Sequence[int] | torch.Tensor
) -> PackedGroupedBatch:
    index = torch.as_tensor(
        indices, dtype=torch.long, device=packed.arrival_times.device
    )
    return PackedGroupedBatch(
        arrival_times=packed.arrival_times.index_select(0, index),
        group_mask=packed.group_mask.index_select(0, index),
        values=tuple(value.index_select(0, index) for value in packed.values),
        qualities=tuple(quality.index_select(0, index) for quality in packed.qualities),
        ages=tuple(age.index_select(0, index) for age in packed.ages),
        event_masks=tuple(mask.index_select(0, index) for mask in packed.event_masks),
        query_times=packed.query_times.index_select(0, index),
    )


def pack_grouped_samples(
    samples: Sequence[AsyncSample],
    input_dims: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedGroupedBatch:
    if not samples:
        raise ValueError("At least one sample is required")

    grouped_samples: list[list[tuple[float, list[AsyncEvent]]]] = []
    for sample in samples:
        groups: list[tuple[float, list[AsyncEvent]]] = []
        for event in _available_events(sample):
            if not groups or event.arrival_time != groups[-1][0]:
                groups.append((event.arrival_time, [event]))
            else:
                groups[-1][1].append(event)
        grouped_samples.append(groups)

    batch_size = len(samples)
    max_groups = max(1, max(len(groups) for groups in grouped_samples))
    arrival_times = torch.zeros(batch_size, max_groups, dtype=dtype)
    group_mask = torch.zeros(batch_size, max_groups, dtype=torch.bool)

    max_events_per_modality = [1 for _ in input_dims]
    for groups in grouped_samples:
        for _, events in groups:
            counts = [0 for _ in input_dims]
            for event in events:
                counts[event.modality] += 1
            for modality, count in enumerate(counts):
                max_events_per_modality[modality] = max(
                    max_events_per_modality[modality], count
                )

    values = tuple(
        torch.zeros(
            batch_size,
            max_groups,
            max_events_per_modality[modality],
            input_dim,
            dtype=dtype,
        )
        for modality, input_dim in enumerate(input_dims)
    )
    qualities = tuple(
        torch.zeros(batch_size, max_groups, max_events_per_modality[modality], dtype=dtype)
        for modality in range(len(input_dims))
    )
    ages = tuple(torch.zeros_like(quality) for quality in qualities)
    event_masks = tuple(
        torch.zeros(batch_size, max_groups, max_events_per_modality[modality], dtype=torch.bool)
        for modality in range(len(input_dims))
    )

    for batch_index, groups in enumerate(grouped_samples):
        for group_index, (arrival_time, events) in enumerate(groups):
            arrival_times[batch_index, group_index] = float(arrival_time)
            group_mask[batch_index, group_index] = True
            offsets = [0 for _ in input_dims]
            for event in events:
                modality = event.modality
                offset = offsets[modality]
                expected_dim = input_dims[modality]
                if event.value.numel() != expected_dim:
                    raise ValueError(
                        f"Modality {modality} expects {expected_dim} values, "
                        f"got {event.value.numel()}"
                    )
                values[modality][batch_index, group_index, offset] = event.value.reshape(-1)
                qualities[modality][batch_index, group_index, offset] = float(event.quality)
                ages[modality][batch_index, group_index, offset] = max(
                    0.0, event.arrival_time - event.event_time
                )
                event_masks[modality][batch_index, group_index, offset] = True
                offsets[modality] += 1

    query_times = torch.tensor(
        [sample.cutoff + sample.forecast_horizon for sample in samples], dtype=dtype
    )
    return PackedGroupedBatch(
        arrival_times=arrival_times.to(device),
        group_mask=group_mask.to(device),
        values=tuple(tensor.to(device) for tensor in values),
        qualities=tuple(tensor.to(device) for tensor in qualities),
        ages=tuple(tensor.to(device) for tensor in ages),
        event_masks=tuple(tensor.to(device) for tensor in event_masks),
        query_times=query_times.to(device),
    )


def pack_sequential_samples(
    samples: Sequence[AsyncSample],
    input_dims: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedSequentialBatch:
    if not samples:
        raise ValueError("At least one sample is required")

    event_lists = [_available_events(sample) for sample in samples]
    batch_size = len(samples)
    max_events = max(1, max(len(events) for events in event_lists))
    max_input_dim = max(input_dims)

    arrival_times = torch.zeros(batch_size, max_events, dtype=dtype)
    event_times = torch.zeros_like(arrival_times)
    modalities = torch.zeros(batch_size, max_events, dtype=torch.long)
    values = torch.zeros(batch_size, max_events, max_input_dim, dtype=dtype)
    qualities = torch.zeros(batch_size, max_events, dtype=dtype)
    event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)

    for batch_index, events in enumerate(event_lists):
        for event_index, event in enumerate(events):
            expected_dim = input_dims[event.modality]
            if event.value.numel() != expected_dim:
                raise ValueError(
                    f"Modality {event.modality} expects {expected_dim} values, "
                    f"got {event.value.numel()}"
                )
            arrival_times[batch_index, event_index] = float(event.arrival_time)
            event_times[batch_index, event_index] = float(event.event_time)
            modalities[batch_index, event_index] = event.modality
            values[batch_index, event_index, :expected_dim] = event.value.reshape(-1)
            qualities[batch_index, event_index] = float(event.quality)
            event_mask[batch_index, event_index] = True

    query_times = torch.tensor(
        [sample.cutoff + sample.forecast_horizon for sample in samples], dtype=dtype
    )
    return PackedSequentialBatch(
        arrival_times=arrival_times.to(device),
        event_times=event_times.to(device),
        modalities=modalities.to(device),
        values=values.to(device),
        qualities=qualities.to(device),
        event_mask=event_mask.to(device),
        query_times=query_times.to(device),
    )
