"""Forecasting adapters shared across event encoders."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .batching import PackedGroupedBatch, pack_grouped_samples
from .data import AsyncSample


def packed_last_values(
    packed: PackedGroupedBatch,
    input_dims: tuple[int, ...],
    modalities: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Extract the latest scalar value from selected event channels."""

    selected = modalities if modalities is not None else tuple(range(len(input_dims)))
    if not selected:
        raise ValueError("At least one persistence modality is required")
    if any(modality < 0 or modality >= len(input_dims) for modality in selected):
        raise ValueError("Persistence modality index is out of range")
    if any(input_dims[modality] != 1 for modality in selected):
        raise ValueError("Persistence modalities must be scalar event channels")
    batch_size = packed.arrival_times.shape[0]
    outputs: list[torch.Tensor] = []
    for modality in selected:
        mask = packed.event_masks[modality].reshape(batch_size, -1)
        values = packed.values[modality].reshape(batch_size, -1)
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last_position = torch.where(mask, positions, -torch.ones_like(positions)).max(
            dim=1
        ).values
        safe_position = last_position.clamp_min(0)
        latest = values.gather(1, safe_position.unsqueeze(1)).squeeze(1)
        latest = torch.where(last_position >= 0, latest, torch.zeros_like(latest))
        outputs.append(latest)
    return torch.stack(outputs, dim=1)


class ResidualEventForecast(nn.Module):
    """Add a learned event-model correction to the last-value forecast."""

    def __init__(
        self,
        base_model: nn.Module,
        input_dims: tuple[int, ...],
        *,
        zero_initialize_correction: bool = True,
        persistence_modalities: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_dims = input_dims
        self.persistence_modalities = (
            persistence_modalities
            if persistence_modalities is not None
            else tuple(range(len(input_dims)))
        )
        if zero_initialize_correction:
            readout = getattr(base_model, "readout", None)
            if not isinstance(readout, nn.Sequential) or not isinstance(
                readout[-1], nn.Linear
            ):
                raise TypeError("Base model must expose a Sequential readout ending in Linear")
            nn.init.zeros_(readout[-1].weight)
            nn.init.zeros_(readout[-1].bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        correction = self.base_model.forward_packed(packed)
        persistence = packed_last_values(
            packed, self.input_dims, self.persistence_modalities
        ).to(correction)
        if correction.shape != persistence.shape:
            raise ValueError(
                f"Correction shape {tuple(correction.shape)} does not match "
                f"persistence {tuple(persistence.shape)}"
            )
        return persistence + correction

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        correction = self.base_model.forward_packed_queries(packed, query_times)
        persistence = packed_last_values(
            packed, self.input_dims, self.persistence_modalities
        ).to(correction)
        if correction.shape[:1] + correction.shape[2:] != persistence.shape:
            raise ValueError(
                f"Correction shape {tuple(correction.shape)} is incompatible with "
                f"persistence {tuple(persistence.shape)}"
            )
        return persistence.unsqueeze(1) + correction

    def forward(self, samples: Sequence[AsyncSample]) -> torch.Tensor:
        if not samples:
            raise ValueError("At least one sample is required")
        parameter = next(self.parameters())
        packed = pack_grouped_samples(
            samples,
            self.input_dims,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.forward_packed(packed)


class PackedLastValueForecast(nn.Module):
    """Parameter-free persistence forecast for packed multi-query windows."""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        *,
        persistence_modalities: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.persistence_modalities = (
            persistence_modalities
            if persistence_modalities is not None
            else tuple(range(len(input_dims)))
        )
        self.register_buffer("device_anchor", torch.zeros(()))

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        latest = packed_last_values(
            packed, self.input_dims, self.persistence_modalities
        ).to(self.device_anchor)
        return latest.unsqueeze(1).expand(-1, query_times.shape[1], -1)


class PackedTrainMeanForecast(nn.Module):
    """Predict zero in train-normalized space (the per-entity train mean)."""

    def __init__(self, target_dim: int) -> None:
        super().__init__()
        if target_dim < 1:
            raise ValueError("target_dim must be positive")
        self.target_dim = target_dim
        self.register_buffer("device_anchor", torch.zeros(()))

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            query_times.shape[0],
            query_times.shape[1],
            self.target_dim,
            device=self.device_anchor.device,
            dtype=self.device_anchor.dtype,
        )


class PackedShrinkagePersistenceForecast(nn.Module):
    """Shrink latest values toward the train mean with fixed train-fit weights."""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        shrinkage: torch.Tensor,
        *,
        persistence_modalities: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.persistence_modalities = (
            persistence_modalities
            if persistence_modalities is not None
            else tuple(range(len(input_dims)))
        )
        weights = torch.as_tensor(shrinkage, dtype=torch.float32).reshape(-1)
        if weights.numel() != len(self.persistence_modalities):
            raise ValueError("Shrinkage size must match persistence modalities")
        if not torch.isfinite(weights).all() or bool(((weights < 0) | (weights > 1)).any()):
            raise ValueError("Shrinkage weights must be finite and in [0, 1]")
        self.register_buffer("shrinkage", weights)

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        latest = packed_last_values(
            packed, self.input_dims, self.persistence_modalities
        ).to(self.shrinkage)
        prediction = latest * self.shrinkage.unsqueeze(0)
        return prediction.unsqueeze(1).expand(-1, query_times.shape[1], -1)
