"""Controlled asynchronous multimodal event data.

The generator exposes independent modality clocks, block missingness, event
dropout, observation noise, delayed arrival, and deliberately simultaneous
events. It is small enough for deterministic unit tests and laptop smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_DIMS = (2, 1, 3)


@dataclass(frozen=True)
class AsyncEvent:
    event_time: float
    arrival_time: float
    modality: int
    value: torch.Tensor
    quality: float
    event_id: int


@dataclass(frozen=True)
class AsyncSample:
    events: tuple[AsyncEvent, ...]
    cutoff: float
    forecast_horizon: float
    target: torch.Tensor
    sample_id: int
    target_mask: torch.Tensor | None = None


def _latent_state(t: float, params: np.ndarray) -> np.ndarray:
    amp_1, amp_2, frequency, phase_1, phase_2, trend = params
    decay_1 = np.exp(-0.035 * t)
    decay_2 = np.exp(-0.020 * t)
    z_1 = amp_1 * decay_1 * np.sin(frequency * t + phase_1)
    z_1 += 0.25 * amp_2 * np.cos(0.55 * frequency * t + phase_2)
    z_2 = amp_2 * decay_2 * np.cos(0.72 * frequency * t + phase_2)
    z_2 += 0.35 * z_1 + trend * (t / 10.0)
    return np.asarray([z_1, z_2], dtype=np.float32)


def _observe(modality: int, latent: np.ndarray) -> np.ndarray:
    z_1, z_2 = latent
    if modality == 0:
        return np.asarray([z_1, z_2], dtype=np.float32)
    if modality == 1:
        return np.asarray([0.8 * z_1 - 0.6 * z_2], dtype=np.float32)
    if modality == 2:
        return np.asarray([z_1 + z_2, z_1 - z_2, z_1 * z_2], dtype=np.float32)
    raise ValueError(f"Unsupported modality: {modality}")


def _poisson_times(rng: np.random.Generator, rate: float, cutoff: float) -> list[float]:
    times: list[float] = []
    current = float(rng.exponential(1.0 / rate))
    while current < cutoff:
        times.append(current)
        current += float(rng.exponential(1.0 / rate))
    return times


def _inside_block(time: float, block: tuple[float, float] | None) -> bool:
    return bool(block is not None and block[0] <= time <= block[1])


def generate_sample(
    sample_id: int,
    seed: int,
    *,
    cutoff: float = 8.0,
    forecast_horizon: float = 1.5,
    rates: Sequence[float] = (1.8, 1.0, 0.55),
    drop_probability: float = 0.12,
    block_missing_probability: float = 0.35,
    delay_scale: float = 0.12,
    noise_multipliers: Sequence[float] = (1.0, 1.0, 1.0),
    simultaneous_groups: int = 2,
    time_resolution: float = 0.1,
) -> AsyncSample:
    if len(rates) != len(INPUT_DIMS):
        raise ValueError(f"Expected {len(INPUT_DIMS)} rates, received {len(rates)}")
    if len(noise_multipliers) != len(INPUT_DIMS):
        raise ValueError(
            f"Expected {len(INPUT_DIMS)} noise multipliers, received {len(noise_multipliers)}"
        )
    if any(float(multiplier) <= 0 for multiplier in noise_multipliers):
        raise ValueError("noise multipliers must be positive")
    if cutoff <= 0 or forecast_horizon <= 0 or time_resolution <= 0:
        raise ValueError("cutoff, forecast_horizon, and time_resolution must be positive")

    rng = np.random.default_rng(seed + sample_id * 1009)
    params = np.asarray(
        [
            rng.uniform(0.7, 1.5),
            rng.uniform(0.7, 1.5),
            rng.uniform(0.55, 1.15),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-0.35, 0.35),
        ],
        dtype=np.float32,
    )

    blocks: list[tuple[float, float] | None] = []
    for _ in INPUT_DIMS:
        if rng.random() < block_missing_probability:
            length = float(rng.uniform(0.8, 2.0))
            start = float(rng.uniform(0.2, max(0.21, cutoff - length)))
            blocks.append((start, min(cutoff, start + length)))
        else:
            blocks.append(None)

    raw_events: list[tuple[float, int, bool]] = []
    for modality, rate in enumerate(rates):
        for time in _poisson_times(rng, float(rate), cutoff):
            snapped = round(time / time_resolution) * time_resolution
            if snapped <= 0 or snapped >= cutoff:
                continue
            if _inside_block(snapped, blocks[modality]) or rng.random() < drop_probability:
                continue
            raw_events.append((float(snapped), modality, False))

    for _ in range(simultaneous_groups):
        anchor = round(float(rng.uniform(0.4, cutoff - 0.4)) / time_resolution) * time_resolution
        selected = rng.choice(len(INPUT_DIMS), size=2, replace=False)
        for modality in selected.tolist():
            if not _inside_block(anchor, blocks[modality]):
                raw_events.append((float(anchor), int(modality), True))

    if not raw_events:
        raw_events.append((cutoff * 0.5, 0, True))

    base_noise_scales = (0.035, 0.055, 0.075)
    events: list[AsyncEvent] = []
    for event_id, (event_time, modality, force_zero_delay) in enumerate(
        sorted(raw_events, key=lambda item: (item[0], item[1], item[2]))
    ):
        latent = _latent_state(event_time, params)
        clean = _observe(modality, latent)
        noise_scale = base_noise_scales[modality] * float(noise_multipliers[modality])
        noise = rng.normal(0.0, noise_scale, size=clean.shape).astype(np.float32)
        value = clean + noise
        quality = float(
            np.exp(-np.mean(np.abs(noise)) / (base_noise_scales[modality] + 1e-8))
        )
        delay = 0.0 if force_zero_delay else float(rng.exponential(delay_scale))
        arrival_time = round((event_time + delay) / time_resolution) * time_resolution
        events.append(
            AsyncEvent(
                event_time=float(event_time),
                arrival_time=float(arrival_time),
                modality=modality,
                value=torch.tensor(value, dtype=torch.float32),
                quality=max(0.0, min(1.0, quality)),
                event_id=event_id,
            )
        )

    events.sort(key=lambda event: (event.arrival_time, event.event_id))
    target_time = cutoff + forecast_horizon
    target = torch.tensor(_latent_state(target_time, params), dtype=torch.float32)
    return AsyncSample(
        events=tuple(events),
        cutoff=float(cutoff),
        forecast_horizon=float(forecast_horizon),
        target=target,
        sample_id=sample_id,
    )


class SyntheticAsyncDataset(Dataset[AsyncSample]):
    def __init__(self, n_samples: int, seed: int, **generator_kwargs: object) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.samples = tuple(
            generate_sample(index, seed, **generator_kwargs) for index in range(n_samples)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AsyncSample:
        return self.samples[index]

    def __iter__(self) -> Iterable[AsyncSample]:
        return iter(self.samples)
