"""Leakage-auditable RepoHealth forecasting adapter.

The upstream Time-IMM protocol uses 31 days of history, a 31 day forecast
window, a 31 day stride, and per-entity chronological 60/20/20 window splits.
This module preserves that windowing while exposing each numeric variable as
its own event stream for the event-state-space models.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .data import AsyncEvent, AsyncSample


SECONDS_PER_DAY = 86400.0
CSV_FIELD_LIMIT_BYTES = 64 * 1024 * 1024
HISTORY_DAYS = 31.0
PREDICTION_DAYS = 31.0
STRIDE_DAYS = 31.0
FEATURE_NAMES = (
    "open_issues",
    "new_issues",
    "closed_issues",
    "avg_age_open_issues_hr",
    "issue_comment_volume",
    "open_prs",
    "new_prs",
    "merged_prs",
    "avg_age_open_prs_hr",
    "pr_comment_volume",
)
NormalizationProtocol = Literal["train_only", "official_full_entity"]


@dataclass(frozen=True)
class EntityScale:
    mean: np.ndarray
    std: np.ndarray
    fitted_rows: int
    fit_end_day: float


@dataclass(frozen=True)
class RepoHealthSampleInfo:
    entity: str
    window_index: int
    query_day: float


@dataclass(frozen=True)
class RepoHealthData:
    train_samples: tuple[AsyncSample, ...]
    val_samples: tuple[AsyncSample, ...]
    test_samples: tuple[AsyncSample, ...]
    sample_info: dict[int, RepoHealthSampleInfo]
    scales: dict[str, EntityScale]
    feature_names: tuple[str, ...]
    window_counts: dict[str, dict[str, int]]
    query_counts: dict[str, dict[str, int]]
    normalization: NormalizationProtocol

    @property
    def input_dims(self) -> tuple[int, ...]:
        return (1,) * len(self.feature_names)


@dataclass(frozen=True)
class _EntityRows:
    name: str
    days: np.ndarray
    values: np.ndarray
    text_days: np.ndarray


@dataclass(frozen=True)
class _Window:
    index: int
    start: float
    history_indices: np.ndarray
    prediction_indices: np.ndarray


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _load_entity(entity_dir: Path) -> _EntityRows:
    with (entity_dir / "time_series.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No numeric rows in {entity_dir}")
    missing = [name for name in FEATURE_NAMES if name not in rows[0]]
    if missing:
        raise ValueError(f"{entity_dir.name}: missing features {missing}")

    timestamps = [_parse_datetime(row["date_time"]) for row in rows]
    order = np.argsort(np.asarray(timestamps, dtype="datetime64[us]"))
    sorted_times = [timestamps[int(index)] for index in order]
    base = sorted_times[0]
    days = np.asarray(
        [(timestamp - base).total_seconds() / SECONDS_PER_DAY for timestamp in sorted_times],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            [float(rows[int(index)][feature]) for feature in FEATURE_NAMES]
            for index in order
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{entity_dir.name}: non-finite numeric values")
    if np.any(np.diff(days) <= 0):
        raise ValueError(f"{entity_dir.name}: numeric timestamps must be unique and sorted")

    csv.field_size_limit(CSV_FIELD_LIMIT_BYTES)
    text_path = entity_dir / "text.csv"
    with text_path.open(encoding="utf-8-sig", newline="") as handle:
        text_rows = list(csv.DictReader(handle))
    text_days = np.asarray(
        [
            (_parse_datetime(row["date_time"]) - base).total_seconds()
            / SECONDS_PER_DAY
            for row in text_rows
            if row.get("date_time")
        ],
        dtype=np.float64,
    )
    text_days.sort()
    return _EntityRows(entity_dir.name, days, values, text_days)


def _make_windows(rows: _EntityRows) -> tuple[_Window, ...]:
    total = HISTORY_DAYS + PREDICTION_DAYS
    start = float(rows.days.min())
    maximum = float(rows.days.max())
    windows: list[_Window] = []
    candidate_index = 0
    while start + total <= maximum:
        in_window = np.flatnonzero((rows.days >= start) & (rows.days < start + total))
        if in_window.size >= 2:
            history = in_window[rows.days[in_window] < start + HISTORY_DAYS]
            prediction = in_window[rows.days[in_window] >= start + HISTORY_DAYS]
            has_text = bool(
                np.any((rows.text_days >= start) & (rows.text_days < start + HISTORY_DAYS))
            )
            if history.size and prediction.size and has_text:
                windows.append(
                    _Window(candidate_index, start, history, prediction)
                )
                candidate_index += 1
        start += STRIDE_DAYS
    if not windows:
        raise RuntimeError(f"No valid windows for {rows.name}")
    return tuple(windows)


def _split_windows(
    windows: tuple[_Window, ...],
) -> dict[str, tuple[_Window, ...]]:
    train_end = int(len(windows) * 0.6)
    val_end = int(len(windows) * 0.8)
    return {
        "train": windows[:train_end],
        "val": windows[train_end:val_end],
        "test": windows[val_end:],
    }


def _fit_scale(
    rows: _EntityRows,
    train_windows: tuple[_Window, ...],
    protocol: NormalizationProtocol,
) -> EntityScale:
    if protocol == "official_full_entity":
        indices = np.arange(len(rows.days), dtype=np.int64)
    elif protocol == "train_only":
        indices = np.unique(
            np.concatenate(
                [
                    np.concatenate((window.history_indices, window.prediction_indices))
                    for window in train_windows
                ]
            )
        )
    else:
        raise ValueError(f"Unknown normalization protocol: {protocol}")
    fit_values = rows.values[indices]
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    return EntityScale(
        mean=mean,
        std=std,
        fitted_rows=int(indices.size),
        fit_end_day=float(rows.days[indices].max()),
    )


def _history_events(
    rows: _EntityRows,
    window: _Window,
    normalized: np.ndarray,
) -> tuple[AsyncEvent, ...]:
    events: list[AsyncEvent] = []
    event_id = 0
    for row_index in window.history_indices:
        relative_day = float(rows.days[row_index] - window.start)
        for modality in range(len(FEATURE_NAMES)):
            events.append(
                AsyncEvent(
                    event_time=relative_day,
                    arrival_time=relative_day,
                    modality=modality,
                    value=torch.tensor(
                        [normalized[row_index, modality]], dtype=torch.float32
                    ),
                    quality=1.0,
                    event_id=event_id,
                )
            )
            event_id += 1
    return tuple(events)


def load_repohealth(
    root: Path,
    *,
    normalization: NormalizationProtocol = "train_only",
) -> RepoHealthData:
    """Load RepoHealth into event-query samples with auditable normalization."""

    processed = root.resolve() / "processed"
    entity_dirs = sorted(path for path in processed.iterdir() if path.is_dir())
    if not entity_dirs:
        raise FileNotFoundError(f"No entity directories under {processed}")

    by_split: dict[str, list[AsyncSample]] = {"train": [], "val": [], "test": []}
    sample_info: dict[int, RepoHealthSampleInfo] = {}
    scales: dict[str, EntityScale] = {}
    window_counts: dict[str, dict[str, int]] = {}
    query_counts: dict[str, dict[str, int]] = {}
    sample_id = 0

    for entity_dir in entity_dirs:
        rows = _load_entity(entity_dir)
        split_windows = _split_windows(_make_windows(rows))
        scale = _fit_scale(rows, split_windows["train"], normalization)
        scales[rows.name] = scale
        normalized = (rows.values - scale.mean) / scale.std
        window_counts[rows.name] = {
            split: len(windows) for split, windows in split_windows.items()
        }
        query_counts[rows.name] = {}

        for split, windows in split_windows.items():
            before = len(by_split[split])
            for window in windows:
                events = _history_events(rows, window, normalized)
                for row_index in window.prediction_indices:
                    query_relative = float(rows.days[row_index] - window.start)
                    target = torch.tensor(normalized[row_index], dtype=torch.float32)
                    sample = AsyncSample(
                        events=events,
                        cutoff=HISTORY_DAYS,
                        forecast_horizon=query_relative - HISTORY_DAYS,
                        target=target,
                        sample_id=sample_id,
                    )
                    by_split[split].append(sample)
                    sample_info[sample_id] = RepoHealthSampleInfo(
                        entity=rows.name,
                        window_index=window.index,
                        query_day=float(rows.days[row_index]),
                    )
                    sample_id += 1
            query_counts[rows.name][split] = len(by_split[split]) - before

    return RepoHealthData(
        train_samples=tuple(by_split["train"]),
        val_samples=tuple(by_split["val"]),
        test_samples=tuple(by_split["test"]),
        sample_info=sample_info,
        scales=scales,
        feature_names=FEATURE_NAMES,
        window_counts=window_counts,
        query_counts=query_counts,
        normalization=normalization,
    )
