"""Generic leakage-auditable adapter for Time-IMM datasets."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .data import AsyncEvent


CSV_FIELD_LIMIT_BYTES = 64 * 1024 * 1024
UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}
NormalizationProtocol = Literal["train_only", "official_full_entity"]
TextPreprocessing = Literal["none", "layer_norm"]
def gpt2_text_cache_name(max_length: int) -> str:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return (
        "text_embeddings_model=GPT2_pool=masked_mean_"
        f"maxlen={max_length}_timebase=numeric.pt"
    )


GPT2_TEXT_CACHE = gpt2_text_cache_name(320)
GPT2_TEXT_CACHE_1024 = gpt2_text_cache_name(1024)


@dataclass(frozen=True)
class TimeIMMSpec:
    dataset: str
    history: float
    prediction: float
    stride: float
    time_unit: str

    @property
    def total(self) -> float:
        return self.history + self.prediction


DATASET_SPECS = {
    "ClusterTrace": TimeIMMSpec("ClusterTrace", 12.0, 12.0, 12.0, "hours"),
    "CESNET": TimeIMMSpec("CESNET", 7.0, 7.0, 7.0, "days"),
    "EPA-Air": TimeIMMSpec("EPA-Air", 7.0, 7.0, 7.0, "days"),
    "FNSPID": TimeIMMSpec("FNSPID", 31.0, 31.0, 31.0, "days"),
    "RepoHealth": TimeIMMSpec("RepoHealth", 31.0, 31.0, 31.0, "days"),
}


@dataclass(frozen=True)
class TimeIMMScale:
    mean: np.ndarray
    std: np.ndarray
    fitted_rows: int
    observed_counts: np.ndarray
    fit_end_time: float


@dataclass(frozen=True)
class TimeIMMWindowSample:
    events: tuple[AsyncEvent, ...]
    cutoff: float
    query_horizons: torch.Tensor
    targets: torch.Tensor
    target_masks: torch.Tensor
    sample_id: int


@dataclass(frozen=True)
class TimeIMMWindowInfo:
    entity: str
    window_index: int
    start_time: float


@dataclass(frozen=True)
class TimeIMMData:
    train_windows: tuple[TimeIMMWindowSample, ...]
    val_windows: tuple[TimeIMMWindowSample, ...]
    test_windows: tuple[TimeIMMWindowSample, ...]
    window_info: dict[int, TimeIMMWindowInfo]
    scales: dict[str, TimeIMMScale]
    feature_names: tuple[str, ...]
    window_counts: dict[str, dict[str, int]]
    target_counts: dict[str, dict[str, int]]
    dropped_no_text: dict[str, int]
    text_cache_name: str | None
    text_embedding_dim: int | None
    text_metadata: dict[str, dict[str, object]]
    text_preprocessing: TextPreprocessing
    normalization: NormalizationProtocol
    spec: TimeIMMSpec

    @property
    def input_dims(self) -> tuple[int, ...]:
        numeric = (1,) * len(self.feature_names)
        if self.text_embedding_dim is None:
            return numeric
        return numeric + (self.text_embedding_dim,)

    @property
    def text_modality(self) -> int | None:
        if self.text_embedding_dim is None:
            return None
        return len(self.feature_names)


@dataclass(frozen=True)
class _EntityRows:
    name: str
    times: np.ndarray
    values: np.ndarray
    text_times: np.ndarray
    text_embeddings: torch.Tensor | None
    text_metadata: dict[str, object] | None


@dataclass(frozen=True)
class _Window:
    index: int
    start: float
    history_indices: np.ndarray
    prediction_indices: np.ndarray


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _present(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {
        "",
        "nan",
        "na",
        "null",
    }


def _feature_names(entity_dir: Path) -> tuple[str, ...]:
    with (entity_dir / "time_series.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        fields = tuple(csv.DictReader(handle).fieldnames or ())
    return tuple(
        field for field in fields if field not in {"date_time", "record_id"}
    )


def _load_entity(
    entity_dir: Path,
    feature_names: tuple[str, ...],
    seconds_per_unit: float,
    text_cache_name: str | None,
    text_preprocessing: TextPreprocessing,
) -> _EntityRows:
    with (entity_dir / "time_series.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No numeric rows in {entity_dir}")
    timestamps = [_parse_datetime(row["date_time"]) for row in rows]
    order = np.argsort(
        np.asarray(timestamps, dtype="datetime64[us]"), kind="stable"
    )
    sorted_times = [timestamps[int(index)] for index in order]
    base = sorted_times[0]
    relative = np.asarray(
        [
            (timestamp - base).total_seconds() / seconds_per_unit
            for timestamp in sorted_times
        ],
        dtype=np.float64,
    )
    values = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float64)
    for sorted_index, source_index in enumerate(order):
        row = rows[int(source_index)]
        for feature_index, feature in enumerate(feature_names):
            if _present(row.get(feature)):
                values[sorted_index, feature_index] = float(row[feature])
    if np.any(np.diff(relative) < 0):
        raise ValueError(f"{entity_dir.name}: timestamps are not sorted")

    csv.field_size_limit(CSV_FIELD_LIMIT_BYTES)
    with (entity_dir / "text.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        text_columns = [
            column
            for column in (reader.fieldnames or ())
            if column not in {"date_time", "record_id"}
        ]
        if len(text_columns) != 1:
            raise ValueError(
                f"{entity_dir.name}: expected one text column, got {text_columns}"
            )
        text_column = text_columns[0]
        text_rows = [
            (row_index, row)
            for row_index, row in enumerate(reader)
            if row.get("date_time") and _present(row.get(text_column))
        ]
    text_rows.sort(
        key=lambda item: (_parse_datetime(item[1]["date_time"]), item[0])
    )
    text_times = np.asarray(
        [
            (_parse_datetime(row["date_time"]) - base).total_seconds()
            / seconds_per_unit
            for _, row in text_rows
        ],
        dtype=np.float64,
    )
    text_embeddings: torch.Tensor | None = None
    text_metadata: dict[str, object] | None = None
    if text_cache_name is not None:
        cache_path = entity_dir / text_cache_name
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing frozen text cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        text_embeddings = cache["embeddings"].detach().float().contiguous()
        cached_times = cache["relative_times"].double().numpy()
        cached_rows = cache["source_row_indices"].long().tolist()
        text_metadata = dict(cache["metadata"])
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(text_times):
            raise ValueError(
                f"{entity_dir.name}: text cache shape {tuple(text_embeddings.shape)} "
                f"does not match {len(text_times)} nonempty notes"
            )
        if not torch.isfinite(text_embeddings).all():
            raise ValueError(f"{entity_dir.name}: text cache contains non-finite values")
        if cached_times.shape != text_times.shape or not np.allclose(
            cached_times, text_times, rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"{entity_dir.name}: cached text timestamps are misaligned")
        expected_rows = [row_index for row_index, _ in text_rows]
        if cached_rows != expected_rows:
            raise ValueError(f"{entity_dir.name}: cached text source rows are misaligned")
        if text_metadata.get("timestamp_base") != "first numeric timestamp":
            raise ValueError(f"{entity_dir.name}: unexpected text timestamp base")
        if int(text_metadata.get("embedding_dim", -1)) != text_embeddings.shape[1]:
            raise ValueError(f"{entity_dir.name}: text embedding dimension mismatch")
        if text_preprocessing == "layer_norm":
            text_embeddings = torch.nn.functional.layer_norm(
                text_embeddings, (text_embeddings.shape[1],)
            )
        elif text_preprocessing != "none":
            raise ValueError(f"Unknown text preprocessing: {text_preprocessing}")
        if not torch.isfinite(text_embeddings).all():
            raise ValueError(
                f"{entity_dir.name}: text preprocessing produced non-finite values"
            )
    return _EntityRows(
        entity_dir.name,
        relative,
        values,
        text_times,
        text_embeddings,
        text_metadata,
    )


def _make_windows(
    rows: _EntityRows, spec: TimeIMMSpec
) -> tuple[tuple[_Window, ...], int]:
    start = float(rows.times.min())
    maximum = float(rows.times.max())
    windows: list[_Window] = []
    candidate_index = 0
    dropped_no_text = 0
    while start + spec.total <= maximum:
        indices = np.flatnonzero(
            (rows.times >= start) & (rows.times < start + spec.total)
        )
        if indices.size >= 2:
            history = indices[rows.times[indices] < start + spec.history]
            prediction = indices[rows.times[indices] >= start + spec.history]
            history_observed = bool(np.isfinite(rows.values[history]).any())
            prediction_observed = bool(np.isfinite(rows.values[prediction]).any())
            if history_observed and prediction_observed:
                window_index = candidate_index
                candidate_index += 1
                has_text = bool(
                    np.any(
                        (rows.text_times >= start)
                        & (rows.text_times < start + spec.history)
                    )
                )
                if has_text:
                    windows.append(_Window(window_index, start, history, prediction))
                else:
                    dropped_no_text += 1
        start += spec.stride
    if not windows:
        raise RuntimeError(f"No retained windows for {rows.name}")
    return tuple(windows), dropped_no_text


def _split_windows(windows: tuple[_Window, ...]) -> dict[str, tuple[_Window, ...]]:
    train_end = int(len(windows) * 0.6)
    val_end = int(len(windows) * 0.8)
    return {
        "train": windows[:train_end],
        "val": windows[train_end:val_end],
        "test": windows[val_end:],
    }


def _fit_indices(
    rows: _EntityRows,
    train_windows: tuple[_Window, ...],
    protocol: NormalizationProtocol,
) -> np.ndarray:
    if protocol == "official_full_entity":
        return np.arange(len(rows.times), dtype=np.int64)
    if protocol == "train_only":
        return np.unique(
            np.concatenate(
                [
                    np.concatenate((window.history_indices, window.prediction_indices))
                    for window in train_windows
                ]
            )
        )
    raise ValueError(f"Unknown normalization protocol: {protocol}")


def _fit_scale(
    rows: _EntityRows,
    train_windows: tuple[_Window, ...],
    protocol: NormalizationProtocol,
    fallback_mean: np.ndarray,
    fallback_std: np.ndarray,
) -> TimeIMMScale:
    indices = _fit_indices(rows, train_windows, protocol)
    fit_values = rows.values[indices]
    observed_counts = np.isfinite(fit_values).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(fit_values, axis=0)
        std = np.nanstd(fit_values, axis=0, ddof=1)
    mean = np.where(observed_counts > 0, mean, fallback_mean)
    std = np.where(
        (observed_counts > 1) & np.isfinite(std) & (std > 0),
        std,
        fallback_std,
    )
    return TimeIMMScale(
        mean=mean,
        std=std,
        fitted_rows=int(indices.size),
        observed_counts=observed_counts,
        fit_end_time=float(rows.times[indices].max()),
    )


def _history_events(
    rows: _EntityRows,
    window: _Window,
    normalized: np.ndarray,
    history: float,
) -> tuple[AsyncEvent, ...]:
    raw_events: list[tuple[float, int, int, torch.Tensor]] = []
    source_order = 0
    for row_index in window.history_indices:
        relative_time = float(rows.times[row_index] - window.start)
        for modality in np.flatnonzero(np.isfinite(normalized[row_index])):
            raw_events.append(
                (
                    relative_time,
                    int(modality),
                    source_order,
                    torch.tensor(
                        [normalized[row_index, modality]], dtype=torch.float32
                    ),
                )
            )
            source_order += 1
    if rows.text_embeddings is not None:
        selected_text = np.flatnonzero(
            (rows.text_times >= window.start)
            & (rows.text_times < window.start + history)
        )
        text_modality = normalized.shape[1]
        for text_index in selected_text:
            relative_time = float(rows.text_times[text_index] - window.start)
            raw_events.append(
                (
                    relative_time,
                    text_modality,
                    source_order,
                    rows.text_embeddings[int(text_index)],
                )
            )
            source_order += 1
    raw_events.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        AsyncEvent(
            event_time=relative_time,
            arrival_time=relative_time,
            modality=modality,
            value=value,
            quality=1.0,
            event_id=event_id,
        )
        for event_id, (relative_time, modality, _, value) in enumerate(raw_events)
    )


def _window_sample(
    rows: _EntityRows,
    window: _Window,
    normalized: np.ndarray,
    spec: TimeIMMSpec,
    sample_id: int,
) -> TimeIMMWindowSample:
    events = _history_events(rows, window, normalized, spec.history)
    query_horizons: list[float] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    for row_index in window.prediction_indices:
        mask = np.isfinite(normalized[row_index])
        if not mask.any():
            continue
        query_horizons.append(
            float(rows.times[row_index] - window.start - spec.history)
        )
        targets.append(np.where(mask, normalized[row_index], 0.0))
        target_masks.append(mask.astype(np.float32))
    if not targets:
        raise RuntimeError(f"{rows.name} window {window.index}: no target values")
    return TimeIMMWindowSample(
        events=events,
        cutoff=spec.history,
        query_horizons=torch.tensor(query_horizons, dtype=torch.float32),
        targets=torch.tensor(np.asarray(targets), dtype=torch.float32),
        target_masks=torch.tensor(np.asarray(target_masks), dtype=torch.float32),
        sample_id=sample_id,
    )


def load_timeimm(
    root: Path,
    spec: TimeIMMSpec,
    *,
    normalization: NormalizationProtocol = "train_only",
    text_cache_name: str | None = None,
    text_preprocessing: TextPreprocessing = "none",
) -> TimeIMMData:
    if text_cache_name is None and text_preprocessing != "none":
        raise ValueError("Text preprocessing requires a text embedding cache")
    processed = root.resolve() / "processed"
    entity_dirs = sorted(path for path in processed.iterdir() if path.is_dir())
    if not entity_dirs:
        raise FileNotFoundError(f"No entity directories under {processed}")
    feature_names = _feature_names(entity_dirs[0])
    seconds_per_unit = UNIT_SECONDS[spec.time_unit]

    by_split: dict[str, list[TimeIMMWindowSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    window_info: dict[int, TimeIMMWindowInfo] = {}
    scales: dict[str, TimeIMMScale] = {}
    window_counts: dict[str, dict[str, int]] = {}
    target_counts: dict[str, dict[str, int]] = {}
    dropped_by_entity: dict[str, int] = {}
    text_metadata: dict[str, dict[str, object]] = {}
    sample_id = 0

    prepared_entities: list[
        tuple[_EntityRows, dict[str, tuple[_Window, ...]], int]
    ] = []
    for entity_dir in entity_dirs:
        entity_feature_names = _feature_names(entity_dir)
        if (
            len(entity_feature_names) != len(feature_names)
            or set(entity_feature_names) != set(feature_names)
        ):
            raise ValueError(f"{entity_dir.name}: feature columns differ across entities")
        rows = _load_entity(
            entity_dir,
            feature_names,
            seconds_per_unit,
            text_cache_name,
            text_preprocessing,
        )
        windows, dropped = _make_windows(rows, spec)
        split_windows = _split_windows(windows)
        prepared_entities.append((rows, split_windows, dropped))

    text_embedding_dim: int | None = None
    if text_cache_name is not None:
        fingerprints: set[tuple[object, ...]] = set()
        for rows, _, _ in prepared_entities:
            if rows.text_embeddings is None or rows.text_metadata is None:
                raise AssertionError("Text cache was requested but not loaded")
            dimension = int(rows.text_embeddings.shape[1])
            if text_embedding_dim is None:
                text_embedding_dim = dimension
            elif dimension != text_embedding_dim:
                raise ValueError("Text embedding dimensions differ across entities")
            text_metadata[rows.name] = rows.text_metadata
            fingerprints.add(
                (
                    rows.text_metadata.get("model_id"),
                    rows.text_metadata.get("resolved_revision"),
                    rows.text_metadata.get("pooling"),
                    rows.text_metadata.get("max_length"),
                    rows.text_metadata.get("embedding_dim"),
                    rows.text_metadata.get("timestamp_base"),
                )
            )
        if len(fingerprints) != 1:
            raise ValueError("Text cache protocols differ across entities")

    global_fit_values = np.concatenate(
        [
            rows.values[
                _fit_indices(rows, split_windows["train"], normalization)
            ]
            for rows, split_windows, _ in prepared_entities
        ],
        axis=0,
    )
    global_counts = np.isfinite(global_fit_values).sum(axis=0)
    if np.any(global_counts == 0):
        missing = np.flatnonzero(global_counts == 0).tolist()
        raise ValueError(f"No training observations anywhere for features {missing}")
    with np.errstate(invalid="ignore", divide="ignore"):
        global_mean = np.nanmean(global_fit_values, axis=0)
        global_std = np.nanstd(global_fit_values, axis=0, ddof=1)
    global_std = np.where(
        np.isfinite(global_std) & (global_std > 0), global_std, 1.0
    )

    for rows, split_windows, dropped in prepared_entities:
        scale = _fit_scale(
            rows,
            split_windows["train"],
            normalization,
            global_mean,
            global_std,
        )
        normalized = (rows.values - scale.mean) / scale.std
        scales[rows.name] = scale
        dropped_by_entity[rows.name] = dropped
        window_counts[rows.name] = {
            split: len(items) for split, items in split_windows.items()
        }
        target_counts[rows.name] = {}
        for split, items in split_windows.items():
            entity_target_count = 0
            for window in items:
                sample = _window_sample(rows, window, normalized, spec, sample_id)
                by_split[split].append(sample)
                entity_target_count += int(sample.target_masks.sum().item())
                window_info[sample_id] = TimeIMMWindowInfo(
                    entity=rows.name,
                    window_index=window.index,
                    start_time=window.start,
                )
                sample_id += 1
            target_counts[rows.name][split] = entity_target_count

    return TimeIMMData(
        train_windows=tuple(by_split["train"]),
        val_windows=tuple(by_split["val"]),
        test_windows=tuple(by_split["test"]),
        window_info=window_info,
        scales=scales,
        feature_names=feature_names,
        window_counts=window_counts,
        target_counts=target_counts,
        dropped_no_text=dropped_by_entity,
        text_cache_name=text_cache_name,
        text_embedding_dim=text_embedding_dim,
        text_metadata=text_metadata,
        text_preprocessing=text_preprocessing,
        normalization=normalization,
        spec=spec,
    )
