"""Window-level multi-query training for sparse Time-IMM datasets."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from .batching import PackedGroupedBatch, index_packed_grouped, pack_grouped_samples
from .data import AsyncSample
from .timeimm import TimeIMMData, TimeIMMWindowSample
from .training import parameter_count


@dataclass(frozen=True)
class PreparedTimeIMMSplit:
    windows: tuple[TimeIMMWindowSample, ...]
    packed: PackedGroupedBatch
    query_times: torch.Tensor
    targets: torch.Tensor
    target_masks: torch.Tensor


@dataclass(frozen=True)
class PreparedTimeIMMData:
    train: PreparedTimeIMMSplit
    val: PreparedTimeIMMSplit
    test: PreparedTimeIMMSplit


def _batch_indices(
    size: int, batch_size: int, *, shuffle: bool, seed: int
) -> list[list[int]]:
    indices = list(range(size))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def prepare_timeimm_split(
    model: nn.Module,
    windows: Sequence[TimeIMMWindowSample],
    input_dims: tuple[int, ...],
    target_dim: int | None = None,
) -> PreparedTimeIMMSplit:
    parameter = next(model.parameters(), None)
    if parameter is None:
        parameter = next(model.buffers(), None)
    if parameter is None:
        raise ValueError("Model must expose a parameter or buffer for device placement")
    window_tuple = tuple(windows)
    if not window_tuple:
        raise ValueError("Cannot prepare an empty Time-IMM split")
    output_dim = target_dim if target_dim is not None else len(input_dims)
    dummy_samples = tuple(
        AsyncSample(
            events=window.events,
            cutoff=window.cutoff,
            forecast_horizon=0.0,
            target=torch.zeros(output_dim),
            sample_id=window.sample_id,
        )
        for window in window_tuple
    )
    packed = pack_grouped_samples(
        dummy_samples,
        input_dims,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    max_queries = max(window.targets.shape[0] for window in window_tuple)
    feature_count = output_dim
    query_times = torch.zeros(
        len(window_tuple), max_queries, dtype=parameter.dtype, device=parameter.device
    )
    targets = torch.zeros(
        len(window_tuple),
        max_queries,
        feature_count,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    target_masks = torch.zeros_like(targets)
    for index, window in enumerate(window_tuple):
        query_count = window.targets.shape[0]
        query_times[index].fill_(window.cutoff)
        query_times[index, :query_count] = (
            window.cutoff + window.query_horizons
        ).to(query_times)
        targets[index, :query_count] = window.targets.to(targets)
        target_masks[index, :query_count] = window.target_masks.to(target_masks)
    return PreparedTimeIMMSplit(
        windows=window_tuple,
        packed=packed,
        query_times=query_times,
        targets=targets,
        target_masks=target_masks,
    )


def prepare_timeimm_data(model: nn.Module, data: TimeIMMData) -> PreparedTimeIMMData:
    return PreparedTimeIMMData(
        train=prepare_timeimm_split(
            model, data.train_windows, data.input_dims, len(data.feature_names)
        ),
        val=prepare_timeimm_split(
            model, data.val_windows, data.input_dims, len(data.feature_names)
        ),
        test=prepare_timeimm_split(
            model, data.test_windows, data.input_dims, len(data.feature_names)
        ),
    )


@torch.no_grad()
def evaluate_timeimm(
    model: nn.Module,
    split: PreparedTimeIMMSplit,
    data: TimeIMMData,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    feature_count = len(data.feature_names)
    squared = torch.zeros(feature_count, dtype=torch.float64)
    absolute = torch.zeros(feature_count, dtype=torch.float64)
    raw_squared = torch.zeros(feature_count, dtype=torch.float64)
    raw_absolute = torch.zeros(feature_count, dtype=torch.float64)
    raw_smape = torch.zeros(feature_count, dtype=torch.float64)
    counts = torch.zeros(feature_count, dtype=torch.float64)
    entity_squared = {
        entity: torch.zeros(feature_count, dtype=torch.float64)
        for entity in data.scales
    }
    entity_counts = {
        entity: torch.zeros(feature_count, dtype=torch.float64)
        for entity in data.scales
    }

    for indices in _batch_indices(len(split.windows), batch_size, shuffle=False, seed=0):
        index = torch.as_tensor(
            indices, dtype=torch.long, device=split.query_times.device
        )
        packed = index_packed_grouped(split.packed, index)
        query_times = split.query_times.index_select(0, index)
        target = split.targets.index_select(0, index)
        mask = split.target_masks.index_select(0, index)
        prediction = model.forward_packed_queries(packed, query_times)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite Time-IMM prediction")
        error = prediction - target
        squared += (error.square() * mask).sum((0, 1)).double().cpu()
        absolute += (error.abs() * mask).sum((0, 1)).double().cpu()
        counts += mask.sum((0, 1)).double().cpu()

        entities = [
            data.window_info[split.windows[row].sample_id].entity for row in indices
        ]
        means = torch.stack(
            [torch.as_tensor(data.scales[entity].mean) for entity in entities]
        ).to(prediction)
        stds = torch.stack(
            [torch.as_tensor(data.scales[entity].std) for entity in entities]
        ).to(prediction)
        raw_prediction = prediction * stds.unsqueeze(1) + means.unsqueeze(1)
        raw_target = target * stds.unsqueeze(1) + means.unsqueeze(1)
        raw_error = raw_prediction - raw_target
        raw_squared += (raw_error.square() * mask).sum((0, 1)).double().cpu()
        raw_absolute += (raw_error.abs() * mask).sum((0, 1)).double().cpu()
        raw_smape += (
            2.0
            * raw_error.abs()
            / (raw_prediction.abs() + raw_target.abs() + 1e-8)
            * mask
        ).sum((0, 1)).double().cpu()

        for row, entity in enumerate(entities):
            row_mask = mask[row].sum(dim=0).double().cpu()
            entity_squared[entity] += (
                error[row].square() * mask[row]
            ).sum(dim=0).double().cpu()
            entity_counts[entity] += row_mask

    available = counts > 0
    feature_mse = squared / counts.clamp_min(1.0)
    feature_mae = absolute / counts.clamp_min(1.0)
    raw_feature_mse = raw_squared / counts.clamp_min(1.0)
    raw_feature_mae = raw_absolute / counts.clamp_min(1.0)
    raw_feature_smape = raw_smape / counts.clamp_min(1.0)
    mse = feature_mse[available].mean().item()
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": feature_mae[available].mean().item(),
        "raw_smape": raw_feature_smape[available].mean().item(),
        "feature_mse": feature_mse.tolist(),
        "feature_mae": feature_mae.tolist(),
        "raw_feature_mse": raw_feature_mse.tolist(),
        "raw_feature_mae": raw_feature_mae.tolist(),
        "entity_mse": {
            entity: (
                entity_squared[entity] / entity_counts[entity].clamp_min(1.0)
            )[entity_counts[entity] > 0].mean().item()
            for entity in entity_squared
        },
        "windows": len(split.windows),
        "query_rows": int(split.target_masks.any(dim=2).sum().item()),
        "observed_values": int(counts.sum().item()),
    }


def train_timeimm_model(
    *,
    name: str,
    model: nn.Module,
    data: TimeIMMData,
    prepared: PreparedTimeIMMData,
    output_dir: Path,
    log_record: Callable[[dict[str, object]], None],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    without_improvement = 0
    start = time.perf_counter()
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()

    initial_val = evaluate_timeimm(model, prepared.val, data, batch_size)
    best_val = float(initial_val["mse"])
    log_record(
        {
            "model": name,
            "seed": seed,
            "epoch": 0,
            "train_mse": None,
            "val_mse": initial_val["mse"],
            "val_mae": initial_val["mae"],
            "elapsed_seconds": time.perf_counter() - start,
        }
    )
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        squared = torch.zeros(len(data.feature_names), dtype=torch.float64)
        counts = torch.zeros(len(data.feature_names), dtype=torch.float64)
        for indices in _batch_indices(
            len(prepared.train.windows),
            batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            index = torch.as_tensor(
                indices, dtype=torch.long, device=prepared.train.query_times.device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_packed_queries(
                index_packed_grouped(prepared.train.packed, index),
                prepared.train.query_times.index_select(0, index),
            )
            target = prepared.train.targets.index_select(0, index)
            mask = prepared.train.target_masks.index_select(0, index)
            batch_squared = (prediction - target).square() * mask
            batch_counts = mask.sum((0, 1))
            available = batch_counts > 0
            loss = (
                batch_squared.sum((0, 1)) / batch_counts.clamp_min(1.0)
            )[available].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {name} at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient for {name} at epoch {epoch}"
                )
            optimizer.step()
            squared += batch_squared.detach().sum((0, 1)).double().cpu()
            counts += batch_counts.detach().double().cpu()

        available = counts > 0
        train_mse = (squared / counts.clamp_min(1.0))[available].mean().item()
        val_metrics = evaluate_timeimm(model, prepared.val, data, batch_size)
        completed_epochs = epoch
        if float(val_metrics["mse"]) < best_val:
            best_val = float(val_metrics["mse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        log_record(
            {
                "model": name,
                "seed": seed,
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        if without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate_timeimm(model, prepared.val, data, batch_size)
    test_metrics = evaluate_timeimm(model, prepared.test, data, batch_size)
    checkpoint = output_dir / f"{name}_seed{seed}.pt"
    torch.save(
        {
            "model": name,
            "seed": seed,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    peak_memory = 0
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        peak_memory = int(torch.cuda.max_memory_allocated())
    return {
        "model": name,
        "seed": seed,
        "parameters": parameter_count(model),
        "initial_val_mse": initial_val["mse"],
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "duration_seconds": time.perf_counter() - start,
        "peak_cuda_memory_bytes": peak_memory,
        "checkpoint": str(checkpoint),
    }
