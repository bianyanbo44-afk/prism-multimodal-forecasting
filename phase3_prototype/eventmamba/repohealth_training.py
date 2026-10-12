"""Deterministic training and official-style metrics for RepoHealth."""

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
from .repohealth import RepoHealthData
from .training import parameter_count


@dataclass(frozen=True)
class PreparedRepoHealthSplit:
    samples: tuple[AsyncSample, ...]
    packed: PackedGroupedBatch
    targets: torch.Tensor
    target_masks: torch.Tensor


@dataclass(frozen=True)
class PreparedRepoHealthData:
    train: PreparedRepoHealthSplit
    val: PreparedRepoHealthSplit
    test: PreparedRepoHealthSplit


def _batch_indices(
    size: int, batch_size: int, *, shuffle: bool, seed: int
) -> list[list[int]]:
    indices = list(range(size))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]


def prepare_repohealth_split(
    model: nn.Module,
    samples: Sequence[AsyncSample],
    input_dims: tuple[int, ...],
) -> PreparedRepoHealthSplit:
    parameter = next(model.parameters())
    sample_tuple = tuple(samples)
    packed = pack_grouped_samples(
        sample_tuple,
        input_dims,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    targets = torch.stack([sample.target for sample in sample_tuple]).to(parameter)
    target_masks = torch.stack(
        [
            sample.target_mask
            if sample.target_mask is not None
            else torch.ones_like(sample.target)
            for sample in sample_tuple
        ]
    ).to(parameter)
    return PreparedRepoHealthSplit(sample_tuple, packed, targets, target_masks)


def prepare_repohealth_data(
    model: nn.Module, data: RepoHealthData
) -> PreparedRepoHealthData:
    return PreparedRepoHealthData(
        train=prepare_repohealth_split(model, data.train_samples, data.input_dims),
        val=prepare_repohealth_split(model, data.val_samples, data.input_dims),
        test=prepare_repohealth_split(model, data.test_samples, data.input_dims),
    )


@torch.no_grad()
def evaluate_repohealth(
    model: nn.Module,
    samples: Sequence[AsyncSample],
    data: RepoHealthData,
    batch_size: int,
    *,
    prepared: PreparedRepoHealthSplit | None = None,
) -> dict[str, object]:
    """Compute per-variable errors first, then average across variables."""

    model.eval()
    feature_count = len(data.feature_names)
    squared = torch.zeros(feature_count, dtype=torch.float64)
    absolute = torch.zeros(feature_count, dtype=torch.float64)
    raw_squared = torch.zeros(feature_count, dtype=torch.float64)
    raw_absolute = torch.zeros(feature_count, dtype=torch.float64)
    raw_smape = torch.zeros(feature_count, dtype=torch.float64)
    counts = torch.zeros(feature_count, dtype=torch.float64)
    entity_squared: dict[str, torch.Tensor] = {
        entity: torch.zeros(feature_count, dtype=torch.float64)
        for entity in data.scales
    }
    entity_counts = {
        entity: torch.zeros(feature_count, dtype=torch.float64)
        for entity in data.scales
    }

    if prepared is not None and len(prepared.samples) != len(samples):
        raise ValueError("Prepared split length does not match samples")
    for indices in _batch_indices(
        len(samples), batch_size, shuffle=False, seed=0
    ):
        batch = [samples[index] for index in indices]
        if prepared is None:
            prediction = model(batch)
            target = torch.stack([sample.target for sample in batch]).to(prediction)
            target_mask = torch.stack(
                [
                    sample.target_mask
                    if sample.target_mask is not None
                    else torch.ones_like(sample.target)
                    for sample in batch
                ]
            ).to(prediction)
        else:
            prediction = model.forward_packed(
                index_packed_grouped(prepared.packed, indices)
            )
            index = torch.as_tensor(
                indices, dtype=torch.long, device=prepared.targets.device
            )
            target = prepared.targets.index_select(0, index).to(prediction)
            target_mask = prepared.target_masks.index_select(0, index).to(prediction)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite RepoHealth prediction")
        if target_mask.shape != target.shape:
            raise ValueError("Target mask shape does not match target")
        mask = target_mask.clamp(0.0, 1.0)
        error = prediction - target
        squared += (error.square() * mask).sum(dim=0).double().cpu()
        absolute += (error.abs() * mask).sum(dim=0).double().cpu()
        counts += mask.sum(dim=0).double().cpu()

        means = torch.stack(
            [
                torch.as_tensor(
                    data.scales[data.sample_info[sample.sample_id].entity].mean,
                    dtype=prediction.dtype,
                    device=prediction.device,
                )
                for sample in batch
            ]
        )
        stds = torch.stack(
            [
                torch.as_tensor(
                    data.scales[data.sample_info[sample.sample_id].entity].std,
                    dtype=prediction.dtype,
                    device=prediction.device,
                )
                for sample in batch
            ]
        )
        raw_prediction = prediction * stds + means
        raw_target = target * stds + means
        raw_error = raw_prediction - raw_target
        raw_squared += (raw_error.square() * mask).sum(dim=0).double().cpu()
        raw_absolute += (raw_error.abs() * mask).sum(dim=0).double().cpu()
        raw_smape += (
            2.0
            * raw_error.abs()
            / (raw_prediction.abs() + raw_target.abs() + 1e-8)
            * mask
        ).sum(dim=0).double().cpu()

        for row, sample in enumerate(batch):
            entity = data.sample_info[sample.sample_id].entity
            row_mask = mask[row].double().cpu()
            entity_squared[entity] += error[row].square().double().cpu() * row_mask
            entity_counts[entity] += row_mask

    if not samples:
        raise ValueError("Cannot evaluate an empty RepoHealth split")
    feature_mse = squared / counts.clamp_min(1.0)
    feature_mae = absolute / counts.clamp_min(1.0)
    raw_feature_mse = raw_squared / counts.clamp_min(1.0)
    raw_feature_mae = raw_absolute / counts.clamp_min(1.0)
    raw_feature_smape = raw_smape / counts.clamp_min(1.0)
    available = counts > 0
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
        "queries": len(samples),
        "observed_values": int(counts.sum().item()),
    }


def train_repohealth_model(
    *,
    name: str,
    model: nn.Module,
    data: RepoHealthData,
    output_dir: Path,
    log_record: Callable[[dict[str, object]], None],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    prepared_data: PreparedRepoHealthData | None = None,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_val = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start = time.perf_counter()
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()

    if prepared_data is None:
        prepared_data = prepare_repohealth_data(model, data)
    prepared_train = prepared_data.train
    prepared_val = prepared_data.val
    prepared_test = prepared_data.test

    initial_val = evaluate_repohealth(
        model, data.val_samples, data, batch_size, prepared=prepared_val
    )
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
        squared_error = torch.zeros(len(data.feature_names), dtype=torch.float64)
        value_count = torch.zeros(len(data.feature_names), dtype=torch.float64)
        for indices in _batch_indices(
            len(data.train_samples), batch_size, shuffle=True, seed=seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_packed(
                index_packed_grouped(prepared_train.packed, indices)
            )
            index = torch.as_tensor(
                indices, dtype=torch.long, device=prepared_train.targets.device
            )
            target = prepared_train.targets.index_select(0, index).to(prediction)
            target_mask = prepared_train.target_masks.index_select(0, index).to(
                prediction
            )
            batch_squared = (prediction - target).square() * target_mask
            batch_counts = target_mask.sum(dim=0)
            available = batch_counts > 0
            loss = (
                batch_squared.sum(dim=0) / batch_counts.clamp_min(1.0)
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
            squared_error += batch_squared.detach().sum(dim=0).double().cpu()
            value_count += batch_counts.detach().double().cpu()

        val_metrics = evaluate_repohealth(
            model,
            data.val_samples,
            data,
            batch_size,
            prepared=prepared_val,
        )
        train_available = value_count > 0
        train_mse = (
            squared_error / value_count.clamp_min(1.0)
        )[train_available].mean().item()
        completed_epochs = epoch
        if float(val_metrics["mse"]) < best_val:
            best_val = float(val_metrics["mse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
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
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate_repohealth(
        model, data.val_samples, data, batch_size, prepared=prepared_val
    )
    test_metrics = evaluate_repohealth(
        model, data.test_samples, data, batch_size, prepared=prepared_test
    )
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
