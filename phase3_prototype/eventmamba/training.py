"""Deterministic laptop-scale training helpers."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .data import AsyncSample


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _batches(
    samples: Sequence[AsyncSample], batch_size: int, *, shuffle: bool, seed: int
) -> list[list[AsyncSample]]:
    indices = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [
        [samples[index] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def evaluate(model: nn.Module, samples: Sequence[AsyncSample], batch_size: int) -> float:
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    with torch.no_grad():
        for batch in _batches(samples, batch_size, shuffle=False, seed=0):
            prediction = model(batch)
            target = torch.stack([sample.target for sample in batch]).to(prediction)
            total_squared_error += torch.sum((prediction - target) ** 2).item()
            total_values += target.numel()
    return total_squared_error / max(1, total_values)


def train_model(
    *,
    name: str,
    model: nn.Module,
    train_samples: Sequence[AsyncSample],
    val_samples: Sequence[AsyncSample],
    test_samples: Sequence[AsyncSample],
    output_dir: Path,
    log_record: Callable[[dict[str, object]], None],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_val = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    start = time.perf_counter()

    initial_val = evaluate(model, val_samples, batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_values = 0
        for batch in _batches(train_samples, batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            target = torch.stack([sample.target for sample in batch]).to(prediction)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {name} at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient for {name} at epoch {epoch}")
            optimizer.step()
            epoch_loss += torch.sum((prediction.detach() - target) ** 2).item()
            seen_values += target.numel()

        train_mse = epoch_loss / max(1, seen_values)
        val_mse = evaluate(model, val_samples, batch_size)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        record = {
            "model": name,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "elapsed_seconds": time.perf_counter() - start,
        }
        log_record(record)

    model.load_state_dict(best_state)
    test_mse = evaluate(model, test_samples, batch_size)
    checkpoint = output_dir / f"{name}.pt"
    torch.save(
        {
            "model": name,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "test_mse": test_mse,
        },
        checkpoint,
    )
    return {
        "model": name,
        "parameters": parameter_count(model),
        "initial_val_mse": initial_val,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "duration_seconds": time.perf_counter() - start,
        "checkpoint": str(checkpoint),
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

