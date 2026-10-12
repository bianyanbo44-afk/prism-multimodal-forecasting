"""Run paired robustness conditions and EventMamba ablations across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventmamba.baselines import DecayEventGRUBaseline, EventGRUBaseline
from eventmamba.data import AsyncSample, SyntheticAsyncDataset
from eventmamba.model import EventMamba, EventMambaConfig
from eventmamba.selective import SelectiveEventMamba
from eventmamba.training import append_jsonl, evaluate, set_deterministic_seed, train_model


MODEL_NAMES = (
    "eventgru",
    "eventgru_matched",
    "decay_eventgru_matched",
    "eventmamba_full",
    "eventmamba_no_reliability",
    "eventmamba_no_shared",
    "eventmamba_no_private",
    "selective_eventmamba_full",
    "selective_eventmamba_no_reliability",
    "selective_eventmamba_no_shared",
    "selective_eventmamba_no_private",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    name: str,
    config: EventMambaConfig,
    device: torch.device,
    matched_gru_hidden_dim: int | None = None,
) -> torch.nn.Module:
    if name in {"eventgru", "eventgru_matched", "decay_eventgru_matched"}:
        if name == "eventgru":
            hidden_dim = config.state_dim
        elif matched_gru_hidden_dim is None:
            raise ValueError("matched_gru_hidden_dim is required for matched GRU baselines")
        else:
            hidden_dim = matched_gru_hidden_dim
        baseline_class = (
            DecayEventGRUBaseline if name == "decay_eventgru_matched" else EventGRUBaseline
        )
        return baseline_class(
            input_dims=config.input_dims,
            event_width=config.event_width,
            hidden_dim=hidden_dim,
            readout_width=config.readout_width,
            target_dim=config.target_dim,
        ).to(device)
    variants = {
        "eventmamba_full": config,
        "eventmamba_no_reliability": replace(config, use_reliability=False),
        "eventmamba_no_shared": replace(config, use_shared_state=False),
        "eventmamba_no_private": replace(config, use_private_states=False),
    }
    if name in variants:
        return EventMamba(variants[name]).to(device)
    selective_variants = {
        "selective_eventmamba_full": config,
        "selective_eventmamba_no_reliability": replace(config, use_reliability=False),
        "selective_eventmamba_no_shared": replace(config, use_shared_state=False),
        "selective_eventmamba_no_private": replace(config, use_private_states=False),
    }
    if name in selective_variants:
        return SelectiveEventMamba(selective_variants[name]).to(device)
    raise ValueError(f"Unknown model: {name}")


def dataset_stats(samples: Sequence[AsyncSample]) -> dict[str, float]:
    observed_counts = [
        sum(event.arrival_time <= sample.cutoff + 1e-9 for event in sample.events)
        for sample in samples
    ]
    qualities = [
        event.quality
        for sample in samples
        for event in sample.events
        if event.arrival_time <= sample.cutoff + 1e-9
    ]
    return {
        "mean_observed_events": statistics.fmean(observed_counts),
        "mean_quality": statistics.fmean(qualities) if qualities else 0.0,
    }


def verify_paired_targets(datasets: dict[str, SyntheticAsyncDataset]) -> None:
    reference = next(iter(datasets.values())).samples
    for condition, dataset in datasets.items():
        if len(reference) != len(dataset.samples):
            raise AssertionError(f"Condition {condition} changed test-set length")
        for reference_sample, condition_sample in zip(reference, dataset.samples):
            if not torch.equal(reference_sample.target, condition_sample.target):
                raise AssertionError(
                    f"Condition {condition} changed target for sample {reference_sample.sample_id}"
                )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "condition_metrics.csv"
    if any(path.exists() for path in (metrics_path, summary_path, table_path)):
        raise FileExistsError(f"Refusing to overwrite an existing matrix run in {output_dir}")

    device = choose_device(args.device)
    training = config["training"]
    torch.set_num_threads(int(training.get("torch_threads", 8)))
    seeds = [int(seed) for seed in config["seeds"]]
    selected_models = list(config.get("models", MODEL_NAMES))
    unknown = sorted(set(selected_models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models in configuration: {unknown}")

    data_config = dict(config["data"])
    train_count = int(data_config.pop("train_samples"))
    val_count = int(data_config.pop("val_samples"))
    test_count = int(data_config.pop("test_samples"))
    conditions = dict(config["test_conditions"])
    if "normal" not in conditions:
        raise ValueError("test_conditions must include a normal condition")

    resolved_config = {
        **config,
        "resolved_device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_model_config = EventMambaConfig(**config["model"])
    matched_gru_hidden_dim = (
        int(config["matched_gru_hidden_dim"])
        if "matched_gru_hidden_dim" in config
        else None
    )
    all_runs: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    data_diagnostics: list[dict[str, object]] = []
    matrix_start = time.perf_counter()

    for seed in seeds:
        set_deterministic_seed(seed)
        train_data = SyntheticAsyncDataset(train_count, seed=seed * 100 + 1, **data_config)
        val_data = SyntheticAsyncDataset(val_count, seed=seed * 100 + 2, **data_config)
        test_datasets = {
            condition: SyntheticAsyncDataset(
                test_count,
                seed=seed * 100 + 3,
                **{**data_config, **overrides},
            )
            for condition, overrides in conditions.items()
        }
        verify_paired_targets(test_datasets)
        for condition, dataset in test_datasets.items():
            data_diagnostics.append(
                {"seed": seed, "condition": condition, **dataset_stats(dataset.samples)}
            )

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for model_name in selected_models:
            set_deterministic_seed(seed)
            model = build_model(
                model_name,
                base_model_config,
                device,
                matched_gru_hidden_dim=matched_gru_hidden_dim,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            summary = train_model(
                name=model_name,
                model=model,
                train_samples=train_data.samples,
                val_samples=val_data.samples,
                test_samples=test_datasets["normal"].samples,
                output_dir=seed_dir,
                log_record=lambda record, current_seed=seed, current_model=model_name: (
                    append_jsonl(
                        metrics_path,
                        {"seed": current_seed, "run_model": current_model, **record},
                    ),
                    print(
                        json.dumps(
                            {"seed": current_seed, "run_model": current_model, **record}
                        ),
                        flush=True,
                    ),
                ),
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                grad_clip=float(training["grad_clip"]),
                seed=seed,
            )
            condition_mse = {
                condition: evaluate(
                    model, dataset.samples, int(training["batch_size"])
                )
                for condition, dataset in test_datasets.items()
            }
            peak_memory_mib = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            )
            run_summary = {
                "seed": seed,
                **summary,
                "condition_mse": condition_mse,
                "peak_memory_mib": peak_memory_mib,
            }
            all_runs.append(run_summary)
            for condition, mse in condition_mse.items():
                condition_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "condition": condition,
                        "mse": mse,
                        "best_epoch": summary["best_epoch"],
                        "best_val_mse": summary["best_val_mse"],
                        "parameters": summary["parameters"],
                        "duration_seconds": summary["duration_seconds"],
                        "peak_memory_mib": peak_memory_mib,
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)

    result = {
        "status": "completed",
        "device": str(device),
        "seeds": seeds,
        "models": selected_models,
        "conditions": list(conditions),
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples_per_condition": test_count,
        "runs": all_runs,
        "data_diagnostics": data_diagnostics,
        "total_duration_seconds": time.perf_counter() - matrix_start,
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in result if key != "runs"}), flush=True)


if __name__ == "__main__":
    main()
