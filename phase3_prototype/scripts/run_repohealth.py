"""Run leakage-audited RepoHealth numerical forecasting experiments."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.baselines import LastValueBaseline, SetDecayGRUBaseline
from eventmamba.forecasting import ResidualEventForecast
from eventmamba.model import EventMambaConfig
from eventmamba.repohealth import RepoHealthData, load_repohealth
from eventmamba.repohealth_training import (
    evaluate_repohealth,
    prepare_repohealth_data,
    train_repohealth_model,
)
from eventmamba.selective import SelectiveEventMamba
from eventmamba.training import append_jsonl, parameter_count, set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def make_selective(
    data: RepoHealthData, config: dict[str, object], device: torch.device
) -> ResidualEventForecast:
    model_config = EventMambaConfig(
        input_dims=data.input_dims,
        event_width=int(config["event_width"]),
        state_dim=int(config["state_dim"]),
        readout_width=int(config["readout_width"]),
        target_dim=len(data.feature_names),
        use_reliability=False,
    )
    return ResidualEventForecast(
        SelectiveEventMamba(model_config), data.input_dims
    ).to(device)


def closest_gru_hidden_dim(
    target_parameters: int,
    data: RepoHealthData,
    model_config: dict[str, object],
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for hidden_dim in range(8, 257):
        model = SetDecayGRUBaseline(
            input_dims=data.input_dims,
            event_width=int(model_config["event_width"]),
            hidden_dim=hidden_dim,
            readout_width=int(model_config["readout_width"]),
            target_dim=len(data.feature_names),
        )
        candidates.append((abs(parameter_count(model) - target_parameters), hidden_dim))
    _, hidden_dim = min(candidates)
    matched = SetDecayGRUBaseline(
        input_dims=data.input_dims,
        event_width=int(model_config["event_width"]),
        hidden_dim=hidden_dim,
        readout_width=int(model_config["readout_width"]),
        target_dim=len(data.feature_names),
    )
    return hidden_dim, parameter_count(matched)


def write_results(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    torch.set_num_threads(int(training["torch_threads"]))
    device = resolve_device(args.device)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    data = load_repohealth(
        args.data_root, normalization=str(config["normalization"])
    )
    probe = make_selective(data, config["model"], torch.device("cpu"))
    selective_parameters = parameter_count(probe)
    matched_hidden, matched_parameters = closest_gru_hidden_dim(
        selective_parameters, data, config["model"]
    )
    del probe

    payload: dict[str, object] = {
        "material_passport": {
            "source": "Time-IMM RepoHealth local sparse checkout",
            "data_commit": git_commit(
                Path(__file__).resolve().parents[1] / "external" / "Time-IMM"
            ),
            "normalization": data.normalization,
            "history_days": 31,
            "prediction_days": 31,
            "stride_days": 31,
            "split": "per-entity chronological 60/20/20 by window",
            "task": "numeric-only irregular multivariate forecasting pilot",
            "primary_metric": "per-feature normalized MSE, then mean over features",
            "text_modality": "not included in this numerical backbone gate",
            "decoder": "zero-initialized correction added to last-value forecast",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "config": config,
        "effective_training": training,
        "data": {
            "feature_names": list(data.feature_names),
            "window_counts": data.window_counts,
            "query_counts": data.query_counts,
            "total_queries": {
                "train": len(data.train_samples),
                "val": len(data.val_samples),
                "test": len(data.test_samples),
            },
            "scales": {
                entity: {
                    "mean": scale.mean.tolist(),
                    "std": scale.std.tolist(),
                    "fitted_rows": scale.fitted_rows,
                    "fit_end_day": scale.fit_end_day,
                }
                for entity, scale in data.scales.items()
            },
        },
        "parameter_match": {
            "selective_eventmamba": selective_parameters,
            "set_decay_gru_hidden_dim": matched_hidden,
            "set_decay_gru": matched_parameters,
            "relative_difference": abs(matched_parameters - selective_parameters)
            / selective_parameters,
        },
        "results": [],
    }
    result_path = output_dir / "results.json"
    write_results(result_path, payload)

    if "last_value" in config["models"]:
        # The baseline iterates scalar events directly; keeping it on CPU avoids
        # hundreds of thousands of tiny host-to-device transfers.
        baseline = LastValueBaseline(data.input_dims).cpu()
        result = {
            "model": "last_value",
            "seed": None,
            "parameters": 0,
            "val_metrics": evaluate_repohealth(
                baseline, data.val_samples, data, int(training["batch_size"])
            ),
            "test_metrics": evaluate_repohealth(
                baseline, data.test_samples, data, int(training["batch_size"])
            ),
        }
        payload["results"].append(result)
        write_results(result_path, payload)
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

    trained_names = [name for name in config["models"] if name != "last_value"]
    prepared_data = None
    if trained_names:
        cache_reference = make_selective(data, config["model"], device)
        cache_start = time.perf_counter()
        prepared_data = prepare_repohealth_data(cache_reference, data)
        cache_seconds = time.perf_counter() - cache_start
        payload["prepacked_cache"] = {
            "seconds": cache_seconds,
            "device": str(device),
            "shared_across_models_and_seeds": True,
        }
        write_results(result_path, payload)
        print(
            json.dumps(
                {"prepacked_cache_seconds": cache_seconds}, ensure_ascii=False
            ),
            flush=True,
        )
        del cache_reference

    for seed in config["seeds"]:
        for name in config["models"]:
            if name == "last_value":
                continue
            set_deterministic_seed(int(seed))
            if name == "selective_eventmamba_no_reliability":
                model = make_selective(data, config["model"], device)
            elif name == "set_decay_gru_matched":
                model = ResidualEventForecast(
                    SetDecayGRUBaseline(
                        input_dims=data.input_dims,
                        event_width=int(config["model"]["event_width"]),
                        hidden_dim=matched_hidden,
                        readout_width=int(config["model"]["readout_width"]),
                        target_dim=len(data.feature_names),
                    ),
                    data.input_dims,
                ).to(device)
            else:
                raise ValueError(f"Unknown model: {name}")

            def log_record(record: dict[str, object]) -> None:
                append_jsonl(log_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

            result = train_repohealth_model(
                name=name,
                model=model,
                data=data,
                output_dir=output_dir,
                log_record=log_record,
                epochs=int(training["epochs"]),
                patience=int(training["patience"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                grad_clip=float(training["grad_clip"]),
                seed=int(seed),
                prepared_data=prepared_data,
            )
            payload["results"].append(result)
            write_results(result_path, payload)
            print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
