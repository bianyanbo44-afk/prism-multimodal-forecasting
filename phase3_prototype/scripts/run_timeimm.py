"""Run window-level multi-query Time-IMM numerical experiments."""

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

from eventmamba.baselines import SetDecayGRUBaseline
from eventmamba.forecasting import PackedLastValueForecast, ResidualEventForecast
from eventmamba.model import EventMambaConfig
from eventmamba.selective import SelectiveEventMamba
from eventmamba.timeimm import (
    DATASET_SPECS,
    GPT2_TEXT_CACHE,
    TimeIMMData,
    load_timeimm,
)
from eventmamba.timeimm_training import (
    evaluate_timeimm,
    prepare_timeimm_data,
    train_timeimm_model,
)
from eventmamba.training import append_jsonl, parameter_count, set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--with-text",
        action="store_true",
        help=f"Load the frozen GPT-2 cache ({GPT2_TEXT_CACHE}).",
    )
    parser.add_argument(
        "--text-cache-name",
        default=GPT2_TEXT_CACHE,
        help="Frozen embedding cache filename under each entity directory.",
    )
    parser.add_argument(
        "--text-preprocessing",
        choices=("none", "layer_norm"),
        default="layer_norm",
        help="Parameter-free preprocessing applied independently to each cached note.",
    )
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
    data: TimeIMMData, model_config: dict[str, object], device: torch.device
) -> ResidualEventForecast:
    config = EventMambaConfig(
        input_dims=data.input_dims,
        event_width=int(model_config["event_width"]),
        state_dim=int(model_config["state_dim"]),
        readout_width=int(model_config["readout_width"]),
        target_dim=len(data.feature_names),
        use_reliability=False,
    )
    return ResidualEventForecast(
        SelectiveEventMamba(config),
        data.input_dims,
        persistence_modalities=tuple(range(len(data.feature_names))),
    ).to(device)


def closest_gru_hidden_dim(
    target_parameters: int,
    data: TimeIMMData,
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

    spec = DATASET_SPECS[args.dataset]
    data = load_timeimm(
        args.data_root.resolve() / args.dataset,
        spec,
        normalization=str(config["normalization"]),
        text_cache_name=args.text_cache_name if args.with_text else None,
        text_preprocessing=args.text_preprocessing if args.with_text else "none",
    )
    probe = make_selective(data, config["model"], torch.device("cpu"))
    selective_parameters = parameter_count(probe)
    matched_hidden, matched_parameters = closest_gru_hidden_dim(
        selective_parameters, data, config["model"]
    )
    del probe

    payload: dict[str, object] = {
        "material_passport": {
            "source": f"Time-IMM {args.dataset} local sparse checkout",
            "data_commit": git_commit(PROJECT_ROOT / "external" / "Time-IMM"),
            "normalization": data.normalization,
            "history": spec.history,
            "prediction": spec.prediction,
            "stride": spec.stride,
            "time_unit": spec.time_unit,
            "split": "per-entity chronological 60/20/20 by retained window",
            "task": (
                "asynchronous numeric-text forecasting external gate"
                if args.with_text
                else "numeric-only sparse irregular forecasting external gate"
            ),
            "primary_metric": "per-feature normalized MSE, then mean over observed features",
            "decoder": "zero-initialized correction added to last-value forecast",
            "text_rule": (
                "frozen GPT-2 masked-mean embeddings are independent timestamped "
                f"events; preprocessing={data.text_preprocessing}"
                if args.with_text
                else "official loader retention rule applied; text not encoded yet"
            ),
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
            "dataset": args.dataset,
            "feature_names": list(data.feature_names),
            "event_input_dims": list(data.input_dims),
            "text_cache_name": data.text_cache_name,
            "text_embedding_dim": data.text_embedding_dim,
            "text_preprocessing": data.text_preprocessing,
            "text_protocol": (
                next(iter(data.text_metadata.values())) if data.text_metadata else None
            ),
            "window_counts": data.window_counts,
            "target_counts": data.target_counts,
            "dropped_no_text": data.dropped_no_text,
            "total_windows": {
                "train": len(data.train_windows),
                "val": len(data.val_windows),
                "test": len(data.test_windows),
            },
            "scales": {
                entity: {
                    "mean": scale.mean.tolist(),
                    "std": scale.std.tolist(),
                    "fitted_rows": scale.fitted_rows,
                    "observed_counts": scale.observed_counts.tolist(),
                    "fit_end_time": scale.fit_end_time,
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

    cache_reference = make_selective(data, config["model"], device)
    cache_start = time.perf_counter()
    prepared = prepare_timeimm_data(cache_reference, data)
    cache_seconds = time.perf_counter() - cache_start
    payload["prepacked_cache"] = {
        "seconds": cache_seconds,
        "device": str(device),
        "shared_across_models_and_seeds": True,
    }
    write_results(result_path, payload)
    print(json.dumps({"prepacked_cache_seconds": cache_seconds}), flush=True)
    del cache_reference

    if "last_value" in config["models"]:
        baseline = PackedLastValueForecast(
            data.input_dims,
            persistence_modalities=tuple(range(len(data.feature_names))),
        ).to(device)
        result = {
            "model": "last_value",
            "seed": None,
            "parameters": 0,
            "val_metrics": evaluate_timeimm(
                baseline, prepared.val, data, int(training["batch_size"])
            ),
            "test_metrics": evaluate_timeimm(
                baseline, prepared.test, data, int(training["batch_size"])
            ),
        }
        payload["results"].append(result)
        write_results(result_path, payload)
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

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
                    persistence_modalities=tuple(range(len(data.feature_names))),
                ).to(device)
            else:
                raise ValueError(f"Unknown model: {name}")

            def log_record(record: dict[str, object]) -> None:
                append_jsonl(log_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

            result = train_timeimm_model(
                name=name,
                model=model,
                data=data,
                prepared=prepared,
                output_dir=output_dir,
                log_record=log_record,
                epochs=int(training["epochs"]),
                patience=int(training["patience"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                grad_clip=float(training["grad_clip"]),
                seed=int(seed),
            )
            payload["results"].append(result)
            write_results(result_path, payload)
            print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
