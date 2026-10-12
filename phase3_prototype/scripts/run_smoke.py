"""Run the first deterministic synthetic training smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventmamba.baselines import EventGRUBaseline
from eventmamba.data import SyntheticAsyncDataset
from eventmamba.model import EventMamba, EventMambaConfig
from eventmamba.training import append_jsonl, set_deterministic_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("eventgru", "eventmamba"),
        default=("eventgru", "eventmamba"),
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run log: {metrics_path}")

    seed = int(config["seed"])
    set_deterministic_seed(seed)
    torch.set_num_threads(int(config["training"].get("torch_threads", 8)))
    device = choose_device(args.device)

    data_config = dict(config["data"])
    train_count = int(data_config.pop("train_samples"))
    val_count = int(data_config.pop("val_samples"))
    test_count = int(data_config.pop("test_samples"))
    train_data = SyntheticAsyncDataset(train_count, seed=seed * 100 + 1, **data_config)
    val_data = SyntheticAsyncDataset(val_count, seed=seed * 100 + 2, **data_config)
    test_data = SyntheticAsyncDataset(test_count, seed=seed * 100 + 3, **data_config)

    model_config = EventMambaConfig(**config["model"])
    all_models = {
        "eventgru": EventGRUBaseline(
            event_width=model_config.event_width,
            hidden_dim=model_config.state_dim,
            readout_width=model_config.readout_width,
            target_dim=model_config.target_dim,
        ).to(device),
        "eventmamba": EventMamba(model_config).to(device),
    }
    models = {name: all_models[name] for name in args.models}

    resolved_config = {
        **config,
        "resolved_device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "selected_models": list(args.models),
    }
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training = config["training"]
    summaries: list[dict[str, object]] = []
    start = time.perf_counter()
    for model_name, model in models.items():
        summary = train_model(
            name=model_name,
            model=model,
            train_samples=train_data.samples,
            val_samples=val_data.samples,
            test_samples=test_data.samples,
            output_dir=output_dir,
            log_record=lambda record: (
                append_jsonl(metrics_path, record),
                print(json.dumps(record), flush=True),
            ),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            grad_clip=float(training["grad_clip"]),
            seed=seed,
        )
        summaries.append(summary)

    result = {
        "status": "completed",
        "seed": seed,
        "device": str(device),
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "models": summaries,
        "total_duration_seconds": time.perf_counter() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
