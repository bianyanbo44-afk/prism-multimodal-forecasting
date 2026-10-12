"""Evaluate persistence and per-entity train-mean Time-IMM baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.forecasting import (  # noqa: E402
    PackedLastValueForecast,
    PackedTrainMeanForecast,
)
from eventmamba.timeimm import DATASET_SPECS, load_timeimm  # noqa: E402
from eventmamba.timeimm_training import (  # noqa: E402
    evaluate_timeimm,
    prepare_timeimm_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("external/Time-IMM/data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/timeimm_trivial_baselines.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    report: dict[str, object] = {
        "protocol": {
            "normalization": "per-entity train-only",
            "train_mean": "zero in normalized space for every target feature",
            "last_value": "latest observed scalar per numeric channel",
            "primary_metric": "per-feature normalized MSE, then mean over observed features",
        },
        "datasets": {},
    }
    for name, spec in DATASET_SPECS.items():
        data = load_timeimm(args.data_root / name, spec, normalization="train_only")
        persistence = PackedLastValueForecast(data.input_dims).to(device)
        prepared = prepare_timeimm_data(persistence, data)
        train_mean = PackedTrainMeanForecast(len(data.feature_names)).to(device)
        report["datasets"][name] = {
            "train_mean": {
                "val": evaluate_timeimm(train_mean, prepared.val, data, 64),
                "test": evaluate_timeimm(train_mean, prepared.test, data, 64),
            },
            "last_value": {
                "val": evaluate_timeimm(persistence, prepared.val, data, 64),
                "test": evaluate_timeimm(persistence, prepared.test, data, 64),
            },
        }
        print(
            json.dumps(
                {
                    "dataset": name,
                    "train_mean_test_mse": report["datasets"][name]["train_mean"]["test"]["mse"],
                    "last_value_test_mse": report["datasets"][name]["last_value"]["test"]["mse"],
                }
            ),
            flush=True,
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
