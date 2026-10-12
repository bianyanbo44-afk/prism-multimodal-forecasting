"""Fit train-only per-feature shrinkage of persistence toward the train mean."""

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
    PackedShrinkagePersistenceForecast,
    packed_last_values,
)
from eventmamba.timeimm import DATASET_SPECS, load_timeimm  # noqa: E402
from eventmamba.timeimm_training import evaluate_timeimm, prepare_timeimm_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("external/Time-IMM/data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/timeimm_shrinkage_baseline.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def fit_shrinkage(prepared: object, input_dims: tuple[int, ...]) -> torch.Tensor:
    latest = packed_last_values(prepared.packed, input_dims)
    predictors = latest.unsqueeze(1).expand_as(prepared.targets)
    mask = prepared.target_masks
    numerator = (predictors * prepared.targets * mask).sum((0, 1))
    denominator = (predictors.square() * mask).sum((0, 1)).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    report: dict[str, object] = {
        "protocol": {
            "fit": "closed-form least squares on training targets only",
            "constraint": "one shrinkage weight per feature, clipped to [0,1]",
            "forecast": "alpha * latest + (1-alpha) * per-entity train mean",
            "normalization": "per-entity train-only",
        },
        "datasets": {},
    }
    for name, spec in DATASET_SPECS.items():
        data = load_timeimm(args.data_root / name, spec, normalization="train_only")
        reference = PackedLastValueForecast(data.input_dims).to(device)
        prepared = prepare_timeimm_data(reference, data)
        weights = fit_shrinkage(prepared.train, data.input_dims)
        model = PackedShrinkagePersistenceForecast(data.input_dims, weights).to(device)
        result = {
            "weights": weights.detach().cpu().tolist(),
            "val": evaluate_timeimm(model, prepared.val, data, 64),
            "test": evaluate_timeimm(model, prepared.test, data, 64),
        }
        report["datasets"][name] = result
        print(
            json.dumps(
                {
                    "dataset": name,
                    "weights": result["weights"],
                    "test_mse": result["test"]["mse"],
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
