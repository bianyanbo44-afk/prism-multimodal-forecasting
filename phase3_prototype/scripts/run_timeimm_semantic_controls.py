"""Run shuffled-semantic and time-only negative controls."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.forecasting import ResidualEventForecast  # noqa: E402
from eventmamba.model import EventMambaConfig  # noqa: E402
from eventmamba.multimodal import (  # noqa: E402
    MeanTextFusionBaseline,
    shuffle_text_values_within_entities,
    zero_text_values,
)
from eventmamba.selective import SelectiveEventMamba  # noqa: E402
from eventmamba.timeimm import (  # noqa: E402
    DATASET_SPECS,
    GPT2_TEXT_CACHE,
    TimeIMMWindowSample,
    load_timeimm,
)
from eventmamba.timeimm_training import (  # noqa: E402
    PreparedTimeIMMData,
    prepare_timeimm_split,
    train_timeimm_model,
)
from eventmamba.training import append_jsonl, set_deterministic_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--text-cache-name", default=GPT2_TEXT_CACHE)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_variant(
    model: torch.nn.Module,
    data: object,
    windows: dict[str, tuple[TimeIMMWindowSample, ...]],
) -> PreparedTimeIMMData:
    target_dim = len(data.feature_names)
    return PreparedTimeIMMData(
        train=prepare_timeimm_split(
            model, windows["train"], data.input_dims, target_dim
        ),
        val=prepare_timeimm_split(model, windows["val"], data.input_dims, target_dim),
        test=prepare_timeimm_split(
            model, windows["test"], data.input_dims, target_dim
        ),
    )


def make_model(name: str, data: object, config: dict[str, object], device: torch.device) -> torch.nn.Module:
    numeric = tuple(range(len(data.feature_names)))
    if data.text_modality is None:
        raise AssertionError("Text modality is unavailable")
    if name == "mean_text_fusion":
        return MeanTextFusionBaseline(
            input_dims=data.input_dims,
            numeric_modalities=numeric,
            text_modality=data.text_modality,
            event_width=int(config["model"]["event_width"]),
            text_width=int(config["model"]["state_dim"]),
            readout_width=int(config["model"]["readout_width"]),
            target_dim=len(data.feature_names),
        ).to(device)
    base = SelectiveEventMamba(
        EventMambaConfig(
            input_dims=data.input_dims,
            event_width=int(config["model"]["event_width"]),
            state_dim=int(config["model"]["state_dim"]),
            readout_width=int(config["model"]["readout_width"]),
            target_dim=len(data.feature_names),
            use_reliability=False,
        )
    )
    return ResidualEventForecast(
        base,
        data.input_dims,
        persistence_modalities=numeric,
    ).to(device)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training_seeds = [
        int(seed)
        for seed in config.get("seeds", [config.get("seed", 101)])
    ]
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    torch.set_num_threads(int(training["torch_threads"]))
    device = resolve_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()
    result_path = output / "results.json"

    spec = DATASET_SPECS[args.dataset]
    data = load_timeimm(
        args.data_root.resolve() / args.dataset,
        spec,
        normalization=str(config["normalization"]),
        text_cache_name=args.text_cache_name,
        text_preprocessing="layer_norm",
    )
    if data.text_modality is None:
        raise AssertionError("Text modality is unavailable")
    original = {
        "train": data.train_windows,
        "val": data.val_windows,
        "test": data.test_windows,
    }
    control_seed = int(config["semantic_control_seed"])
    variants = {
        "semantic_shuffle": {
            split: shuffle_text_values_within_entities(
                windows,
                data.window_info,
                data.text_modality,
                seed=control_seed + split_index,
            )
            for split_index, (split, windows) in enumerate(original.items())
        },
        "time_only_zero_text": {
            split: zero_text_values(windows, data.text_modality)
            for split, windows in original.items()
        },
    }
    payload: dict[str, object] = {
        "material_passport": {
            "source": f"Time-IMM {args.dataset} local sparse checkout",
            "data_commit": git_commit(PROJECT_ROOT / "external" / "Time-IMM"),
            "normalization": data.normalization,
            "split": "per-entity chronological 60/20/20 by retained window",
            "task": "semantic negative controls",
            "text_preprocessing": "parameter-free per-note LayerNorm",
            "semantic_shuffle": "independent within each entity and split; timestamps/counts/marginals preserved",
            "time_only": "all text vectors zero; timestamps and counts preserved",
            "control_seed": control_seed,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "config": config,
        "effective_training": training,
        "results": [],
    }
    write(result_path, payload)

    for variant_name, windows in variants.items():
        for architecture in ("mean_text_fusion", "selective_ssm"):
            probe = make_model(architecture, data, config, device)
            prepared = prepare_variant(probe, data, windows)
            del probe
            name = f"{architecture}_{variant_name}"

            for seed in training_seeds:
                set_deterministic_seed(seed)
                model = make_model(architecture, data, config, device)

                def log_record(record: dict[str, object]) -> None:
                    append_jsonl(log_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)

                result = train_timeimm_model(
                    name=name,
                    model=model,
                    data=data,
                    prepared=prepared,
                    output_dir=output,
                    log_record=log_record,
                    epochs=int(training["epochs"]),
                    patience=int(training["patience"]),
                    batch_size=int(training["batch_size"]),
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                    grad_clip=float(training["grad_clip"]),
                    seed=seed,
                )
                result["control"] = variant_name
                payload["results"].append(result)
                write(result_path, payload)
                print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
