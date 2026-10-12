"""Run text-only and naive mean-text fusion controls on Time-IMM."""

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

from eventmamba.baselines import SetDecayGRUBaseline  # noqa: E402
from eventmamba.multimodal import (  # noqa: E402
    MeanTextFusionBaseline,
    select_window_modalities,
    zero_initialize_readout,
)
from eventmamba.timeimm import (  # noqa: E402
    DATASET_SPECS,
    GPT2_TEXT_CACHE,
    load_timeimm,
)
from eventmamba.timeimm_training import (  # noqa: E402
    PreparedTimeIMMData,
    prepare_timeimm_data,
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


def prepare_text_only(model: torch.nn.Module, data: object) -> PreparedTimeIMMData:
    text_modality = data.text_modality
    if text_modality is None or data.text_embedding_dim is None:
        raise ValueError("Text-only control requires cached text events")
    input_dims = (data.text_embedding_dim,)
    target_dim = len(data.feature_names)
    return PreparedTimeIMMData(
        train=prepare_timeimm_split(
            model,
            select_window_modalities(data.train_windows, (text_modality,)),
            input_dims,
            target_dim,
        ),
        val=prepare_timeimm_split(
            model,
            select_window_modalities(data.val_windows, (text_modality,)),
            input_dims,
            target_dim,
        ),
        test=prepare_timeimm_split(
            model,
            select_window_modalities(data.test_windows, (text_modality,)),
            input_dims,
            target_dim,
        ),
    )


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
    text_protocol = next(iter(data.text_metadata.values()))
    protocol_keys = (
        "model_id",
        "resolved_revision",
        "pooling",
        "max_length",
        "embedding_dim",
        "timestamp_base",
    )
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
            "task": "multimodal control experiments",
            "primary_metric": "per-feature normalized MSE, then mean over observed features",
            "text_preprocessing": "parameter-free per-note LayerNorm",
            "text_protocol": {key: text_protocol[key] for key in protocol_keys},
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
        "data": {
            "dataset": args.dataset,
            "feature_names": list(data.feature_names),
            "event_input_dims": list(data.input_dims),
            "windows": {
                "train": len(data.train_windows),
                "val": len(data.val_windows),
                "test": len(data.test_windows),
            },
        },
        "results": [],
        "prepacked_cache": {},
    }
    write(result_path, payload)

    numeric_modalities = tuple(range(len(data.feature_names)))
    if data.text_modality is None:
        raise AssertionError("Text modality is unavailable")
    for name, view in (
        ("mean_text_fusion", "full"),
        ("text_only_set_decay_gru", "text_only"),
    ):
        def build_model() -> torch.nn.Module:
            if name == "mean_text_fusion":
                return MeanTextFusionBaseline(
                    input_dims=data.input_dims,
                    numeric_modalities=numeric_modalities,
                    text_modality=data.text_modality,
                    event_width=int(config["model"]["event_width"]),
                    text_width=int(config["model"]["state_dim"]),
                    readout_width=int(config["model"]["readout_width"]),
                    target_dim=len(data.feature_names),
                ).to(device)
            control = SetDecayGRUBaseline(
                input_dims=(int(data.text_embedding_dim),),
                event_width=int(config["model"]["event_width"]),
                hidden_dim=int(config["model"]["state_dim"]),
                readout_width=int(config["model"]["readout_width"]),
                target_dim=len(data.feature_names),
            ).to(device)
            zero_initialize_readout(control)
            return control

        probe = build_model()
        cache_start = time.perf_counter()
        prepared = (
            prepare_timeimm_data(probe, data)
            if view == "full"
            else prepare_text_only(probe, data)
        )
        payload["prepacked_cache"][name] = {
            "seconds": time.perf_counter() - cache_start,
            "view": view,
            "shared_across_seeds": True,
        }
        write(result_path, payload)
        del probe

        for seed in training_seeds:
            set_deterministic_seed(seed)
            model = build_model()

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
            result["event_view"] = view
            payload["results"].append(result)
            write(result_path, payload)
            print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
