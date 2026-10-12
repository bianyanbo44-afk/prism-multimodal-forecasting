"""Audit the official IMM-TSF RecAvg + GR-Add fusion under a fair data protocol."""

from __future__ import annotations

import argparse
import hashlib
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

from eventmamba.immtsf_baseline import (  # noqa: E402
    IMMTSFRecAvgGRAddBaseline,
)
from eventmamba.multimodal import (  # noqa: E402
    shuffle_text_values_within_entities,
    zero_text_values,
)
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
from eventmamba.training import (  # noqa: E402
    append_jsonl,
    parameter_count,
    set_deterministic_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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


def write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
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
        text_preprocessing="none",
    )
    if data.text_modality is None:
        raise AssertionError("Text modality unavailable")
    original = {
        "train": data.train_windows,
        "val": data.val_windows,
        "test": data.test_windows,
    }
    control_seed = int(config["semantic_control_seed"])
    shuffled = {
        split: shuffle_text_values_within_entities(
            windows,
            data.window_info,
            data.text_modality,
            seed=control_seed + split_index,
        )
        for split_index, (split, windows) in enumerate(original.items())
    }
    time_only = {
        split: zero_text_values(windows, data.text_modality)
        for split, windows in original.items()
    }
    variants = {
        "aligned_unit_consistent": (original, "unit_consistent"),
        "semantic_shuffle_unit_consistent": (shuffled, "unit_consistent"),
        "time_only_unit_consistent": (time_only, "unit_consistent"),
        "aligned_released_mixed_units": (original, "released_mixed_units"),
    }

    upstream = PROJECT_ROOT / "external" / "IMM-TSF"
    source_files = {
        name: upstream / "fusions" / name
        for name in ("TTF_RecAvg.py", "MMF_GR_Add.py")
    }
    payload: dict[str, object] = {
        "material_passport": {
            "dataset": args.dataset,
            "data_commit": git_commit(PROJECT_ROOT / "external" / "Time-IMM"),
            "immtsf_commit": git_commit(upstream),
            "immtsf_license": "MIT",
            "immtsf_source_sha256": {
                name: file_sha256(path) for name, path in source_files.items()
            },
            "normalization": data.normalization,
            "split": "per-entity chronological 60/20/20 by retained window",
            "base_forecast": "last observed numeric value",
            "text_embeddings": "frozen full GPT-2 masked mean; no per-note LayerNorm",
            "official_modules": "TTF_RecAvg + MMF_GR_Add, imported from pinned checkout",
            "adaptation": (
                "explicit note mask for content ablations; packed leakage-free windows; "
                "unit-consistent and released-mixed-unit time paths both reported"
            ),
            "time_unit_audit": {
                "unit_consistent": "tau and t_hat divided by history+prediction",
                "released_mixed_units": (
                    "tau remains physical while t_hat is divided by history+prediction"
                ),
            },
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
        "effective_seeds": seeds,
        "data": {
            "features": list(data.feature_names),
            "input_dims": list(data.input_dims),
            "windows": {
                "train": len(data.train_windows),
                "val": len(data.val_windows),
                "test": len(data.test_windows),
            },
        },
        "prepacked_cache": {},
        "results": [],
    }
    write(result_path, payload)

    numeric = tuple(range(len(data.feature_names)))

    def build_model(time_mode: str) -> IMMTSFRecAvgGRAddBaseline:
        return IMMTSFRecAvgGRAddBaseline(
            input_dims=data.input_dims,
            numeric_modalities=numeric,
            text_modality=int(data.text_modality),
            target_dim=len(data.feature_names),
            total_window=spec.total,
            time_mode=time_mode,
            d_txt=768,
            recency_sigma=1.0,
            dropout=0.1,
        ).to(device)

    for variant, (windows, time_mode) in variants.items():
        probe = build_model(time_mode)
        start = time.perf_counter()
        prepared = prepare_variant(probe, data, windows)
        payload["prepacked_cache"][variant] = {
            "seconds": time.perf_counter() - start,
            "shared_across_seeds": True,
            "parameters": parameter_count(probe),
        }
        write(result_path, payload)
        del probe

        for seed in seeds:
            set_deterministic_seed(int(seed))
            model = build_model(time_mode)
            name = f"immtsf_recavg_gradd_{variant}"

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
                seed=int(seed),
            )
            result["variant"] = variant
            result["time_mode"] = time_mode
            payload["results"].append(result)
            write(result_path, payload)
            print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
