"""Reproduce the upstream IMM-TSF sample-split loader.

This harness deliberately imports the unmodified upstream
``external/IMM-TSF/lib/parse_datasets.py``.  The selected ``sample`` split does
not use scikit-learn, so a fail-closed stub prevents accidental execution of
the unrelated ``instance`` split when running in the lightweight audit
runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path


def install_fail_closed_optional_stubs() -> None:
    """Stub optional imports that are unused by the selected code path."""

    sklearn = types.ModuleType("sklearn")
    model_selection = types.ModuleType("sklearn.model_selection")

    def forbidden_train_test_split(*args, **kwargs):
        raise RuntimeError(
            "train_test_split was called: the official sample-split audit "
            "must not enter the instance-split path"
        )

    model_selection.train_test_split = forbidden_train_test_split
    sklearn.model_selection = model_selection
    sklearn.metrics = types.SimpleNamespace()
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.model_selection"] = model_selection

    prettytable = types.ModuleType("prettytable")

    class ForbiddenPrettyTable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PrettyTable was called although show_summary=False in the "
                "official loader audit"
            )

    prettytable.PrettyTable = ForbiddenPrettyTable
    sys.modules["prettytable"] = prettytable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="RepoHealth")
    parser.add_argument("--history", type=float, default=31.0)
    parser.add_argument("--prediction", type=float, default=31.0)
    parser.add_argument("--stride", type=float, default=31.0)
    parser.add_argument(
        "--time-unit",
        choices=("seconds", "minutes", "hours", "days", "weeks"),
        default="days",
    )
    return parser.parse_args()


def upstream_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def main() -> None:
    args_cli = parse_args()
    repo_root = args_cli.repo_root.resolve()
    data_root = args_cli.data_root.resolve()
    install_fail_closed_optional_stubs()
    sys.path.insert(0, str(repo_root))

    import pandas as pd
    import torch
    from lib.parse_datasets import parse_datasets

    official_args = argparse.Namespace(
        data_root=str(data_root),
        dataset=args_cli.dataset,
        device=torch.device("cpu"),
        history=args_cli.history,
        pred_window=args_cli.prediction,
        stride=args_cli.stride,
        batch_size=8,
        time_unit=args_cli.time_unit,
        unit_scale=None,
        enable_text=False,
        use_text_embeddings=False,
        llm_model_fusion="GPT2",
        llm_layers_fusion=None,
        max_length=1024,
        model="DLinear",
        split_method="sample",
        rec_ids=None,
    )
    data_obj = parse_datasets(official_args, show_summary=False)
    first_batch = next(iter(data_obj["train_dataloader"]))

    split_sizes = {
        "train": len(data_obj["train_dataloader"].dataset),
        "val": len(data_obj["val_dataloader"].dataset),
        "test": len(data_obj["test_dataloader"].dataset),
    }
    batch_shapes = {
        key: list(value.shape)
        for key, value in first_batch.items()
        if isinstance(value, torch.Tensor)
    }
    result = {
        "material_passport": {
            "source": f"Time-IMM {args_cli.dataset} local sparse checkout",
            "upstream_loader": "IMM-TSF/lib/parse_datasets.py",
            "upstream_commit": upstream_commit(repo_root),
            "data_root": str(data_root),
            "dataset": args_cli.dataset,
            "task": (
                f"{args_cli.history:g}-{args_cli.time_unit} history to "
                f"{args_cli.prediction:g}-{args_cli.time_unit} forecast"
            ),
            "split": "per-entity chronological 60/20/20 sample split",
            "stride": f"{args_cli.stride:g} {args_cli.time_unit}",
            "normalization": "upstream full-entity z-score before split",
            "text": "disabled, but upstream requires a history text event",
        },
        "runtime": {
            "python": sys.version,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "chunks": len(data_obj["ds"]),
        "split_sizes": split_sizes,
        "input_dim": data_obj["input_dim"],
        "first_train_batch_shapes": batch_shapes,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
