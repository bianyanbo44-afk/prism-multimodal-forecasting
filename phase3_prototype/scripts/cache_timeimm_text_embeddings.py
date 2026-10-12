"""Token-audit and cache frozen Time-IMM text embeddings.

The cache deliberately anchors text timestamps to the first numeric timestamp.
This avoids the upstream precompute/raw-text inconsistency where precomputed
timestamps are anchored to the first text timestamp instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.timeimm import (  # noqa: E402
    CSV_FIELD_LIMIT_BYTES,
    DATASET_SPECS,
    UNIT_SECONDS,
    gpt2_text_cache_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("external/Time-IMM/data")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision; 'main' is resolved to an immutable commit before loading.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--token-audit-only", action="store_true")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("results/timeimm_gpt2_token_audit.json"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def percentile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def resolve_revision(model_id: str, revision: str) -> str:
    if revision != "main":
        return revision
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
    return info.sha


def read_entity(entity_dir: Path, seconds_per_unit: float) -> dict[str, object]:
    numeric_path = entity_dir / "time_series.csv"
    with numeric_path.open(encoding="utf-8-sig", newline="") as handle:
        numeric_times = [
            parse_datetime(row["date_time"])
            for row in csv.DictReader(handle)
            if row.get("date_time")
        ]
    if not numeric_times:
        raise ValueError(f"No numeric timestamps in {numeric_path}")
    numeric_base = min(numeric_times)

    text_path = entity_dir / "text.csv"
    csv.field_size_limit(CSV_FIELD_LIMIT_BYTES)
    with text_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_columns = [
            column
            for column in (reader.fieldnames or ())
            if column not in {"date_time", "record_id"}
        ]
        if len(text_columns) != 1:
            raise ValueError(f"{entity_dir}: expected one text column, got {text_columns}")
        text_column = text_columns[0]
        rows: list[tuple[float, int, str]] = []
        for row_index, row in enumerate(reader):
            note = row.get(text_column)
            if note is None or not note.strip():
                continue
            timestamp = parse_datetime(row["date_time"])
            relative_time = (
                timestamp - numeric_base
            ).total_seconds() / seconds_per_unit
            rows.append((relative_time, row_index, note))
    rows.sort(key=lambda item: (item[0], item[1]))
    return {
        "entity": entity_dir.name,
        "numeric_base": numeric_base.isoformat(),
        "times": [item[0] for item in rows],
        "row_indices": [item[1] for item in rows],
        "notes": [item[2] for item in rows],
        "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
    }


def length_summary(lengths: list[int], max_length: int) -> dict[str, object]:
    return {
        "notes": len(lengths),
        "min": min(lengths),
        "median": median(lengths),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "max_length": max_length,
        "truncated_notes": sum(length > max_length for length in lengths),
        "truncated_fraction": sum(length > max_length for length in lengths)
        / len(lengths),
    }


def tokenize_lengths(tokenizer: object, notes: list[str]) -> list[int]:
    encoded = tokenizer(
        notes,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    return [len(token_ids) for token_ids in encoded["input_ids"]]


@torch.inference_mode()
def embed_notes(
    *,
    tokenizer: object,
    model: torch.nn.Module,
    notes: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    progress_every: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(notes), batch_size):
        batch = notes[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            hidden = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            expanded = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError(f"Non-finite embeddings at note offset {start}")
        outputs.append(pooled.detach().to(dtype=torch.float16, device="cpu"))
        batch_index = start // batch_size
        if (
            batch_index == 0
            or batch_index % progress_every == 0
            or start + len(batch) == len(notes)
        ):
            print(
                json.dumps(
                    {
                        "encoded": min(start + len(batch), len(notes)),
                        "total": len(notes),
                    }
                ),
                flush=True,
            )
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()
    if args.max_length < 8 or args.batch_size < 1 or args.progress_every < 1:
        raise ValueError(
            "max-length must be >=8; batch-size and progress-every must be positive"
        )
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

    resolved_revision = resolve_revision(args.model_id, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=resolved_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    entities: list[tuple[str, Path, dict[str, object]]] = []
    dataset_notes: dict[str, list[str]] = {name: [] for name in args.datasets}
    dataset_lengths: dict[str, list[int]] = {name: [] for name in args.datasets}
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        processed = (args.data_root / dataset / "processed").resolve()
        for entity_dir in sorted(path for path in processed.iterdir() if path.is_dir()):
            entity = read_entity(entity_dir, UNIT_SECONDS[spec.time_unit])
            notes = list(entity["notes"])
            lengths = tokenize_lengths(tokenizer, notes)
            entity["token_lengths"] = lengths
            entities.append((dataset, entity_dir, entity))
            dataset_notes[dataset].extend(notes)
            dataset_lengths[dataset].extend(lengths)

    all_lengths = [length for lengths in dataset_lengths.values() for length in lengths]
    audit = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "transformers": transformers_version,
        "max_length_candidate": args.max_length,
        "datasets": {
            dataset: length_summary(dataset_lengths[dataset], args.max_length)
            for dataset in args.datasets
        },
        "all": length_summary(all_lengths, args.max_length),
    }
    audit_output = args.audit_output.resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"token_audit": audit}, ensure_ascii=False), flush=True)
    if args.token_audit_only:
        return

    device = resolve_device(args.device)
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=resolved_revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resolved_from_model = getattr(model.config, "_commit_hash", None)
    if resolved_from_model and resolved_from_model != resolved_revision:
        raise RuntimeError(
            f"Model revision {resolved_from_model} != requested {resolved_revision}"
        )

    started = time.perf_counter()
    for dataset, entity_dir, entity in entities:
        output = entity_dir / gpt2_text_cache_name(args.max_length)
        if output.exists() and not args.force:
            print(json.dumps({"skipped": str(output)}), flush=True)
            continue
        embeddings = embed_notes(
            tokenizer=tokenizer,
            model=model,
            notes=list(entity["notes"]),
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            progress_every=args.progress_every,
        )
        metadata = {
            "dataset": dataset,
            "entity": entity["entity"],
            "model_id": args.model_id,
            "resolved_revision": resolved_revision,
            "pooling": "attention-mask weighted token mean",
            "max_length": args.max_length,
            "padding": "dynamic per batch",
            "truncation": True,
            "embedding_dtype": "float16",
            "embedding_dim": int(embeddings.shape[1]),
            "timestamp_base": "first numeric timestamp",
            "numeric_base": entity["numeric_base"],
            "text_sha256": entity["text_sha256"],
            "token_lengths": length_summary(
                list(entity["token_lengths"]), args.max_length
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "transformers": transformers_version,
        }
        torch.save(
            {
                "embeddings": embeddings,
                "relative_times": torch.tensor(entity["times"], dtype=torch.float64),
                "source_row_indices": torch.tensor(
                    entity["row_indices"], dtype=torch.int64
                ),
                "metadata": metadata,
            },
            output,
        )
        print(
            json.dumps(
                {
                    "cached": str(output),
                    "shape": list(embeddings.shape),
                    "seconds_total": time.perf_counter() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
