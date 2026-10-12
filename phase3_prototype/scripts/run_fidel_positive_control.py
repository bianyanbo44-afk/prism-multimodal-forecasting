"""Preregistered Fidel-TS/TGTSF positive semantic control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "external" / "fidel-ts-framework"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

_TGTSF_SPEC = importlib.util.spec_from_file_location(
    "fidel_ts_official_tgtsf", UPSTREAM / "models" / "TGTSF.py"
)
if _TGTSF_SPEC is None or _TGTSF_SPEC.loader is None:
    raise ImportError("Could not load the pinned official TGTSF source")
_TGTSF_MODULE = importlib.util.module_from_spec(_TGTSF_SPEC)
_TGTSF_SPEC.loader.exec_module(_TGTSF_MODULE)
OfficialTGTSF = _TGTSF_MODULE.Model


SEEDS = (53, 67, 79)
VARIANTS = ("aligned", "semantic_shuffle", "zero")
SPLITS = {
    "train": (pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")),
    "val": (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")),
    "test": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")),
}
CONTROL_SEED = 9101
SEQ_LEN = 360
PRED_LEN = 24
WINDOW_STRIDE = 24
TEXT_STRIDE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PlantSeries:
    plant_id: str
    values: np.ndarray
    timestamps: np.ndarray
    train_mean: float
    train_std: float
    channel_embedding: np.ndarray


class TGTSFExplicitMask(nn.Module):
    """Official TGTSF components with an explicit weather-note padding mask."""

    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(
            enc_in=1,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            e_layers=3,
            cross_layers=3,
            self_layers=3,
            mixer_self_layers=3,
            n_heads=4,
            d_model=256,
            text_dim=256,
            dropout=0.3,
            individual=False,
            patch_len=6,
            stride=TEXT_STRIDE,
            revin=True,
        )
        self.official = OfficialTGTSF(config)

    def _text_forward(
        self,
        news: torch.Tensor,
        description: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.official.text_encoder
        batch, length, channels, width = description.shape
        news_flat = news.reshape(batch * length, news.shape[2], width)
        description_flat = description.reshape(batch * length, channels, width)
        pad_flat = (~note_mask).reshape(batch * length, note_mask.shape[2])
        text = encoder.cross_encoder(
            tgt=description_flat,
            memory=news_flat,
            memory_key_padding_mask=pad_flat,
        )
        text = text.reshape(batch, length, channels, width)
        text = rearrange(text, "b l c d -> (b c) l d", b=batch, c=channels)
        text = encoder.dropout_layer(text + encoder.W_pos.permute(1, 0, 2))
        return rearrange(text, "(b c) l d -> b l c d", b=batch, c=channels)

    def forward(
        self,
        history: torch.Tensor,
        news: torch.Tensor,
        channel_embedding: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> torch.Tensor:
        model = self.official
        description = channel_embedding.unsqueeze(1).repeat(1, news.shape[1], 1, 1)
        x_mean = history.mean(dim=1, keepdim=True)
        centered = history - x_mean
        x_var = centered.var(dim=1, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)
        temporal = model.TS_encoder(normalized)
        text = self._text_forward(news, description, note_mask)
        mixed, _ = model.mixer(text, temporal)
        patches = model.head(mixed).permute(0, 2, 1, 3)
        batch, channels, count_patches, _ = patches.shape
        output = torch.zeros(
            batch, channels, model.total_length, device=patches.device, dtype=patches.dtype
        )
        count = torch.zeros_like(output)
        for index in range(count_patches):
            start = index * model.stride
            end = start + model.patch_len
            output[:, :, start:end] += patches[:, :, index, :]
            count[:, :, start:end] += 1
        forecast = (output[:, :, :PRED_LEN] / count[:, :, :PRED_LEN]).permute(0, 2, 1)
        return forecast * torch.sqrt(x_var) + x_mean


class FidelDailyDataset(Dataset):
    def __init__(
        self,
        plants: tuple[PlantSeries, ...],
        report_times: np.ndarray,
        report_values: np.ndarray,
        *,
        split: str,
        variant: str,
        permutation: np.ndarray,
    ) -> None:
        self.plants = plants
        self.report_times = report_times
        self.report_values = report_values
        self.variant = variant
        self.permutation = permutation
        start, end = SPLITS[split]
        samples: list[tuple[int, int]] = []
        for plant_index, plant in enumerate(plants):
            times = pd.DatetimeIndex(plant.timestamps)
            first_target = max(start, times[0] + pd.Timedelta(hours=SEQ_LEN))
            first_target = first_target.ceil("24h")
            last_target = min(end, times[-1] + pd.Timedelta(hours=1)) - pd.Timedelta(hours=PRED_LEN)
            candidates = pd.date_range(first_target, last_target, freq=f"{WINDOW_STRIDE}h")
            positions = times.searchsorted(candidates)
            for position, candidate in zip(positions, candidates):
                if position < SEQ_LEN or position + PRED_LEN > len(times):
                    continue
                segment = times[position - SEQ_LEN : position + PRED_LEN]
                expected = pd.date_range(
                    candidate - pd.Timedelta(hours=SEQ_LEN),
                    periods=SEQ_LEN + PRED_LEN,
                    freq="h",
                )
                if segment.equals(expected):
                    samples.append((plant_index, int(position)))
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        plant_index, position = self.samples[index]
        plant = self.plants[plant_index]
        history = plant.values[position - SEQ_LEN : position]
        target = plant.values[position : position + PRED_LEN]
        query_times = plant.timestamps[position : position + PRED_LEN : TEXT_STRIDE]
        report_indices = np.searchsorted(self.report_times, query_times, side="right") - 1
        report_indices = np.maximum(report_indices, 0)
        if self.variant == "semantic_shuffle":
            report_indices = self.permutation[report_indices]
        weather = self.report_values[report_indices]
        if self.variant == "zero":
            weather = np.zeros_like(weather)
        padded = np.zeros((weather.shape[0], 8, 256), dtype=np.float32)
        padded[:, :7] = weather
        note_mask = np.zeros((weather.shape[0], 8), dtype=bool)
        note_mask[:, :7] = True
        return (
            torch.from_numpy(history[:, None]),
            torch.from_numpy(target[:, None]),
            torch.from_numpy(padded),
            torch.from_numpy(plant.channel_embedding[None, :]),
            torch.from_numpy(note_mask),
        )


def load_materials(root: Path) -> tuple[tuple[PlantSeries, ...], np.ndarray, np.ndarray]:
    static = joblib.load(root / "static_info_embeddings.pkl")
    plants: list[PlantSeries] = []
    for path in sorted((root / "time_series").glob("id_*.parquet")):
        plant_id = path.stem.removeprefix("id_")
        frame = pd.read_parquet(path).sort_values("date")
        timestamps = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")
        raw = frame["kWh"].to_numpy(dtype=np.float32)
        train_mask = timestamps < np.datetime64("2021-01-01")
        mean = float(raw[train_mask].mean())
        std = float(raw[train_mask].std())
        if not np.isfinite(std) or std < 1e-8:
            raise ValueError(f"Invalid training standard deviation for {plant_id}")
        values = ((raw - mean) / std).astype(np.float32)
        channel = np.asarray(static["channel_info"][plant_id], dtype=np.float32).reshape(-1)
        plants.append(
            PlantSeries(plant_id, values, timestamps, mean, std, channel)
        )

    reports: dict[str, np.ndarray] = {}
    weather_root = root / "weather" / "calgary" / "report_embedding" / "formal_report"
    for path in sorted(weather_root.glob("fast_general_formal_embeddings_*.pkl")):
        reports.update(joblib.load(path))
    ordered_keys = sorted(key for key in reports if "2018" <= key[:4] <= "2022")
    report_times = pd.to_datetime(ordered_keys, format="%Y%m%d%H%M").to_numpy(
        dtype="datetime64[ns]"
    )
    report_values = np.stack([reports[key] for key in ordered_keys]).astype(np.float32)
    if report_values.shape[1:] != (7, 256):
        raise ValueError(f"Unexpected weather embedding shape: {report_values.shape}")
    return tuple(plants), report_times, report_values


def split_permutation(report_times: np.ndarray) -> np.ndarray:
    mapping = np.arange(len(report_times))
    for split_index, (_, (start, end)) in enumerate(SPLITS.items()):
        indices = np.flatnonzero(
            (report_times >= start.to_datetime64()) & (report_times < end.to_datetime64())
        )
        rng = np.random.default_rng(CONTROL_SEED + split_index)
        mapping[indices] = rng.permutation(indices)
    return mapping


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    count = 0
    for history, target, weather, channel, note_mask in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weather = weather.to(device, non_blocking=True)
        channel = channel.to(device, non_blocking=True)
        note_mask = note_mask.to(device, non_blocking=True)
        prediction = model(history, weather, channel, note_mask)
        error = prediction - target
        squared += float(error.square().sum().cpu())
        absolute += float(error.abs().sum().cpu())
        count += error.numel()
    return {"mse": squared / count, "mae": absolute / count, "observations": count}


def train_one(
    *,
    seed: int,
    variant: str,
    datasets: dict[str, FidelDailyDataset],
    output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    log_path: Path,
) -> dict[str, object]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model = TGTSFExplicitMask().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    checkpoint = output / f"tgtsf_{variant}_seed{seed}.pt"
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        train_squared = 0.0
        train_count = 0
        for history, target, weather, channel, note_mask in loaders["train"]:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            channel = channel.to(device, non_blocking=True)
            note_mask = note_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, weather, channel, note_mask)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {variant}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_squared += float(loss.detach().cpu()) * target.numel()
            train_count += target.numel()
        val = evaluate(model, loaders["val"], device)
        record = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_mse": train_squared / train_count,
            "val_mse": val["mse"],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if val["mse"] < best_val - 1e-8:
            best_val = val["mse"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 3:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation = evaluate(model, loaders["val"], device)
    test = evaluate(model, loaders["test"], device)
    return {
        "variant": variant,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "val_metrics": validation,
        "test_metrics": test,
        "duration_seconds": time.perf_counter() - start,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()
    result_path = output / "results.json"
    root = args.data_root.resolve()
    plants, report_times, report_values = load_materials(root)
    permutation = split_permutation(report_times)
    datasets_by_variant = {
        variant: {
            split: FidelDailyDataset(
                plants,
                report_times,
                report_values,
                split=split,
                variant=variant,
                permutation=permutation,
            )
            for split in SPLITS
        }
        for variant in args.variants
    }
    payload: dict[str, object] = {
        "material_passport": {
            "dataset": "VEWOXIC/Canada_photovoltaics_plants",
            "fidel_ts_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
            ).strip(),
            "source_sha256": {
                name: sha256(UPSTREAM / name)
                for name in ("models/TGTSF.py", "layers/TGTSF_torch.py")
            },
            "entities": [plant.plant_id for plant in plants],
            "splits": {
                split: [str(start.date()), str(end.date())]
                for split, (start, end) in SPLITS.items()
            },
            "normalization": "per-plant training-period mean/std only",
            "window": {"history": SEQ_LEN, "prediction": PRED_LEN, "stride": WINDOW_STRIDE},
            "weather": {
                "reports": int(len(report_times)),
                "shape_per_report": [7, 256],
                "matching": "latest report at or before each 3-hour query",
            },
            "adaptation": (
                "official TGTSF components/equations with explicit 7-real+1-padding "
                "note mask; daily rather than stride-1 window starts"
            ),
            "semantic_control_seed": CONTROL_SEED,
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "training": {
            "epochs": args.epochs,
            "patience": 3,
            "batch_size": args.batch_size,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "seeds": args.seeds,
        },
        "sample_counts": {
            variant: {split: len(dataset) for split, dataset in datasets.items()}
            for variant, datasets in datasets_by_variant.items()
        },
        "results": [],
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for variant in args.variants:
        for seed in args.seeds:
            result = train_one(
                seed=int(seed),
                variant=variant,
                datasets=datasets_by_variant[variant],
                output=output,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                log_path=log_path,
            )
            payload["results"].append(result)
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"completed": result}), flush=True)


if __name__ == "__main__":
    main()
