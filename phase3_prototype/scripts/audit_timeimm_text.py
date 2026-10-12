"""Audit Time-IMM text events and their retained-window coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.timeimm import (  # noqa: E402
    CSV_FIELD_LIMIT_BYTES,
    DATASET_SPECS,
    UNIT_SECONDS,
    load_timeimm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("external/Time-IMM/data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/timeimm_text_audit.json")
    )
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_entity_text(entity_dir: Path, seconds_per_unit: float) -> list[dict[str, object]]:
    with (entity_dir / "time_series.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        numeric_reader = csv.DictReader(handle)
        numeric_times = [
            parse_datetime(row["date_time"])
            for row in numeric_reader
            if row.get("date_time")
        ]
    if not numeric_times:
        raise ValueError(f"No numeric timestamps in {entity_dir}")
    base = min(numeric_times)

    csv.field_size_limit(CSV_FIELD_LIMIT_BYTES)
    with (entity_dir / "text.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        text_columns = [
            column
            for column in (reader.fieldnames or ())
            if column not in {"date_time", "record_id"}
        ]
        if len(text_columns) != 1:
            raise ValueError(
                f"{entity_dir}: expected one text column, got {text_columns}"
            )
        text_column = text_columns[0]
        rows = []
        for row_index, row in enumerate(reader):
            raw = row.get(text_column)
            if raw is None or not raw.strip():
                continue
            timestamp = parse_datetime(row["date_time"])
            rows.append(
                {
                    "row_index": row_index,
                    "time": (timestamp - base).total_seconds() / seconds_per_unit,
                    "text": raw,
                }
            )
    rows.sort(key=lambda item: (float(item["time"]), int(item["row_index"])))
    return rows


def main() -> None:
    args = parse_args()
    datasets: dict[str, object] = {}
    for name, spec in DATASET_SPECS.items():
        root = (args.data_root / name).resolve()
        data = load_timeimm(root, spec, normalization="train_only")
        seconds_per_unit = UNIT_SECONDS[spec.time_unit]
        texts_by_entity: dict[str, list[dict[str, object]]] = {}
        all_lengths: list[int] = []
        total_non_ascii = 0
        total_characters = 0
        duplicate_timestamp_groups = 0
        for entity_dir in sorted((root / "processed").iterdir()):
            if not entity_dir.is_dir():
                continue
            rows = read_entity_text(entity_dir, seconds_per_unit)
            texts_by_entity[entity_dir.name] = rows
            lengths = [len(str(row["text"])) for row in rows]
            all_lengths.extend(lengths)
            for row in rows:
                raw = str(row["text"])
                total_characters += len(raw)
                total_non_ascii += sum(ord(character) > 127 for character in raw)
            counts = Counter(float(row["time"]) for row in rows)
            duplicate_timestamp_groups += sum(count > 1 for count in counts.values())

        split_windows = {
            "train": data.train_windows,
            "val": data.val_windows,
            "test": data.test_windows,
        }
        window_stats: dict[str, object] = {}
        unique_used: set[tuple[str, int]] = set()
        for split, windows in split_windows.items():
            notes_per_window: list[int] = []
            split_unique: set[tuple[str, int]] = set()
            for window in windows:
                info = data.window_info[window.sample_id]
                selected = [
                    row
                    for row in texts_by_entity[info.entity]
                    if info.start_time
                    <= float(row["time"])
                    < info.start_time + spec.history
                ]
                if not selected:
                    raise AssertionError(
                        f"retained window {window.sample_id} has no text"
                    )
                notes_per_window.append(len(selected))
                for row in selected:
                    key = (info.entity, int(row["row_index"]))
                    split_unique.add(key)
                    unique_used.add(key)
            window_stats[split] = {
                "windows": len(windows),
                "note_occurrences": sum(notes_per_window),
                "unique_notes": len(split_unique),
                "notes_per_window_min": min(notes_per_window),
                "notes_per_window_median": median(notes_per_window),
                "notes_per_window_p95": percentile(notes_per_window, 0.95),
                "notes_per_window_max": max(notes_per_window),
            }

        datasets[name] = {
            "entities": len(texts_by_entity),
            "raw_nonempty_notes": sum(len(rows) for rows in texts_by_entity.values()),
            "unique_notes_used_by_retained_windows": len(unique_used),
            "text_characters": total_characters,
            "non_ascii_character_fraction": (
                total_non_ascii / total_characters if total_characters else 0.0
            ),
            "duplicate_timestamp_groups": duplicate_timestamp_groups,
            "characters_per_note": {
                "min": min(all_lengths),
                "median": median(all_lengths),
                "p95": percentile(all_lengths, 0.95),
                "p99": percentile(all_lengths, 0.99),
                "max": max(all_lengths),
            },
            "rough_tokens_at_four_characters_per_token": int(
                math.ceil(total_characters / 4.0)
            ),
            "window_stats": window_stats,
        }

    report = {
        "source": str(args.data_root.resolve()),
        "data_commit": "e934cef23d526817dcf0d1a66c6f82274c96c04b",
        "normalization": "train_only (numeric only; text encoder remains frozen)",
        "window_rule": "text events with start <= t < start + history",
        "datasets": datasets,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
