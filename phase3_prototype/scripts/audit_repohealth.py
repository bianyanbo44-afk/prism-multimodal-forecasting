"""Audit RepoHealth CSVs and reproduce official window-count logic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


SECONDS_PER_DAY = 86400.0
CSV_FIELD_LIMIT_BYTES = 64 * 1024 * 1024

# Repository issue/PR text can contain long concatenated discussions.  The
# standard-library default (128 KiB) is too small for legitimate rows here.
csv.field_size_limit(CSV_FIELD_LIMIT_BYTES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def is_present(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "nan", "na", "null"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    args = parse_args()
    processed = args.root.resolve() / "processed"
    entities = sorted(path for path in processed.iterdir() if path.is_dir())
    result_entities: list[dict[str, object]] = []
    total_rows = 0
    total_text = 0
    all_feature_names: set[str] = set()

    for entity_dir in entities:
        ts_fields, ts_rows = read_csv(entity_dir / "time_series.csv")
        text_fields, text_rows = read_csv(entity_dir / "text.csv")
        schedule_fields, schedule_rows = read_csv(entity_dir / "schedule_intervals.csv")
        feature_names = [
            field for field in ts_fields if field not in {"date_time", "record_id"}
        ]
        all_feature_names.update(feature_names)
        timestamps = [parse_datetime(row["date_time"]) for row in ts_rows]
        base_time = min(timestamps)
        relative_days = [
            (timestamp - base_time).total_seconds() / SECONDS_PER_DAY
            for timestamp in timestamps
        ]
        text_timestamps = [parse_datetime(row["date_time"]) for row in text_rows]
        relative_text_days = [
            (timestamp - base_time).total_seconds() / SECONDS_PER_DAY
            for timestamp in text_timestamps
        ]
        missing_by_feature = {
            feature: sum(not is_present(row.get(feature)) for row in ts_rows)
            for feature in feature_names
        }
        observed_values = sum(
            is_present(row.get(feature)) for row in ts_rows for feature in feature_names
        )

        history = 31.0
        prediction = 31.0
        stride = 31.0
        start = min(relative_days)
        max_time = max(relative_days)
        kept_windows: list[int] = []
        dropped_no_text = 0
        dropped_invalid_numeric = 0
        candidate_windows = 0
        chunk_counter = 0
        while start + history + prediction <= max_time:
            candidate_windows += 1
            indices = [
                index
                for index, time in enumerate(relative_days)
                if start <= time < start + history + prediction
            ]
            if len(indices) < 2:
                dropped_invalid_numeric += 1
                start += stride
                continue
            history_present = any(
                relative_days[index] < start + history
                and any(is_present(ts_rows[index].get(feature)) for feature in feature_names)
                for index in indices
            )
            prediction_present = any(
                relative_days[index] >= start + history
                and any(is_present(ts_rows[index].get(feature)) for feature in feature_names)
                for index in indices
            )
            if not history_present or not prediction_present:
                dropped_invalid_numeric += 1
                start += stride
                continue
            chunk_id = chunk_counter
            chunk_counter += 1
            has_text = any(start <= time < start + history for time in relative_text_days)
            if not has_text:
                dropped_no_text += 1
            else:
                kept_windows.append(chunk_id)
            start += stride

        n_windows = len(kept_windows)
        train_end = int(n_windows * 0.6)
        val_end = int(n_windows * 0.8)
        schedule_types = Counter(row.get("interval_type", "") for row in schedule_rows)
        result_entities.append(
            {
                "entity": entity_dir.name,
                "numeric_rows": len(ts_rows),
                "text_rows": len(text_rows),
                "features": feature_names,
                "observed_values": observed_values,
                "missing_values": sum(missing_by_feature.values()),
                "missing_by_feature": missing_by_feature,
                "numeric_start": min(timestamps).isoformat(),
                "numeric_end": max(timestamps).isoformat(),
                "duplicate_numeric_timestamps": len(timestamps) - len(set(timestamps)),
                "numeric_timestamps_sorted": timestamps == sorted(timestamps),
                "text_start": min(text_timestamps).isoformat() if text_timestamps else None,
                "text_end": max(text_timestamps).isoformat() if text_timestamps else None,
                "schedule_rows": len(schedule_rows),
                "schedule_interval_types": dict(schedule_types),
                "official_window_logic": {
                    "candidate_windows": candidate_windows,
                    "kept_windows": n_windows,
                    "dropped_no_text": dropped_no_text,
                    "dropped_invalid_numeric": dropped_invalid_numeric,
                    "train_windows": train_end,
                    "val_windows": val_end - train_end,
                    "test_windows": n_windows - val_end,
                },
                "source_columns": {
                    "time_series": ts_fields,
                    "text": text_fields,
                    "schedule_intervals": schedule_fields,
                },
            }
        )
        total_rows += len(ts_rows)
        total_text += len(text_rows)

    totals = {
        "entities": len(result_entities),
        "numeric_rows": total_rows,
        "text_rows": total_text,
        "features": sorted(all_feature_names),
        "official_train_windows": sum(
            int(entity["official_window_logic"]["train_windows"]) for entity in result_entities
        ),
        "official_val_windows": sum(
            int(entity["official_window_logic"]["val_windows"]) for entity in result_entities
        ),
        "official_test_windows": sum(
            int(entity["official_window_logic"]["test_windows"]) for entity in result_entities
        ),
    }
    report = {
        "source": str(processed),
        "official_window_days": {"history": 31, "prediction": 31, "stride": 31},
        "totals": totals,
        "entities": result_entities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
