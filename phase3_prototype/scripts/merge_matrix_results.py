"""Merge non-overlapping condition-metric tables from paired matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "condition_metrics.csv"
    manifest_path = output_dir / "merge_manifest.json"
    if table_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite merged results in {output_dir}")

    rows: list[dict[str, str]] = []
    sources: list[str] = []
    fieldnames: list[str] | None = None
    for input_dir in args.inputs:
        source = input_dir.resolve() / "condition_metrics.csv"
        with source.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"Column mismatch in {source}")
            rows.extend(reader)
        sources.append(str(source))

    keys = [(row["seed"], row["model"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed-model-condition rows across input matrices")
    rows.sort(key=lambda row: (int(row["seed"]), row["model"], row["condition"]))
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "sources": sources,
        "rows": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "conditions": sorted({row["condition"] for row in rows}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
