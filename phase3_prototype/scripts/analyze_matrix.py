"""Aggregate paired robustness-matrix outputs without inferential overclaiming."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--full-model", default="eventmamba_full")
    return parser.parse_args()


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    source_path = input_dir / "condition_metrics.csv"
    output_paths = {
        "aggregate": input_dir / "aggregate_metrics.csv",
        "degradation": input_dir / "robustness_degradation.csv",
        "paired": input_dir / "paired_differences.csv",
        "summary": input_dir / "analysis_summary.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError(f"Refusing to overwrite an existing analysis in {input_dir}")

    with source_path.open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        {
            **row,
            "seed": int(row["seed"]),
            "mse": float(row["mse"]),
        }
        for row in raw_rows
    ]
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    seeds = sorted({int(row["seed"]) for row in rows})
    expected = len(models) * len(conditions) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"Expected a complete {expected}-cell matrix, found {len(rows)} rows")

    lookup = {
        (int(row["seed"]), str(row["model"]), str(row["condition"])): float(row["mse"])
        for row in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("Duplicate seed-model-condition rows detected")

    aggregate_rows: list[dict[str, object]] = []
    for model in models:
        for condition in conditions:
            values = [lookup[(seed, model, condition)] for seed in seeds]
            aggregate_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "n_seeds": len(values),
                    "mean_mse": statistics.fmean(values),
                    "sd_mse": sample_sd(values),
                    "min_mse": min(values),
                    "max_mse": max(values),
                }
            )

    degradation_rows: list[dict[str, object]] = []
    for model in models:
        for condition in conditions:
            if condition == "normal":
                continue
            percentages = [
                100.0
                * (lookup[(seed, model, condition)] - lookup[(seed, model, "normal")])
                / lookup[(seed, model, "normal")]
                for seed in seeds
            ]
            degradation_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "n_seeds": len(percentages),
                    "mean_degradation_pct": statistics.fmean(percentages),
                    "sd_degradation_pct": sample_sd(percentages),
                    "min_degradation_pct": min(percentages),
                    "max_degradation_pct": max(percentages),
                }
            )

    full_model = args.full_model
    if full_model not in models:
        raise ValueError(f"Required full model is missing: {full_model}")
    paired_rows: list[dict[str, object]] = []
    for comparator in models:
        if comparator == full_model:
            continue
        for condition in conditions:
            differences = [
                lookup[(seed, full_model, condition)]
                - lookup[(seed, comparator, condition)]
                for seed in seeds
            ]
            relative = [
                100.0 * difference / lookup[(seed, comparator, condition)]
                for seed, difference in zip(seeds, differences)
            ]
            paired_rows.append(
                {
                    "comparator": comparator,
                    "condition": condition,
                    "n_seeds": len(differences),
                    "mean_full_minus_comparator": statistics.fmean(differences),
                    "sd_full_minus_comparator": sample_sd(differences),
                    "mean_relative_difference_pct": statistics.fmean(relative),
                    "full_wins": sum(difference < 0.0 for difference in differences),
                }
            )

    macro_mse = {
        model: statistics.fmean(
            lookup[(seed, model, condition)]
            for seed in seeds
            for condition in conditions
        )
        for model in models
    }
    win_counts = defaultdict(int)
    for seed in seeds:
        for condition in conditions:
            winner = min(models, key=lambda model: lookup[(seed, model, condition)])
            win_counts[winner] += 1

    write_csv(output_paths["aggregate"], aggregate_rows)
    write_csv(output_paths["degradation"], degradation_rows)
    write_csv(output_paths["paired"], paired_rows)
    summary = {
        "source": str(source_path),
        "seeds": seeds,
        "models": models,
        "conditions": conditions,
        "macro_mean_mse": macro_mse,
        "win_counts_out_of_seed_conditions": dict(win_counts),
        "aggregate": aggregate_rows,
        "robustness_degradation": degradation_rows,
        "paired_differences": paired_rows,
        "inference_policy": "Descriptive only; n=3 seeds, no p-values or significance claims.",
    }
    output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "macro_mean_mse": macro_mse,
                "win_counts_out_of_seed_conditions": dict(win_counts),
                "outputs": {key: str(path) for key, path in output_paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
