"""Apply the frozen decision rule to the Fidel-TS positive control."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    source_path = ROOT / "results" / "fidel_tgtsf_positive_control" / "results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    variants = ("aligned", "semantic_shuffle", "zero")
    grouped: dict[str, dict[int, float]] = {name: {} for name in variants}
    for result in source["results"]:
        grouped[str(result["variant"])][int(result["seed"])] = float(
            result["test_metrics"]["mse"]
        )
    if any(set(values) != {53, 67, 79} for values in grouped.values()):
        raise ValueError("Positive-control result matrix is incomplete")
    means = {name: statistics.mean(values.values()) for name, values in grouped.items()}
    sds = {name: statistics.stdev(values.values()) for name, values in grouped.items()}
    aligned = grouped["aligned"]
    shuffle_wins = sum(
        aligned[seed] < grouped["semantic_shuffle"][seed] for seed in aligned
    )
    zero_wins = sum(aligned[seed] < grouped["zero"][seed] for seed in aligned)
    control_mean = (means["semantic_shuffle"] + means["zero"]) / 2.0
    combined_gain = gain(control_mean, means["aligned"])
    passed = (
        means["aligned"] < means["semantic_shuffle"]
        and means["aligned"] < means["zero"]
        and shuffle_wins >= 2
        and zero_wins >= 2
        and combined_gain >= 2.0
    )
    payload = {
        "analysis": "preregistered independent positive semantic control",
        "dataset": "VEWOXIC/Canada_photovoltaics_plants",
        "model": "official TGTSF",
        "seeds": [53, 67, 79],
        "means": means,
        "sample_sds": sds,
        "by_seed": {
            name: {str(seed): value for seed, value in sorted(values.items())}
            for name, values in grouped.items()
        },
        "aligned_seed_wins_vs_shuffle": shuffle_wins,
        "aligned_seed_wins_vs_zero": zero_wins,
        "aligned_vs_shuffle_percent": gain(
            means["semantic_shuffle"], means["aligned"]
        ),
        "aligned_vs_zero_percent": gain(means["zero"], means["aligned"]),
        "aligned_vs_control_mean_percent": combined_gain,
        "required_combined_gain_percent": 2.0,
        "decision": "PASS" if passed else "FAIL",
    }
    json_path = ROOT / "results" / "fidel_positive_control_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fidel-TS independent positive semantic control",
        "",
        "Mean normalized test MSE ± sample SD over seeds 53, 67, and 79.",
        "",
        "| Condition | MSE | Seed values |",
        "|---|---:|---|",
    ]
    for name in variants:
        values = ", ".join(
            f"{seed}: {value:.6f}" for seed, value in sorted(grouped[name].items())
        )
        lines.append(f"| {name} | {means[name]:.6f} ± {sds[name]:.6f} | {values} |")
    lines.extend(
        [
            "",
            f"- Aligned versus shuffle: {payload['aligned_vs_shuffle_percent']:+.2f}% "
            f"and {shuffle_wins}/3 paired-seed wins.",
            f"- Aligned versus zero: {payload['aligned_vs_zero_percent']:+.2f}% "
            f"and {zero_wins}/3 paired-seed wins.",
            f"- Aligned versus the two-control mean: {combined_gain:+.2f}% "
            "(2% required).",
            "",
            f"Preregistered decision: **{payload['decision']}**.",
        ]
    )
    (ROOT / "results" / "fidel_positive_control_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
