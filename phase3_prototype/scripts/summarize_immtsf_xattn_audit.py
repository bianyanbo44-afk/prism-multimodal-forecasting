"""Summarize the preregistered five-dataset stronger fusion audit."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "ClusterTrace": "clustertrace_immtsf_xattn_audit_fresh3",
    "CESNET": "cesnet_immtsf_xattn_audit_fresh3",
    "EPA-Air": "epa_air_immtsf_xattn_audit_fresh3",
    "FNSPID": "fnspid_immtsf_xattn_audit_fresh3",
    "RepoHealth": "repohealth_immtsf_xattn_audit_fresh3",
}
VARIANTS = ("aligned", "semantic_shuffle", "time_only")


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    rows: list[dict[str, object]] = []
    for dataset, folder in DATASETS.items():
        path = ROOT / "results" / folder / "results.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[str, dict[int, float]] = {name: {} for name in VARIANTS}
        for result in source["results"]:
            grouped[str(result["variant"])][int(result["seed"])] = float(
                result["test_metrics"]["mse"]
            )
        if any(set(values) != {53, 67, 79} for values in grouped.values()):
            raise ValueError(f"Incomplete audit: {path}")
        means = {name: statistics.mean(values.values()) for name, values in grouped.items()}
        sds = {name: statistics.stdev(values.values()) for name, values in grouped.items()}
        aligned = grouped["aligned"]
        shuffle = grouped["semantic_shuffle"]
        time_only = grouped["time_only"]
        shuffle_seed_wins = sum(aligned[seed] < shuffle[seed] for seed in aligned)
        time_seed_wins = sum(aligned[seed] < time_only[seed] for seed in aligned)
        dataset_support = (
            means["aligned"] < means["semantic_shuffle"]
            and means["aligned"] < means["time_only"]
            and shuffle_seed_wins >= 2
            and time_seed_wins >= 2
        )
        rows.append(
            {
                "dataset": dataset,
                "source": str(path),
                "parameters": int(source["prepacked_cache"]["aligned"]["parameters"]),
                "means": means,
                "sample_sds": sds,
                "by_seed": {
                    name: {str(seed): value for seed, value in sorted(values.items())}
                    for name, values in grouped.items()
                },
                "aligned_vs_shuffle_percent": gain(means["semantic_shuffle"], means["aligned"]),
                "aligned_vs_time_only_percent": gain(means["time_only"], means["aligned"]),
                "aligned_seed_wins_vs_shuffle": shuffle_seed_wins,
                "aligned_seed_wins_vs_time_only": time_seed_wins,
                "preregistered_dataset_support": dataset_support,
            }
        )

    support_count = sum(bool(row["preregistered_dataset_support"]) for row in rows)
    passed = support_count >= 3
    payload = {
        "analysis": "preregistered stronger public fusion-family audit",
        "official_modules": "TTF_T2V_XAttn + MMF_XAttn_Add",
        "seeds": [53, 67, 79],
        "supporting_datasets": support_count,
        "required_datasets": 3,
        "decision": "PASS" if passed else "FAIL",
        "interpretation": (
            "The stronger public attention fusion overturns the original negative conclusion."
            if passed
            else "The stronger public attention fusion does not overturn the original negative conclusion."
        ),
        "rows": rows,
    }
    json_path = ROOT / "results" / "immtsf_xattn_audit_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stronger IMM-TSF fusion-family audit",
        "",
        "Mean normalized test MSE ± sample SD over seeds 53, 67, and 79.",
        "",
        "| Dataset | Aligned | Shuffle | Time-only | A vs S | A vs T | Seed wins S/T | Support |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        means = row["means"]
        sds = row["sample_sds"]
        lines.append(
            f"| {row['dataset']} | {means['aligned']:.6f} ± {sds['aligned']:.6f} | "
            f"{means['semantic_shuffle']:.6f} ± {sds['semantic_shuffle']:.6f} | "
            f"{means['time_only']:.6f} ± {sds['time_only']:.6f} | "
            f"{row['aligned_vs_shuffle_percent']:+.3f}% | "
            f"{row['aligned_vs_time_only_percent']:+.3f}% | "
            f"{row['aligned_seed_wins_vs_shuffle']}/"
            f"{row['aligned_seed_wins_vs_time_only']} | "
            f"{'yes' if row['preregistered_dataset_support'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Preregistered decision: **{payload['decision']}** "
            f"({support_count}/5 supporting datasets; 3/5 required).",
            "",
            str(payload["interpretation"]),
        ]
    )
    (ROOT / "results" / "immtsf_xattn_audit_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
