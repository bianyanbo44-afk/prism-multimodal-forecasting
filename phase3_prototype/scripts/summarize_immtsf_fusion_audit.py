"""Summarize the five-dataset IMM-TSF fusion implementation audit."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DATASETS = {
    "ClusterTrace": "clustertrace_immtsf_fusion_audit_fresh3",
    "CESNET": "cesnet_immtsf_fusion_audit_fresh3",
    "EPA-Air": "epa_air_immtsf_fusion_audit_fresh3",
    "FNSPID": "fnspid_immtsf_fusion_audit_fresh3",
    "RepoHealth": "repohealth_immtsf_fusion_audit_fresh3",
}
VARIANTS = (
    "aligned_unit_consistent",
    "semantic_shuffle_unit_consistent",
    "time_only_unit_consistent",
    "aligned_released_mixed_units",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/immtsf_fusion_audit_summary.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/immtsf_fusion_audit_summary.json"),
    )
    return parser.parse_args()


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    trivial = json.loads(
        (
            root
            / "results"
            / "fnspid_untouched_gate"
            / "trivial_baselines.json"
        ).read_text(encoding="utf-8")
    )["datasets"]
    rows: list[dict[str, object]] = []
    for dataset, folder in DATASETS.items():
        path = root / "results" / folder / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[str, dict[int, float]] = {name: {} for name in VARIANTS}
        epochs: dict[str, dict[int, int]] = {name: {} for name in VARIANTS}
        for item in payload["results"]:
            variant = str(item["variant"])
            if variant not in grouped:
                continue
            seed = int(item["seed"])
            grouped[variant][seed] = float(item["test_metrics"]["mse"])
            epochs[variant][seed] = int(item["best_epoch"])
        if any(set(values) != {53, 67, 79} for values in grouped.values()):
            raise ValueError(f"Incomplete three-seed audit in {path}")
        means = {
            name: statistics.mean(values.values())
            for name, values in grouped.items()
        }
        sds = {
            name: statistics.stdev(values.values())
            for name, values in grouped.items()
        }
        aligned = grouped["aligned_unit_consistent"]
        shuffled = grouped["semantic_shuffle_unit_consistent"]
        time_only = grouped["time_only_unit_consistent"]
        last = float(trivial[dataset]["last_value"]["test"]["mse"])
        rows.append(
            {
                "dataset": dataset,
                "source": str(path),
                "parameters": int(
                    payload["prepacked_cache"]["aligned_unit_consistent"][
                        "parameters"
                    ]
                ),
                "last_value_mse": last,
                "means": means,
                "sample_sds": sds,
                "by_seed": {
                    name: {str(seed): value for seed, value in sorted(values.items())}
                    for name, values in grouped.items()
                },
                "best_epochs": {
                    name: {str(seed): value for seed, value in sorted(values.items())}
                    for name, values in epochs.items()
                },
                "aligned_vs_shuffle_percent": gain(
                    means["semantic_shuffle_unit_consistent"],
                    means["aligned_unit_consistent"],
                ),
                "aligned_vs_time_only_percent": gain(
                    means["time_only_unit_consistent"],
                    means["aligned_unit_consistent"],
                ),
                "aligned_vs_last_value_percent": gain(
                    last, means["aligned_unit_consistent"]
                ),
                "released_vs_unit_consistent_percent": gain(
                    means["aligned_unit_consistent"],
                    means["aligned_released_mixed_units"],
                ),
                "aligned_seed_wins_vs_shuffle": sum(
                    aligned[seed] < shuffled[seed] for seed in aligned
                ),
                "aligned_seed_wins_vs_time_only": sum(
                    aligned[seed] < time_only[seed] for seed in aligned
                ),
            }
        )

    semantic_dataset_wins = sum(
        float(row["aligned_vs_shuffle_percent"]) > 0 for row in rows
    )
    time_only_dataset_wins = sum(
        float(row["aligned_vs_time_only_percent"]) > 0 for row in rows
    )
    maximum_unit_effect = max(
        rows, key=lambda row: abs(float(row["released_vs_unit_consistent_percent"]))
    )
    payload = {
        "analysis": "post-gate exploratory audit of the official IMM-TSF fusion modules",
        "official_modules": "TTF_RecAvg + MMF_GR_Add",
        "protocol": (
            "pinned official MIT modules over a persistence forecast, train-only "
            "normalization, chronological split, frozen full GPT-2 embeddings"
        ),
        "seeds": [53, 67, 79],
        "semantic_alignment_dataset_wins": f"{semantic_dataset_wins}/{len(rows)}",
        "aligned_vs_time_only_dataset_wins": f"{time_only_dataset_wins}/{len(rows)}",
        "largest_mixed_unit_effect": {
            "dataset": maximum_unit_effect["dataset"],
            "percent": maximum_unit_effect["released_vs_unit_consistent_percent"],
        },
        "interpretation": (
            "Aligned semantics beat shuffled semantics on only 2/5 datasets, and "
            "those two gains are negligible. The 1.20M-parameter fusion often "
            "collapses close to persistence. On EPA-Air, reproducing the released "
            "mixed-unit time path changes mean MSE by more than the semantic-control "
            "effect, demonstrating protocol sensitivity."
        ),
        "statistical_caveat": "n=3 seeds; all comparisons are descriptive.",
        "rows": rows,
    }
    json_output = (root / args.json_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# IMM-TSF fusion audit",
        "",
        "Post-gate exploratory audit using the pinned official MIT modules",
        "`TTF_RecAvg + MMF_GR_Add` under the leakage-free common protocol.",
        "All entries are mean test MSE ± sample SD over seeds 53, 67, 79.",
        "",
        "| Dataset | Last value | Aligned, consistent | Semantic shuffle | Time-only | Released mixed units | Aligned vs shuffle | Aligned vs last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        means = row["means"]
        sds = row["sample_sds"]
        lines.append(
            f"| {row['dataset']} | {row['last_value_mse']:.6f} | "
            f"{means['aligned_unit_consistent']:.6f} ± {sds['aligned_unit_consistent']:.6f} | "
            f"{means['semantic_shuffle_unit_consistent']:.6f} ± {sds['semantic_shuffle_unit_consistent']:.6f} | "
            f"{means['time_only_unit_consistent']:.6f} ± {sds['time_only_unit_consistent']:.6f} | "
            f"{means['aligned_released_mixed_units']:.6f} ± {sds['aligned_released_mixed_units']:.6f} | "
            f"{row['aligned_vs_shuffle_percent']:+.3f}% | "
            f"{row['aligned_vs_last_value_percent']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Audit findings",
            "",
            f"- Aligned semantics beat semantic shuffle on {semantic_dataset_wins}/5 datasets.",
            f"- Aligned semantics beat time-only on {time_only_dataset_wins}/5 datasets.",
            f"- The largest released-vs-consistent unit effect is "
            f"{maximum_unit_effect['released_vs_unit_consistent_percent']:+.3f}% on "
            f"{maximum_unit_effect['dataset']}.",
            "- The fusion module has 1,201,501 trainable parameters in FNSPID, yet",
            "  frequently converges to a prediction nearly identical to last-value.",
            "",
            payload["interpretation"],
            "",
            "These are descriptive post-gate analyses, not untouched confirmation results.",
        ]
    )
    markdown_output = (root / args.output).resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
