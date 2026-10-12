"""Unify all seed-101 multimodal shortcut-audit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "ClusterTrace": {
        "numeric": "clustertrace_development_seed101_cuda",
        "multimodal": "clustertrace_multimodal_ln_development_seed101_cuda",
        "controls": "clustertrace_multimodal_controls_seed101_cuda",
        "semantic": "clustertrace_semantic_controls_seed101_cuda",
    },
    "CESNET": {
        "numeric": "cesnet_development_seed101_cuda",
        "multimodal": "cesnet_multimodal_ln_development_seed101_cuda",
        "controls": "cesnet_multimodal_controls_seed101_cuda",
        "semantic": "cesnet_semantic_controls_seed101_cuda",
    },
    "EPA-Air": {
        "numeric": "epa_air_development_seed101_cuda",
        "multimodal": "epa_air_multimodal_ln_development_seed101_cuda",
        "controls": "epa_air_multimodal_controls_seed101_cuda",
        "semantic": "epa_air_semantic_controls_seed101_cuda",
    },
    "RepoHealth": {
        "numeric": "repohealth_generic_numeric_development_seed101_cuda",
        "multimodal": "repohealth_multimodal_ln_development_seed101_cuda",
        "controls": "repohealth_multimodal_controls_seed101_cuda",
        "semantic": "repohealth_semantic_controls_seed101_cuda",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("results/shortcut_audit_summary.md")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/shortcut_audit_summary.json"),
    )
    return parser.parse_args()


def result_map(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["model"]: float(item["test_metrics"]["mse"])
        for item in payload["results"]
    }


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    args = parse_args()
    trivial = json.loads(
        (args.root / "results/timeimm_trivial_baselines.json").read_text(
            encoding="utf-8"
        )
    )["datasets"]
    shrinkage = json.loads(
        (args.root / "results/timeimm_shrinkage_baseline.json").read_text(
            encoding="utf-8"
        )
    )["datasets"]
    rows: list[dict[str, object]] = []
    for dataset, directories in DATASETS.items():
        groups = {
            group: result_map(
                args.root / "results" / directory / "results.json"
            )
            for group, directory in directories.items()
        }
        candidates = {
            "train_mean": float(trivial[dataset]["train_mean"]["test"]["mse"]),
            "last_value": float(trivial[dataset]["last_value"]["test"]["mse"]),
            "shrinkage": float(shrinkage[dataset]["test"]["mse"]),
            "numeric_gru": groups["numeric"]["set_decay_gru_matched"],
            "numeric_selective": groups["numeric"][
                "selective_eventmamba_no_reliability"
            ],
            "multimodal_gru": groups["multimodal"]["set_decay_gru_matched"],
            "multimodal_selective": groups["multimodal"][
                "selective_eventmamba_no_reliability"
            ],
            "mean_text_fusion": groups["controls"]["mean_text_fusion"],
            "text_only_gru": groups["controls"]["text_only_set_decay_gru"],
            "mean_semantic_shuffle": groups["semantic"][
                "mean_text_fusion_semantic_shuffle"
            ],
            "selective_semantic_shuffle": groups["semantic"][
                "selective_ssm_semantic_shuffle"
            ],
            "mean_time_only": groups["semantic"][
                "mean_text_fusion_time_only_zero_text"
            ],
            "selective_time_only": groups["semantic"][
                "selective_ssm_time_only_zero_text"
            ],
        }
        best_name, best_mse = min(candidates.items(), key=lambda item: item[1])
        rows.append(
            {
                "dataset": dataset,
                **candidates,
                "selective_vs_multimodal_gru_percent": improvement(
                    candidates["multimodal_gru"], candidates["multimodal_selective"]
                ),
                "selective_vs_mean_fusion_percent": improvement(
                    candidates["mean_text_fusion"],
                    candidates["multimodal_selective"],
                ),
                "selective_semantic_alignment_percent": improvement(
                    candidates["selective_semantic_shuffle"],
                    candidates["multimodal_selective"],
                ),
                "selective_content_vs_time_only_percent": improvement(
                    candidates["selective_time_only"],
                    candidates["multimodal_selective"],
                ),
                "best_model": best_name,
                "best_mse": best_mse,
            }
        )

    payload = {
        "scope": "seed-101 development shortcut audit",
        "warning": (
            "All listed test periods were inspected during development. These are "
            "not untouched external confirmation results."
        ),
        "findings": {
            "selective_beats_matched_multimodal_gru": sum(
                row["multimodal_selective"] < row["multimodal_gru"] for row in rows
            ),
            "selective_beats_naive_mean_fusion": sum(
                row["multimodal_selective"] < row["mean_text_fusion"] for row in rows
            ),
            "selective_is_global_best": sum(
                row["best_model"] == "multimodal_selective" for row in rows
            ),
            "datasets": len(rows),
        },
        "decision": (
            "Do not claim universal multimodal superiority. Freeze the protocol and "
            "run multi-seed confirmation on RepoHealth and ClusterTrace, while "
            "framing EPA-Air/CESNET as shortcut-dominated failure cases."
        ),
        "rows": rows,
    }
    json_output = (args.root / args.json_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Shortcut-aware multimodal development audit",
        "",
        "**Development evidence only:** every listed test period has been inspected.",
        "",
        "| Dataset | Train mean | Last | Shrinkage | MM-GRU | Mean fusion | Selective | Selective shuffled | Selective time-only | Semantic alignment gain | Global best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {train_mean:.4f} | {last_value:.4f} | {shrinkage:.4f} | "
            "{multimodal_gru:.4f} | {mean_text_fusion:.4f} | "
            "{multimodal_selective:.4f} | {selective_semantic_shuffle:.4f} | "
            "{selective_time_only:.4f} | {selective_semantic_alignment_percent:+.2f}% | "
            "{best_model} ({best_mse:.4f}) |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Evidence-bounded conclusion",
            "",
            "The selective model beats the parameter-matched multimodal GRU on",
            f"{payload['findings']['selective_beats_matched_multimodal_gru']}/4 datasets,",
            "but beats naive mean fusion on only",
            f"{payload['findings']['selective_beats_naive_mean_fusion']}/4 and is the global",
            f"best on {payload['findings']['selective_is_global_best']}/4.",
            "ClusterTrace shows a large within-model semantic-alignment effect,",
            "RepoHealth a small positive effect, while EPA-Air and CESNET are",
            "shortcut-dominated. A universal semantic-fusion claim is falsified.",
            "",
            "Next gate: multi-seed confirmation on RepoHealth and ClusterTrace, then",
            "one untouched Time-IMM dataset selected before seeing its targets.",
        ]
    )
    markdown_output = (args.root / args.output).resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
