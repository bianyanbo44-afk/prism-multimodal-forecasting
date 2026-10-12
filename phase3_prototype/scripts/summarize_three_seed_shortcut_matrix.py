"""Create the unified five-dataset, three-seed shortcut-audit matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (53, 67, 79)
DATASETS = {
    "ClusterTrace": {
        "numeric": "clustertrace_numeric_fresh3_cuda/results.json",
        "multimodal": "clustertrace_multimodal_ln_fresh3_cuda/results.json",
        "controls": "clustertrace_multimodal_controls_fresh3_cuda/results.json",
        "semantic": "clustertrace_semantic_controls_fresh3_cuda/results.json",
        "evidence_tier": "development",
    },
    "CESNET": {
        "numeric": "cesnet_numeric_fresh3_cuda/results.json",
        "multimodal": "cesnet_multimodal_ln_fresh3_cuda/results.json",
        "controls": "cesnet_multimodal_controls_fresh3_cuda/results.json",
        "semantic": "cesnet_semantic_controls_fresh3_cuda/results.json",
        "evidence_tier": "development",
    },
    "EPA-Air": {
        "numeric": "epa_air_numeric_fresh3_cuda/results.json",
        "multimodal": "epa_air_multimodal_ln_fresh3_cuda/results.json",
        "controls": "epa_air_multimodal_controls_fresh3_cuda/results.json",
        "semantic": "epa_air_semantic_controls_fresh3_cuda/results.json",
        "evidence_tier": "development",
    },
    "FNSPID": {
        "numeric": "fnspid_untouched_gate/numeric_models/results.json",
        "multimodal": "fnspid_untouched_gate/multimodal_models/results.json",
        "controls": "fnspid_untouched_gate/multimodal_controls/results.json",
        "semantic": "fnspid_untouched_gate/semantic_controls/results.json",
        "evidence_tier": "preregistered_untouched_gate",
    },
    "RepoHealth": {
        "numeric": "repohealth_generic_numeric_fresh3_cuda/results.json",
        "multimodal": "repohealth_multimodal_ln_fresh3_cuda/results.json",
        "controls": "repohealth_multimodal_controls_fresh3_cuda/results.json",
        "semantic": "repohealth_semantic_controls_fresh3_cuda/results.json",
        "evidence_tier": "development",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/three_seed_shortcut_matrix.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/three_seed_shortcut_matrix.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_values(payload: dict[str, object], name: str) -> dict[int, float]:
    values = {
        int(item["seed"]): float(item["test_metrics"]["mse"])
        for item in payload["results"]
        if item["model"] == name and item.get("seed") is not None
    }
    if tuple(sorted(values)) != SEEDS:
        raise ValueError(f"{name}: got seeds {sorted(values)}, expected {SEEDS}")
    return values


def describe(values: dict[int, float]) -> dict[str, object]:
    ordered = [values[seed] for seed in SEEDS]
    return {
        "by_seed": {str(seed): values[seed] for seed in SEEDS},
        "mean": statistics.mean(ordered),
        "sample_sd": statistics.stdev(ordered),
    }


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def seed_wins(candidate: dict[int, float], reference: dict[int, float]) -> int:
    return sum(candidate[seed] < reference[seed] for seed in SEEDS)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_root = root / "results"
    trivial = load(
        results_root / "fnspid_untouched_gate" / "trivial_baselines.json"
    )["datasets"]
    shrinkage = load(
        results_root / "fnspid_untouched_gate" / "shrinkage_baseline.json"
    )["datasets"]
    official = load(results_root / "immtsf_fusion_audit_summary.json")
    official_rows = {row["dataset"]: row for row in official["rows"]}

    rows: list[dict[str, object]] = []
    for dataset, paths in DATASETS.items():
        numeric = load(results_root / str(paths["numeric"]))
        multimodal = load(results_root / str(paths["multimodal"]))
        controls = load(results_root / str(paths["controls"]))
        semantic = load(results_root / str(paths["semantic"]))
        values = {
            "numeric_gru": model_values(numeric, "set_decay_gru_matched"),
            "numeric_selective": model_values(
                numeric, "selective_eventmamba_no_reliability"
            ),
            "multimodal_gru": model_values(multimodal, "set_decay_gru_matched"),
            "aligned_selective": model_values(
                multimodal, "selective_eventmamba_no_reliability"
            ),
            "naive_mean_fusion": model_values(controls, "mean_text_fusion"),
            "text_only_gru": model_values(controls, "text_only_set_decay_gru"),
            "selective_semantic_shuffle": model_values(
                semantic, "selective_ssm_semantic_shuffle"
            ),
            "selective_time_only": model_values(
                semantic, "selective_ssm_time_only_zero_text"
            ),
            "mean_semantic_shuffle": model_values(
                semantic, "mean_text_fusion_semantic_shuffle"
            ),
            "mean_time_only": model_values(
                semantic, "mean_text_fusion_time_only_zero_text"
            ),
        }
        descriptions = {name: describe(series) for name, series in values.items()}
        means = {name: float(item["mean"]) for name, item in descriptions.items()}
        aligned = values["aligned_selective"]
        shuffled = values["selective_semantic_shuffle"]
        time_only = values["selective_time_only"]
        numeric_selective = values["numeric_selective"]
        naive = values["naive_mean_fusion"]
        matched_gru = values["multimodal_gru"]
        effects = {
            "semantic_alignment_gain_percent": gain(
                means["selective_semantic_shuffle"], means["aligned_selective"]
            ),
            "content_beyond_timing_gain_percent": gain(
                means["selective_time_only"], means["aligned_selective"]
            ),
            "multimodal_utility_percent": gain(
                means["numeric_selective"], means["aligned_selective"]
            ),
            "aligned_vs_naive_percent": gain(
                means["naive_mean_fusion"], means["aligned_selective"]
            ),
            "aligned_vs_multimodal_gru_percent": gain(
                means["multimodal_gru"], means["aligned_selective"]
            ),
        }
        wins = {
            "aligned_vs_shuffle": seed_wins(aligned, shuffled),
            "aligned_vs_time_only": seed_wins(aligned, time_only),
            "aligned_vs_numeric": seed_wins(aligned, numeric_selective),
            "aligned_vs_naive": seed_wins(aligned, naive),
            "aligned_vs_multimodal_gru": seed_wins(aligned, matched_gru),
        }
        semantic_support = (
            effects["semantic_alignment_gain_percent"] > 0
            and effects["content_beyond_timing_gain_percent"] > 0
            and wins["aligned_vs_shuffle"] >= 2
            and wins["aligned_vs_time_only"] >= 2
        )
        static = {
            "last_value": float(trivial[dataset]["last_value"]["test"]["mse"]),
            "train_mean": float(trivial[dataset]["train_mean"]["test"]["mse"]),
            "shrinkage": float(shrinkage[dataset]["test"]["mse"]),
        }
        candidates = {
            "last_value": static["last_value"],
            "shrinkage": static["shrinkage"],
            **means,
        }
        best_name, best_mse = min(candidates.items(), key=lambda item: item[1])
        official_row = official_rows[dataset]
        official_semantic_support = (
            float(official_row["aligned_vs_shuffle_percent"]) > 0
            and float(official_row["aligned_vs_time_only_percent"]) > 0
        )
        rows.append(
            {
                "dataset": dataset,
                "evidence_tier": paths["evidence_tier"],
                "models": descriptions,
                "static_baselines": static,
                "effects": effects,
                "paired_seed_wins": wins,
                "selective_semantic_support": semantic_support,
                "official_semantic_support": official_semantic_support,
                "both_architectures_support_semantics": (
                    semantic_support and official_semantic_support
                ),
                "best_model": best_name,
                "best_mse": best_mse,
                "aligned_is_best": best_name == "aligned_selective",
                "official_effects": {
                    "semantic_alignment_gain_percent": official_row[
                        "aligned_vs_shuffle_percent"
                    ],
                    "content_beyond_timing_gain_percent": official_row[
                        "aligned_vs_time_only_percent"
                    ],
                    "released_vs_consistent_percent": official_row[
                        "released_vs_unit_consistent_percent"
                    ],
                },
            }
        )

    semantic_support_count = sum(
        bool(row["selective_semantic_support"]) for row in rows
    )
    aligned_best_count = sum(bool(row["aligned_is_best"]) for row in rows)
    cross_architecture_count = sum(
        bool(row["both_architectures_support_semantics"]) for row in rows
    )
    payload = {
        "analysis": "unified five-dataset paired three-seed shortcut matrix",
        "seeds": list(SEEDS),
        "metrics": {
            "semantic_alignment_gain": "100*(MSE_shuffle-MSE_aligned)/MSE_shuffle",
            "content_beyond_timing_gain": "100*(MSE_time_only-MSE_aligned)/MSE_time_only",
            "multimodal_utility": "100*(MSE_numeric-MSE_multimodal)/MSE_numeric",
            "positive": "lower aligned MSE",
        },
        "summary": {
            "selective_semantic_support_datasets": f"{semantic_support_count}/5",
            "aligned_selective_is_best_datasets": f"{aligned_best_count}/5",
            "cross_architecture_semantic_support_datasets": f"{cross_architecture_count}/5",
            "conclusion": (
                "No dataset provides cross-architecture support that aligned text "
                "semantics beat both semantic-shuffle and time-only controls. The "
                "aligned selective model is never the best tested comparator."
            ),
        },
        "statistical_caveat": (
            "Three seeds permit paired descriptive effects and win counts, not "
            "a statistical-significance claim. FNSPID alone is preregistered and untouched."
        ),
        "rows": rows,
    }
    json_output = (root / args.json_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Unified three-seed shortcut matrix",
        "",
        "All learned entries use seeds 53, 67, 79 and the same train-only",
        "normalization, chronological split, and training budget. Lower MSE is better.",
        "FNSPID is the preregistered untouched gate; all other rows are development evidence.",
        "",
        "| Dataset | Numeric selective | Aligned multimodal | Semantic shuffle | Time-only | Naive fusion | SAG | CBTG | MMU | Semantic support | Best comparator |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        models = row["models"]
        effects = row["effects"]
        lines.append(
            f"| {row['dataset']} | "
            f"{models['numeric_selective']['mean']:.6f} ± {models['numeric_selective']['sample_sd']:.6f} | "
            f"{models['aligned_selective']['mean']:.6f} ± {models['aligned_selective']['sample_sd']:.6f} | "
            f"{models['selective_semantic_shuffle']['mean']:.6f} ± {models['selective_semantic_shuffle']['sample_sd']:.6f} | "
            f"{models['selective_time_only']['mean']:.6f} ± {models['selective_time_only']['sample_sd']:.6f} | "
            f"{models['naive_mean_fusion']['mean']:.6f} ± {models['naive_mean_fusion']['sample_sd']:.6f} | "
            f"{effects['semantic_alignment_gain_percent']:+.3f}% | "
            f"{effects['content_beyond_timing_gain_percent']:+.3f}% | "
            f"{effects['multimodal_utility_percent']:+.3f}% | "
            f"{'Yes' if row['selective_semantic_support'] else 'No'} | "
            f"{row['best_model']} ({row['best_mse']:.6f}) |"
        )
    lines.extend(
        [
            "",
            "SAG = semantic alignment gain; CBTG = content-beyond-timing gain;",
            "MMU = multimodal utility over the numeric selective model. Positive is favorable.",
            "",
            "## Main findings",
            "",
            f"- Selective-model semantic support: {semantic_support_count}/5 datasets.",
            f"- Aligned selective model is the best tested comparator: {aligned_best_count}/5 datasets.",
            f"- Both selective and official IMM-TSF fusion support semantics: {cross_architecture_count}/5 datasets.",
            "- On the preregistered FNSPID gate, aligned text loses to both semantic",
            "  shuffle and time-only despite beating GRU and naive-fusion performance baselines.",
            "",
            payload["summary"]["conclusion"],
            "",
            "No universal superiority or significance claim is made.",
        ]
    )
    markdown_output = (root / args.output).resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
