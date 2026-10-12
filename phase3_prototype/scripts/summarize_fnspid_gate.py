"""Summarize the preregistered FNSPID gate after blind execution."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (53, 67, 79)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/fnspid_untouched_gate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fnspid_untouched_gate/gate_summary.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/fnspid_untouched_gate/gate_summary.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_mse(path: Path, model: str) -> dict[int, float]:
    payload = load(path)
    values = {
        int(item["seed"]): float(item["test_metrics"]["mse"])
        for item in payload["results"]
        if item["model"] == model and item.get("seed") is not None
    }
    if tuple(sorted(values)) != SEEDS:
        raise ValueError(f"{path}: {model} has seeds {sorted(values)}, expected {SEEDS}")
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


def wins(candidate: dict[int, float], reference: dict[int, float]) -> int:
    return sum(candidate[seed] < reference[seed] for seed in SEEDS)


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    numeric_path = root / "numeric_models" / "results.json"
    multimodal_path = root / "multimodal_models" / "results.json"
    control_path = root / "multimodal_controls" / "results.json"
    semantic_path = root / "semantic_controls" / "results.json"

    series = {
        "numeric_gru": seed_mse(numeric_path, "set_decay_gru_matched"),
        "numeric_selective": seed_mse(
            numeric_path, "selective_eventmamba_no_reliability"
        ),
        "multimodal_gru": seed_mse(multimodal_path, "set_decay_gru_matched"),
        "multimodal_selective": seed_mse(
            multimodal_path, "selective_eventmamba_no_reliability"
        ),
        "mean_text_fusion": seed_mse(control_path, "mean_text_fusion"),
        "text_only_gru": seed_mse(control_path, "text_only_set_decay_gru"),
        "mean_semantic_shuffle": seed_mse(
            semantic_path, "mean_text_fusion_semantic_shuffle"
        ),
        "selective_semantic_shuffle": seed_mse(
            semantic_path, "selective_ssm_semantic_shuffle"
        ),
        "mean_time_only": seed_mse(
            semantic_path, "mean_text_fusion_time_only_zero_text"
        ),
        "selective_time_only": seed_mse(
            semantic_path, "selective_ssm_time_only_zero_text"
        ),
    }
    descriptions = {name: describe(values) for name, values in series.items()}
    means = {name: float(item["mean"]) for name, item in descriptions.items()}

    candidate = series["multimodal_selective"]
    matched_gru = series["multimodal_gru"]
    naive = series["mean_text_fusion"]
    shuffled = series["selective_semantic_shuffle"]
    time_only = series["selective_time_only"]
    candidate_gru_wins = wins(candidate, matched_gru)
    candidate_naive_wins = wins(candidate, naive)

    conditions = [
        {
            "id": 1,
            "rule": "candidate mean MSE < parameter-matched multimodal GRU",
            "passed": means["multimodal_selective"] < means["multimodal_gru"],
        },
        {
            "id": 2,
            "rule": "candidate mean MSE < naive mean-text fusion",
            "passed": means["multimodal_selective"] < means["mean_text_fusion"],
        },
        {
            "id": 3,
            "rule": "candidate wins at least 2/3 seeds against both comparators",
            "passed": candidate_gru_wins >= 2 and candidate_naive_wins >= 2,
            "candidate_vs_gru_seed_wins": candidate_gru_wins,
            "candidate_vs_naive_seed_wins": candidate_naive_wins,
        },
        {
            "id": 4,
            "rule": "aligned text mean MSE < semantic shuffle",
            "passed": (
                means["multimodal_selective"]
                < means["selective_semantic_shuffle"]
            ),
        },
        {
            "id": 5,
            "rule": "aligned text mean MSE < time-only zero-text control",
            "passed": means["multimodal_selective"] < means["selective_time_only"],
        },
    ]
    overall_pass = all(bool(item["passed"]) for item in conditions)

    trivial = load(root / "trivial_baselines.json")["datasets"]["FNSPID"]
    shrinkage = load(root / "shrinkage_baseline.json")["datasets"]["FNSPID"]
    manifest = load(root / "execution_manifest.json")
    static_baselines = {
        "train_mean": float(trivial["train_mean"]["test"]["mse"]),
        "last_value": float(trivial["last_value"]["test"]["mse"]),
        "shrinkage_persistence": float(shrinkage["test"]["mse"]),
        "shrinkage_weights": shrinkage["weights"],
    }
    effects = {
        "candidate_vs_multimodal_gru_percent": gain(
            means["multimodal_gru"], means["multimodal_selective"]
        ),
        "candidate_vs_naive_fusion_percent": gain(
            means["mean_text_fusion"], means["multimodal_selective"]
        ),
        "aligned_vs_semantic_shuffle_percent": gain(
            means["selective_semantic_shuffle"], means["multimodal_selective"]
        ),
        "aligned_vs_time_only_percent": gain(
            means["selective_time_only"], means["multimodal_selective"]
        ),
        "multimodal_selective_vs_numeric_selective_percent": gain(
            means["numeric_selective"], means["multimodal_selective"]
        ),
        "multimodal_gru_vs_numeric_gru_percent": gain(
            means["numeric_gru"], means["multimodal_gru"]
        ),
        "mean_aligned_vs_semantic_shuffle_percent": gain(
            means["mean_semantic_shuffle"], means["mean_text_fusion"]
        ),
        "mean_aligned_vs_time_only_percent": gain(
            means["mean_time_only"], means["mean_text_fusion"]
        ),
    }
    payload = {
        "gate": "preregistered untouched FNSPID confirmation",
        "status": "PASS" if overall_pass else "FAIL",
        "seeds": list(SEEDS),
        "statistical_caveat": (
            "Three seeds support descriptive means, sample SDs, and paired wins; "
            "no significance claim is made."
        ),
        "conditions": conditions,
        "models": descriptions,
        "static_baselines": static_baselines,
        "effects": effects,
        "blind_execution": {
            "status": manifest.get("status"),
            "intermediate_metrics_exposed": manifest.get(
                "intermediate_metrics_exposed"
            ),
            "duration_seconds": sum(
                float(item["duration_seconds"]) for item in manifest["steps"]
            ),
            "config_sha256": manifest["config_sha256"],
            "preregistration_sha256": manifest["preregistration_sha256"],
        },
        "interpretation": (
            "The candidate satisfies predictive-comparison conditions 1-3 but "
            "fails semantic conditions 4-5: shuffled semantics and zero text "
            "content perform better on average. The preregistered semantic-fusion "
            "claim is falsified; the result supports a timing/count shortcut audit."
        ),
    }

    json_output = args.json_output.resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "numeric_gru": "Numeric matched GRU",
        "numeric_selective": "Numeric selective SSM",
        "multimodal_gru": "Multimodal matched GRU",
        "multimodal_selective": "Aligned multimodal selective SSM",
        "mean_text_fusion": "Naive mean-text fusion",
        "text_only_gru": "Text-only GRU",
        "mean_semantic_shuffle": "Mean fusion + semantic shuffle",
        "selective_semantic_shuffle": "Selective SSM + semantic shuffle",
        "mean_time_only": "Mean fusion + time-only",
        "selective_time_only": "Selective SSM + time-only",
    }
    lines = [
        "# FNSPID preregistered gate",
        "",
        f"**Decision: {'PASS' if overall_pass else 'FAIL'}**",
        "",
        "The entire batch completed before any model test metric was opened. All",
        "figures below are descriptive across the preregistered seeds 53, 67, 79.",
        "",
        "## Gate conditions",
        "",
        "| # | Frozen condition | Result |",
        "|---:|---|:---:|",
    ]
    for condition in conditions:
        lines.append(
            f"| {condition['id']} | {condition['rule']} | "
            f"{'PASS' if condition['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Model results",
            "",
            "| Model | Mean MSE ± sample SD | Seed 53 | Seed 67 | Seed 79 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in labels.items():
        item = descriptions[key]
        by_seed = item["by_seed"]
        lines.append(
            f"| {label} | {item['mean']:.6f} ± {item['sample_sd']:.6f} | "
            f"{by_seed['53']:.6f} | {by_seed['67']:.6f} | {by_seed['79']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Static baselines: train mean "
            f"{static_baselines['train_mean']:.6f}; last value "
            f"{static_baselines['last_value']:.6f}; shrinkage persistence "
            f"{static_baselines['shrinkage_persistence']:.6f}.",
            "",
            "## Effect audit",
            "",
            f"- Candidate vs multimodal GRU: {effects['candidate_vs_multimodal_gru_percent']:+.2f}% "
            f"with {candidate_gru_wins}/3 seed wins.",
            f"- Candidate vs naive fusion: {effects['candidate_vs_naive_fusion_percent']:+.2f}% "
            f"with {candidate_naive_wins}/3 seed wins.",
            f"- Aligned candidate vs semantic shuffle: {effects['aligned_vs_semantic_shuffle_percent']:+.2f}%.",
            f"- Aligned candidate vs time-only: {effects['aligned_vs_time_only_percent']:+.2f}%.",
            f"- Multimodal vs numeric selective SSM: {effects['multimodal_selective_vs_numeric_selective_percent']:+.2f}%.",
            f"- Multimodal vs numeric matched GRU: {effects['multimodal_gru_vs_numeric_gru_percent']:+.2f}%.",
            "",
            "Positive percentages mean the first named candidate has lower MSE.",
            "The aligned model is worse than both semantic controls on average,",
            "so the semantic-fusion claim is falsified even though conditions 1-3 pass.",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "No universal model-superiority or statistical-significance claim is warranted.",
        ]
    )
    markdown_output = args.output.resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
