"""Summarize paired multi-seed multimodal confirmation runs.

The report deliberately keeps the seed-level values visible.  With only three
seeds, it reports descriptive sample standard deviations and never labels a
difference statistically significant.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


REFERENCE = "set_decay_gru_matched"
CANDIDATE = "selective_eventmamba_no_reliability"
RUNS = {
    "ClusterTrace": "results/clustertrace_multimodal_ln_fresh3_cuda/results.json",
    "RepoHealth": "results/repohealth_multimodal_ln_fresh3_cuda/results.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("results/multiseed_gate_summary.md")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/multiseed_gate_summary.json"),
    )
    return parser.parse_args()


def load_seed_mse(path: Path) -> dict[str, dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[int, float]] = {REFERENCE: {}, CANDIDATE: {}}
    for item in payload["results"]:
        model = item["model"]
        seed = item.get("seed")
        if model in result and seed is not None:
            result[model][int(seed)] = float(item["test_metrics"]["mse"])
    if set(result[REFERENCE]) != set(result[CANDIDATE]):
        raise ValueError(f"Unpaired seeds in {path}")
    if len(result[REFERENCE]) < 2:
        raise ValueError(f"Need at least two paired seeds in {path}")
    return result


def summarize_dataset(dataset: str, path: Path) -> dict[str, object]:
    values = load_seed_mse(path)
    seeds = sorted(values[REFERENCE])
    reference = [values[REFERENCE][seed] for seed in seeds]
    candidate = [values[CANDIDATE][seed] for seed in seeds]
    paired_gain = [
        100.0 * (ref - cand) / ref for ref, cand in zip(reference, candidate)
    ]
    wins = sum(cand < ref for ref, cand in zip(reference, candidate))
    reference_mean = statistics.mean(reference)
    candidate_mean = statistics.mean(candidate)
    aggregate_gain = 100.0 * (reference_mean - candidate_mean) / reference_mean
    mechanical_pass = candidate_mean < reference_mean and wins >= 2
    return {
        "dataset": dataset,
        "source": str(path),
        "seeds": seeds,
        "reference_test_mse": reference,
        "candidate_test_mse": candidate,
        "paired_candidate_gain_percent": paired_gain,
        "reference_mean_mse": reference_mean,
        "reference_sample_sd": statistics.stdev(reference),
        "candidate_mean_mse": candidate_mean,
        "candidate_sample_sd": statistics.stdev(candidate),
        "candidate_aggregate_gain_percent": aggregate_gain,
        "candidate_seed_wins": wins,
        "seed_count": len(seeds),
        "mechanical_gate": "PASS" if mechanical_pass else "FAIL",
        "gate_rule": "candidate mean MSE lower and candidate wins at least 2/3 seeds",
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = [
        summarize_dataset(dataset, (root / relative_path).resolve())
        for dataset, relative_path in RUNS.items()
    ]
    passes = sum(row["mechanical_gate"] == "PASS" for row in rows)
    payload = {
        "analysis": "paired three-seed descriptive confirmation",
        "reference_model": REFERENCE,
        "candidate_model": CANDIDATE,
        "statistical_caveat": (
            "n=3 seeds; sample SD and seed-level outcomes are descriptive only. "
            "No significance claim is made."
        ),
        "overall_status": "MIXED" if 0 < passes < len(rows) else (
            "PASS" if passes == len(rows) else "FAIL"
        ),
        "dataset_gates_passed": f"{passes}/{len(rows)}",
        "decision": (
            "Do not claim a generally superior fusion architecture. RepoHealth "
            "passes the minimal descriptive rule but has sign-changing, high-variance "
            "paired effects; ClusterTrace fails. Treat the current model as an audited "
            "candidate and pivot the primary contribution toward shortcut-aware "
            "multimodal evaluation before any model redesign."
        ),
        "rows": rows,
    }

    json_path = (root / args.json_output).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Paired multi-seed gate summary",
        "",
        "All values are test MSE from frozen three-seed runs. Lower is better. ",
        "Sample SD is descriptive; n=3 is insufficient for a significance claim.",
        "",
        "| Dataset | Matched GRU mean ± SD | Selective mean ± SD | Aggregate gain | Seed wins | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {reference_mean_mse:.6f} ± {reference_sample_sd:.6f} | "
            "{candidate_mean_mse:.6f} ± {candidate_sample_sd:.6f} | "
            "{candidate_aggregate_gain_percent:+.2f}% | "
            "{candidate_seed_wins}/{seed_count} | {mechanical_gate} |".format(**row)
        )

    for row in rows:
        lines.extend(["", f"## {row['dataset']}", ""])
        lines.extend(
            [
                "| Seed | Matched GRU | Selective | Selective gain | Winner |",
                "|---:|---:|---:|---:|:---|",
            ]
        )
        for seed, ref, cand, gain in zip(
            row["seeds"],
            row["reference_test_mse"],
            row["candidate_test_mse"],
            row["paired_candidate_gain_percent"],
        ):
            winner = "Selective" if cand < ref else "GRU"
            lines.append(
                f"| {seed} | {ref:.6f} | {cand:.6f} | {gain:+.2f}% | {winner} |"
            )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            f"Overall status: **{payload['overall_status']}** "
            f"({payload['dataset_gates_passed']} dataset gates passed).",
        ]
    )
    markdown_path = (root / args.output).resolve()
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
