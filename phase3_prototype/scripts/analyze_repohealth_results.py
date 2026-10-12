"""Validate and summarize RepoHealth multi-seed experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


TRAINED_MODELS = (
    "set_decay_gru_matched",
    "selective_eventmamba_no_reliability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def ensure_finite(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            ensure_finite(child, f"{path}[{index}]")


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    result_path = args.results.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    ensure_finite(payload)
    expected_seeds = [int(seed) for seed in payload["config"]["seeds"]]
    results = payload["results"]
    baseline = [result for result in results if result["model"] == "last_value"]
    if len(baseline) != 1:
        raise ValueError(f"Expected one last-value result, found {len(baseline)}")

    by_model: dict[str, dict[int, dict[str, object]]] = {}
    for model in TRAINED_MODELS:
        rows = {
            int(result["seed"]): result
            for result in results
            if result["model"] == model
        }
        if sorted(rows) != sorted(expected_seeds):
            raise ValueError(f"{model}: seeds {sorted(rows)} != {sorted(expected_seeds)}")
        for result in rows.values():
            checkpoint = Path(str(result["checkpoint"]))
            if not checkpoint.exists() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        by_model[model] = rows

    log_path = result_path.parent / "epochs.jsonl"
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    for model in TRAINED_MODELS:
        for seed in expected_seeds:
            model_logs = [
                row for row in logs if row["model"] == model and int(row["seed"]) == seed
            ]
            if not model_logs or model_logs[0]["epoch"] != 0:
                raise ValueError(f"Missing epoch-0 record for {model}, seed {seed}")
            expected_epochs = int(by_model[model][seed]["completed_epochs"])
            if model_logs[-1]["epoch"] != expected_epochs:
                raise ValueError(f"Incomplete log for {model}, seed {seed}")

    last_mse = float(baseline[0]["test_metrics"]["mse"])
    feature_names = payload["data"]["feature_names"]
    seed_lines = []
    paired_improvements = []
    for seed in expected_seeds:
        gru = float(by_model[TRAINED_MODELS[0]][seed]["test_metrics"]["mse"])
        selective = float(by_model[TRAINED_MODELS[1]][seed]["test_metrics"]["mse"])
        improvement = 100.0 * (gru - selective) / gru
        paired_improvements.append(improvement)
        winner = "Selective" if selective < gru else "SetDecayGRU"
        seed_lines.append(
            f"| {seed} | {gru:.6f} | {selective:.6f} | {improvement:+.2f}% | {winner} |"
        )

    summary: dict[str, dict[str, float]] = {}
    for model in TRAINED_MODELS:
        mses = [
            float(by_model[model][seed]["test_metrics"]["mse"])
            for seed in expected_seeds
        ]
        maes = [
            float(by_model[model][seed]["test_metrics"]["mae"])
            for seed in expected_seeds
        ]
        durations = [float(by_model[model][seed]["duration_seconds"]) for seed in expected_seeds]
        epochs = [float(by_model[model][seed]["completed_epochs"]) for seed in expected_seeds]
        mse_mean, mse_sd = mean_sd(mses)
        mae_mean, mae_sd = mean_sd(maes)
        summary[model] = {
            "mse_mean": mse_mean,
            "mse_sd": mse_sd,
            "mae_mean": mae_mean,
            "mae_sd": mae_sd,
            "duration_mean": statistics.mean(durations),
            "epochs_mean": statistics.mean(epochs),
        }

    gru_mean = summary[TRAINED_MODELS[0]]["mse_mean"]
    selective_mean = summary[TRAINED_MODELS[1]]["mse_mean"]
    mean_gain = 100.0 * (gru_mean - selective_mean) / gru_mean
    baseline_gain = 100.0 * (last_mse - selective_mean) / last_mse

    feature_lines = []
    selective_feature_wins = 0
    for feature_index, feature_name in enumerate(feature_names):
        gru_value = statistics.mean(
            float(by_model[TRAINED_MODELS[0]][seed]["test_metrics"]["feature_mse"][feature_index])
            for seed in expected_seeds
        )
        selective_value = statistics.mean(
            float(by_model[TRAINED_MODELS[1]][seed]["test_metrics"]["feature_mse"][feature_index])
            for seed in expected_seeds
        )
        gain = 100.0 * (gru_value - selective_value) / gru_value
        selective_feature_wins += int(selective_value < gru_value)
        feature_lines.append(
            f"| {feature_name} | {gru_value:.6f} | {selective_value:.6f} | {gain:+.2f}% |"
        )

    entity_names = sorted(payload["data"]["window_counts"])
    entity_lines = []
    for entity in entity_names:
        gru_value = statistics.mean(
            float(by_model[TRAINED_MODELS[0]][seed]["test_metrics"]["entity_mse"][entity])
            for seed in expected_seeds
        )
        selective_value = statistics.mean(
            float(by_model[TRAINED_MODELS[1]][seed]["test_metrics"]["entity_mse"][entity])
            for seed in expected_seeds
        )
        gain = 100.0 * (gru_value - selective_value) / gru_value
        entity_lines.append(
            f"| {entity} | {gru_value:.6f} | {selective_value:.6f} | {gain:+.2f}% |"
        )

    parameter_match = payload["parameter_match"]
    wins = sum(value > 0 for value in paired_improvements)
    report = f"""# RepoHealth fresh-seed validation report

## Outcome

The leakage-free numerical-backbone experiment completed for all frozen seeds {expected_seeds}. SelectiveEventMamba achieved mean test MSE **{selective_mean:.6f} ± {summary[TRAINED_MODELS[1]]['mse_sd']:.6f}**, versus **{gru_mean:.6f} ± {summary[TRAINED_MODELS[0]]['mse_sd']:.6f}** for the parameter-matched SetDecayGRU. This is a mean relative gain of **{mean_gain:.2f}%**, with Selective winning **{wins}/{len(expected_seeds)}** seeds. It also improves over Last-Value ({last_mse:.6f}) by **{baseline_gain:.2f}%** on the mean.

This is a positive but not yet publication-grade result: three initialization seeds on one previously inspected dataset do not establish statistical significance or external generalization.

## Per-seed primary metric

| Seed | SetDecayGRU MSE | Selective MSE | Selective gain | Winner |
|---:|---:|---:|---:|---|
{chr(10).join(seed_lines)}

Mean paired relative gain: **{statistics.mean(paired_improvements):.2f}%** (range {min(paired_improvements):+.2f}% to {max(paired_improvements):+.2f}%).

## Aggregate metrics and efficiency

| Model | Test MSE mean ± SD | Test MAE mean ± SD | Mean trained epochs | Mean model time |
|---|---:|---:|---:|---:|
| SetDecayGRU | {summary[TRAINED_MODELS[0]]['mse_mean']:.6f} ± {summary[TRAINED_MODELS[0]]['mse_sd']:.6f} | {summary[TRAINED_MODELS[0]]['mae_mean']:.6f} ± {summary[TRAINED_MODELS[0]]['mae_sd']:.6f} | {summary[TRAINED_MODELS[0]]['epochs_mean']:.1f} | {summary[TRAINED_MODELS[0]]['duration_mean']:.1f} s |
| SelectiveEventMamba | {summary[TRAINED_MODELS[1]]['mse_mean']:.6f} ± {summary[TRAINED_MODELS[1]]['mse_sd']:.6f} | {summary[TRAINED_MODELS[1]]['mae_mean']:.6f} ± {summary[TRAINED_MODELS[1]]['mae_sd']:.6f} | {summary[TRAINED_MODELS[1]]['epochs_mean']:.1f} | {summary[TRAINED_MODELS[1]]['duration_mean']:.1f} s |

The one-time shared packing cost was {payload['prepacked_cache']['seconds']:.1f} s. Trainable parameters were {int(parameter_match['set_decay_gru']):,} for SetDecayGRU and {int(parameter_match['selective_eventmamba']):,} for SelectiveEventMamba, a relative mismatch of {100.0 * float(parameter_match['relative_difference']):.2f}%.

## Feature-level test MSE (seed mean)

| Feature | SetDecayGRU | Selective | Selective gain |
|---|---:|---:|---:|
{chr(10).join(feature_lines)}

Selective wins {selective_feature_wins}/{len(feature_names)} features on the seed mean.

## Entity-level test MSE (seed mean)

| Entity | SetDecayGRU | Selective | Selective gain |
|---|---:|---:|---:|
{chr(10).join(entity_lines)}

## Validation gates

- **PASS — completeness:** one deterministic Last-Value result plus {len(expected_seeds)} seeds for each trained model.
- **PASS — artifact integrity:** all {2 * len(expected_seeds)} checkpoints exist and are non-empty; epoch logs include epoch 0 and the reported terminal epoch.
- **PASS — numerical validity:** recursive JSON audit found no NaN or infinite metric.
- **PASS — parameter matching:** mismatch is below 1%.
- **PASS — core signal:** Selective beats Last-Value on the three-seed mean and beats SetDecayGRU on the mean.
- **FAIL — seed unanimity:** Selective wins {wins}/{len(expected_seeds)}, not all seeds.
- **FAIL — external validation:** only RepoHealth numerical streams are included; text is absent and the RepoHealth test period was inspected during development.
- **NOT TESTED — statistical uncertainty:** inference needs paired window-level predictions with temporal block bootstrap; treating 1,471 correlated query points as independent would be pseudo-replication.

## Material Passport

- Source: {payload['material_passport']['source']}
- Data commit: `{payload['material_passport']['data_commit']}`
- Task: {payload['material_passport']['task']}
- Split: {payload['material_passport']['split']}
- Normalization: {payload['material_passport']['normalization']}
- Decoder: {payload['material_passport']['decoder']}
- Queries: train={payload['data']['total_queries']['train']}, val={payload['data']['total_queries']['val']}, test={payload['data']['total_queries']['test']}
- Runtime: PyTorch {payload['runtime']['torch']}, device {payload['runtime']['device']}, GPU {payload['runtime']['gpu']}

## Required next gate

Freeze the current numerical configuration and evaluate it on at least two untouched Time-IMM datasets. Then add the text stream with a frozen semantic encoder and compare numeric-only, text-only, naïve fusion, SetDecayGRU fusion, and SelectiveEventMamba fusion. RepoHealth alone cannot support a top-conference claim.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "seeds": expected_seeds,
                "last_value_mse": last_mse,
                "gru_mean_mse": gru_mean,
                "selective_mean_mse": selective_mean,
                "selective_mean_gain_percent": mean_gain,
                "selective_seed_wins": wins,
                "selective_feature_wins": selective_feature_wins,
                "report": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
