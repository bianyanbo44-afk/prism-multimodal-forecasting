"""Entity-aware hierarchical moving-block bootstrap for shortcut effects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eventmamba.batching import index_packed_grouped  # noqa: E402
from eventmamba.forecasting import ResidualEventForecast  # noqa: E402
from eventmamba.model import EventMambaConfig  # noqa: E402
from eventmamba.multimodal import (  # noqa: E402
    shuffle_text_values_within_entities,
    zero_text_values,
)
from eventmamba.selective import SelectiveEventMamba  # noqa: E402
from eventmamba.timeimm import (  # noqa: E402
    DATASET_SPECS,
    GPT2_TEXT_CACHE,
    GPT2_TEXT_CACHE_1024,
    load_timeimm,
)
from eventmamba.timeimm_training import prepare_timeimm_split  # noqa: E402


SEEDS = (53, 67, 79)
DATASETS = {
    "ClusterTrace": {
        "numeric": "clustertrace_numeric_fresh3_cuda",
        "multimodal": "clustertrace_multimodal_ln_fresh3_cuda",
        "semantic": "clustertrace_semantic_controls_fresh3_cuda",
    },
    "CESNET": {
        "numeric": "cesnet_numeric_fresh3_cuda",
        "multimodal": "cesnet_multimodal_ln_fresh3_cuda",
        "semantic": "cesnet_semantic_controls_fresh3_cuda",
    },
    "EPA-Air": {
        "numeric": "epa_air_numeric_fresh3_cuda",
        "multimodal": "epa_air_multimodal_ln_fresh3_cuda",
        "semantic": "epa_air_semantic_controls_fresh3_cuda",
    },
    "FNSPID": {
        "numeric": "fnspid_untouched_gate/numeric_models",
        "multimodal": "fnspid_untouched_gate/multimodal_models",
        "semantic": "fnspid_untouched_gate/semantic_controls",
    },
    "RepoHealth": {
        "numeric": "repohealth_generic_numeric_fresh3_cuda",
        "multimodal": "repohealth_multimodal_ln_fresh3_cuda",
        "semantic": "repohealth_semantic_controls_fresh3_cuda",
    },
}
COMPARISONS = {
    "semantic_alignment_gain": ("aligned", "shuffle"),
    "content_beyond_timing_gain": ("aligned", "time_only"),
    "multimodal_utility": ("aligned", "numeric"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--sensitivity-replicates", type=int, default=2_000)
    parser.add_argument("--block-length", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=99_017)
    parser.add_argument(
        "--output", type=Path, default=Path("results/block_bootstrap_report.md")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/block_bootstrap_report.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/block_bootstrap_window_errors"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_selective(
    input_dims: tuple[int, ...],
    target_dim: int,
    model_config: dict[str, object],
    numeric_modalities: tuple[int, ...],
    device: torch.device,
) -> ResidualEventForecast:
    config = EventMambaConfig(
        input_dims=input_dims,
        event_width=int(model_config["event_width"]),
        state_dim=int(model_config["state_dim"]),
        readout_width=int(model_config["readout_width"]),
        target_dim=target_dim,
        use_reliability=False,
    )
    return ResidualEventForecast(
        SelectiveEventMamba(config),
        input_dims,
        persistence_modalities=numeric_modalities,
    ).to(device)


@torch.no_grad()
def window_error_arrays(
    model: torch.nn.Module,
    prepared: object,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    window_count = len(prepared.windows)
    feature_count = prepared.targets.shape[2]
    squared = np.zeros((window_count, feature_count), dtype=np.float64)
    counts = np.zeros_like(squared)
    for start in range(0, window_count, batch_size):
        stop = min(start + batch_size, window_count)
        indices = torch.arange(
            start, stop, dtype=torch.long, device=prepared.targets.device
        )
        packed = index_packed_grouped(prepared.packed, indices)
        predictions = model.forward_packed_queries(
            packed, prepared.query_times.index_select(0, indices)
        )
        targets = prepared.targets.index_select(0, indices)
        masks = prepared.target_masks.index_select(0, indices)
        batch_squared = ((predictions - targets).square() * masks).sum(dim=1)
        batch_counts = masks.sum(dim=1)
        squared[start:stop] = batch_squared.double().cpu().numpy()
        counts[start:stop] = batch_counts.double().cpu().numpy()
    return squared, counts


def aggregate_metric(squared: np.ndarray, counts: np.ndarray) -> float:
    feature_sums = squared.sum(axis=0)
    feature_counts = counts.sum(axis=0)
    available = feature_counts > 0
    if not np.any(available):
        raise ValueError("Bootstrap sample has no observed targets")
    return float(np.mean(feature_sums[available] / feature_counts[available]))


def expected_test_mse(payload: dict[str, object], model: str, seed: int) -> float:
    matches = [
        item
        for item in payload["results"]
        if item["model"] == model and item.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one result for {model} seed {seed}")
    return float(matches[0]["test_metrics"]["mse"])


def load_state(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])


def moving_block_sample(
    ordered_indices: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    size = len(ordered_indices)
    if size <= 1:
        return ordered_indices.copy()
    length = min(block_length, size)
    possible_starts = size - length + 1
    block_count = math.ceil(size / length)
    starts = rng.integers(0, possible_starts, size=block_count)
    sampled = np.concatenate(
        [ordered_indices[start : start + length] for start in starts]
    )
    return sampled[:size]


def bootstrap_gain(
    candidate: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
    counts: np.ndarray,
    entity_indices: list[np.ndarray],
    *,
    block_length: int,
    replicates: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    output = np.empty(replicates, dtype=np.float64)
    seed_array = np.asarray(SEEDS, dtype=np.int64)
    for replicate in range(replicates):
        candidate_sum = np.zeros(counts.shape[1], dtype=np.float64)
        reference_sum = np.zeros_like(candidate_sum)
        count_sum = np.zeros_like(candidate_sum)
        sampled_seeds = rng.choice(seed_array, size=len(SEEDS), replace=True)
        for seed in sampled_seeds:
            for indices in entity_indices:
                selected = moving_block_sample(indices, block_length, rng)
                candidate_sum += candidate[int(seed)][selected].sum(axis=0)
                reference_sum += reference[int(seed)][selected].sum(axis=0)
                count_sum += counts[selected].sum(axis=0)
        available = count_sum > 0
        candidate_mse = np.mean(candidate_sum[available] / count_sum[available])
        reference_mse = np.mean(reference_sum[available] / count_sum[available])
        output[replicate] = 100.0 * (reference_mse - candidate_mse) / reference_mse
    return output


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "bootstrap_mean_percent": float(values.mean()),
        "ci95_lower_percent": float(lower),
        "ci95_upper_percent": float(upper),
        "positive_fraction": float(np.mean(values > 0)),
    }


def main() -> None:
    args = parse_args()
    if args.replicates < 100 or args.sensitivity_replicates < 100:
        raise ValueError("Bootstrap replicate counts are too small")
    if args.block_length < 1:
        raise ValueError("block length must be positive")
    root = args.root.resolve()
    results_root = root / "results"
    config = load_json(root / "configs" / "timeimm_multiseed_gate.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    reconstruction_checks: list[dict[str, object]] = []

    selected_datasets = [
        (dataset, DATASETS[dataset]) for dataset in args.datasets
    ]
    for dataset_index, (dataset, paths) in enumerate(selected_datasets):
        spec = DATASET_SPECS[dataset]
        cache_name = GPT2_TEXT_CACHE_1024 if dataset == "RepoHealth" else GPT2_TEXT_CACHE
        data = load_timeimm(
            root / "external" / "Time-IMM" / "data" / dataset,
            spec,
            normalization="train_only",
            text_cache_name=cache_name,
            text_preprocessing="layer_norm",
        )
        numeric_data = load_timeimm(
            root / "external" / "Time-IMM" / "data" / dataset,
            spec,
            normalization="train_only",
        )
        if data.text_modality is None:
            raise AssertionError("Text modality missing")
        aligned_windows = data.test_windows
        shuffled_windows = shuffle_text_values_within_entities(
            data.test_windows,
            data.window_info,
            int(data.text_modality),
            seed=int(config["semantic_control_seed"]) + 2,
        )
        time_only_windows = zero_text_values(
            data.test_windows, int(data.text_modality)
        )
        sample_ids = np.asarray(
            [window.sample_id for window in aligned_windows], dtype=np.int64
        )
        numeric_ids = np.asarray(
            [window.sample_id for window in numeric_data.test_windows], dtype=np.int64
        )
        if not np.array_equal(sample_ids, numeric_ids):
            raise ValueError(f"{dataset}: numeric and multimodal sample IDs differ")

        numeric_modalities = tuple(range(len(data.feature_names)))
        multimodal_probe = make_selective(
            data.input_dims,
            len(data.feature_names),
            config["model"],
            numeric_modalities,
            device,
        )
        prepared = {
            "aligned": prepare_timeimm_split(
                multimodal_probe,
                aligned_windows,
                data.input_dims,
                len(data.feature_names),
            ),
            "shuffle": prepare_timeimm_split(
                multimodal_probe,
                shuffled_windows,
                data.input_dims,
                len(data.feature_names),
            ),
            "time_only": prepare_timeimm_split(
                multimodal_probe,
                time_only_windows,
                data.input_dims,
                len(data.feature_names),
            ),
        }
        del multimodal_probe
        numeric_probe = make_selective(
            numeric_data.input_dims,
            len(numeric_data.feature_names),
            config["model"],
            tuple(range(len(numeric_data.feature_names))),
            device,
        )
        prepared["numeric"] = prepare_timeimm_split(
            numeric_probe,
            numeric_data.test_windows,
            numeric_data.input_dims,
            len(numeric_data.feature_names),
        )
        del numeric_probe

        numeric_results = load_json(
            results_root / str(paths["numeric"]) / "results.json"
        )
        multimodal_results = load_json(
            results_root / str(paths["multimodal"]) / "results.json"
        )
        semantic_results = load_json(
            results_root / str(paths["semantic"]) / "results.json"
        )
        result_payloads = {
            "aligned": (
                multimodal_results,
                "selective_eventmamba_no_reliability",
                results_root / str(paths["multimodal"]),
            ),
            "shuffle": (
                semantic_results,
                "selective_ssm_semantic_shuffle",
                results_root / str(paths["semantic"]),
            ),
            "time_only": (
                semantic_results,
                "selective_ssm_time_only_zero_text",
                results_root / str(paths["semantic"]),
            ),
            "numeric": (
                numeric_results,
                "selective_eventmamba_no_reliability",
                results_root / str(paths["numeric"]),
            ),
        }
        squared: dict[str, dict[int, np.ndarray]] = {
            name: {} for name in result_payloads
        }
        common_counts: np.ndarray | None = None
        cache_payload: dict[str, np.ndarray] = {
            "sample_ids": sample_ids,
        }
        for variant, (result_payload, model_name, checkpoint_root) in result_payloads.items():
            variant_data = numeric_data if variant == "numeric" else data
            variant_numeric = tuple(range(len(variant_data.feature_names)))
            for seed in SEEDS:
                model = make_selective(
                    variant_data.input_dims,
                    len(variant_data.feature_names),
                    config["model"],
                    variant_numeric,
                    device,
                )
                checkpoint = checkpoint_root / f"{model_name}_seed{seed}.pt"
                load_state(model, checkpoint, device)
                error, counts = window_error_arrays(
                    model,
                    prepared[variant],
                    batch_size=int(config["training"]["batch_size"]),
                )
                reconstructed = aggregate_metric(error, counts)
                expected = expected_test_mse(result_payload, model_name, seed)
                absolute_error = abs(reconstructed - expected)
                if absolute_error > 2e-6:
                    raise ValueError(
                        f"{dataset} {variant} seed {seed}: reconstructed "
                        f"{reconstructed} != saved {expected}"
                    )
                reconstruction_checks.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "reconstructed_mse": reconstructed,
                        "saved_mse": expected,
                        "absolute_error": absolute_error,
                    }
                )
                if common_counts is None:
                    common_counts = counts
                elif not np.array_equal(common_counts, counts):
                    raise ValueError(f"{dataset}: target counts differ for {variant}")
                squared[variant][seed] = error
                cache_payload[f"{variant}_seed{seed}_squared_sum"] = error
                del model
        if common_counts is None:
            raise AssertionError("No errors reconstructed")
        cache_payload["target_counts"] = common_counts

        entities = np.asarray(
            [data.window_info[int(sample_id)].entity for sample_id in sample_ids]
        )
        window_indices = np.asarray(
            [data.window_info[int(sample_id)].window_index for sample_id in sample_ids],
            dtype=np.int64,
        )
        cache_payload["entities"] = entities.astype("U")
        cache_payload["window_indices"] = window_indices
        np.savez_compressed(
            cache_dir / f"{dataset.lower().replace('-', '_')}_window_errors.npz",
            **cache_payload,
        )
        entity_indices = []
        for entity in sorted(set(entities.tolist())):
            indices = np.flatnonzero(entities == entity)
            indices = indices[np.argsort(window_indices[indices], kind="stable")]
            entity_indices.append(indices)

        comparisons: dict[str, object] = {}
        for comparison_index, (name, (candidate_name, reference_name)) in enumerate(
            COMPARISONS.items()
        ):
            candidate_point = statistics_mean_metric(
                squared[candidate_name], common_counts
            )
            reference_point = statistics_mean_metric(
                squared[reference_name], common_counts
            )
            point_gain = 100.0 * (
                reference_point - candidate_point
            ) / reference_point
            primary_seed = (
                args.random_seed + 10_000 * dataset_index + 100 * comparison_index
            )
            distribution = bootstrap_gain(
                squared[candidate_name],
                squared[reference_name],
                common_counts,
                entity_indices,
                block_length=args.block_length,
                replicates=args.replicates,
                random_seed=primary_seed,
            )
            sensitivity = {}
            for block_length in (1, 5):
                sensitivity_distribution = bootstrap_gain(
                    squared[candidate_name],
                    squared[reference_name],
                    common_counts,
                    entity_indices,
                    block_length=block_length,
                    replicates=args.sensitivity_replicates,
                    random_seed=primary_seed + block_length,
                )
                sensitivity[str(block_length)] = summarize_distribution(
                    sensitivity_distribution
                )
            comparisons[name] = {
                "candidate": candidate_name,
                "reference": reference_name,
                "candidate_point_mse": candidate_point,
                "reference_point_mse": reference_point,
                "point_gain_percent": point_gain,
                "block_length": args.block_length,
                "replicates": args.replicates,
                **summarize_distribution(distribution),
                "sensitivity": sensitivity,
            }
        report_rows.append(
            {
                "dataset": dataset,
                "entities": len(entity_indices),
                "test_windows": len(sample_ids),
                "entity_test_window_counts": [len(indices) for indices in entity_indices],
                "comparisons": comparisons,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {"dataset": dataset, "status": "completed", "comparisons": comparisons},
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "analysis": "exploratory entity-aware hierarchical moving-block bootstrap",
        "plan": str(results_root / "block_bootstrap_analysis_plan.md"),
        "settings": {
            "seeds": list(SEEDS),
            "outer_resampling": "training seeds with replacement",
            "inner_resampling": "moving contiguous test-window blocks within entity",
            "primary_block_length": args.block_length,
            "primary_replicates": args.replicates,
            "sensitivity_block_lengths": [1, 5],
            "sensitivity_replicates": args.sensitivity_replicates,
            "random_seed": args.random_seed,
            "interval": "2.5th and 97.5th percentiles",
        },
        "interpretation_boundary": (
            "Positive-fraction is a bootstrap stability descriptor, not a p-value. "
            "No significance claim is made from three training seeds."
        ),
        "reconstruction_checks": reconstruction_checks,
        "rows": report_rows,
    }
    json_output = (root / args.json_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Entity-aware temporal block bootstrap",
        "",
        "Exploratory hierarchical bootstrap: training seeds are resampled as the",
        "outer level, and contiguous test-window blocks are resampled within each",
        "entity. Query rows are never treated as independent. Positive gain favors",
        "aligned multimodal forecasting.",
        "",
        "| Dataset | Comparison | Point gain | 95% bootstrap interval | Positive fraction | L=1 interval | L=5 interval |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "semantic_alignment_gain": "Aligned vs semantic shuffle",
        "content_beyond_timing_gain": "Aligned vs time-only",
        "multimodal_utility": "Aligned vs numeric-only",
    }
    for row in report_rows:
        for name, comparison in row["comparisons"].items():
            sensitivity = comparison["sensitivity"]
            lines.append(
                f"| {row['dataset']} | {labels[name]} | "
                f"{comparison['point_gain_percent']:+.3f}% | "
                f"[{comparison['ci95_lower_percent']:+.3f}%, {comparison['ci95_upper_percent']:+.3f}%] | "
                f"{comparison['positive_fraction']:.3f} | "
                f"[{sensitivity['1']['ci95_lower_percent']:+.3f}%, {sensitivity['1']['ci95_upper_percent']:+.3f}%] | "
                f"[{sensitivity['5']['ci95_lower_percent']:+.3f}%, {sensitivity['5']['ci95_upper_percent']:+.3f}%] |"
            )
    lines.extend(
        [
            "",
            "Intervals are percentile bootstrap intervals, not multiplicity-adjusted",
            "significance tests. Block-length sensitivity is reported for every cell.",
            "FNSPID's preregistered PASS/FAIL decision remains based on its frozen",
            "three-seed rule, not this post-gate analysis.",
        ]
    )
    markdown_output = (root / args.output).resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def statistics_mean_metric(
    squared_by_seed: dict[int, np.ndarray], counts: np.ndarray
) -> float:
    return float(
        np.mean(
            [aggregate_metric(squared_by_seed[seed], counts) for seed in SEEDS]
        )
    )


if __name__ == "__main__":
    main()
