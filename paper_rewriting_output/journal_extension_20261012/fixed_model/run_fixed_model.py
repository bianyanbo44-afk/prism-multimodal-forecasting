"""Exploratory, checkpoint-frozen test-input dependence evaluation for SemAudit.

No optimizer, fitting, threshold selection, or checkpoint writing is performed.
The historical Time-IMM test sets have already been inspected. These intervals
are descriptive sensitivity analyses, not untouched confirmation or equivalence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "phase3_prototype"
sys.path.insert(0, str(ROOT))
from eventmamba.batching import index_packed_grouped
from eventmamba.forecasting import ResidualEventForecast
from eventmamba.immtsf_baseline import IMMTSFRecAvgGRAddBaseline, IMMTSFT2VXAttnBaseline
from eventmamba.model import EventMambaConfig
from eventmamba.multimodal import shuffle_text_values_within_entities, zero_text_values
from eventmamba.selective import SelectiveEventMamba
from eventmamba.timeimm import DATASET_SPECS, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024, load_timeimm
from eventmamba.timeimm_training import prepare_timeimm_split

SLUGS = {"ClusterTrace": "clustertrace", "CESNET": "cesnet", "EPA-Air": "epa_air",
         "FNSPID": "fnspid", "RepoHealth": "repohealth"}
CONTROL_SEED = 9103  # Released split-specific test seed: 9101 + 2.
DOSES = (0.25, 0.5, 0.75, 1.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def groups_for(windows, info, text_modality):
    groups = defaultdict(list)
    for wi, window in enumerate(windows):
        for ei, event in enumerate(window.events):
            if event.modality == text_modality:
                groups[info[window.sample_id].entity].append((wi, ei))
    return dict(groups)


def derangement(n, rng):
    if n < 2:
        return list(range(n))
    order = list(range(n))
    for _ in range(10000):
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            return order
    raise RuntimeError("Derangement rejection sampling failed")


def dose_source_maps(groups, fraction, seed):
    """Nested selected subsets; shuffle only within each selected entity subset.

    A one-position subset is left unchanged. This avoids claiming an impossible
    one-note permutation and reports actual rather than nominal coverage.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Dose fraction outside [0, 1]")
    maps = {}
    for entity in sorted(groups):
        # Stable entity-specific streams make selected subsets nested over doses.
        salt = int(hashlib.sha256(entity.encode()).hexdigest()[:16], 16)
        chosen = list(range(len(groups[entity])))
        random.Random(seed + salt).shuffle(chosen)
        k = math.floor(fraction * len(chosen))
        if k < 2:
            k = 0
        selected = chosen[:k]
        permutation = derangement(k, random.Random(seed + salt + 1))
        maps[entity] = list(range(len(chosen)))
        for i, source in enumerate(permutation):
            maps[entity][selected[i]] = selected[source]
    return maps


def replace_values(windows, replacements):
    return tuple(replace(w, events=tuple(replace(e, value=replacements.get((wi, ei), e.value))
                                        for ei, e in enumerate(w.events)))
                 for wi, w in enumerate(windows))


def verify_invariants(original, altered, text_modality):
    if len(original) != len(altered):
        raise AssertionError("Window count changed")
    for a, b in zip(original, altered, strict=True):
        if (a.sample_id, a.cutoff) != (b.sample_id, b.cutoff):
            raise AssertionError("Forecast unit changed")
        for attr in ("query_horizons", "targets", "target_masks"):
            if not torch.equal(getattr(a, attr), getattr(b, attr)):
                raise AssertionError(f"{attr} changed")
        if len(a.events) != len(b.events):
            raise AssertionError("Event count changed")
        for x, y in zip(a.events, b.events, strict=True):
            if (x.event_time, x.arrival_time, x.modality, x.quality, x.event_id) != (
                    y.event_time, y.arrival_time, y.modality, y.quality, y.event_id):
                raise AssertionError("Event process changed")
            if x.modality != text_modality and not torch.equal(x.value, y.value):
                raise AssertionError("Numerical input changed")
            if not torch.isfinite(y.value).all():
                raise AssertionError("Nonfinite altered value")


def train_centroid(windows, text_modality):
    """Mean of train retained note positions only; no validation or test values."""
    total, n = None, 0
    for w in windows:
        for e in w.events:
            if e.modality == text_modality:
                total = e.value.double().clone() if total is None else total + e.value.double()
                n += 1
    if n == 0:
        raise ValueError("No train text for neutral centroid")
    return (total / n).float(), n


def make_variants(data):
    original = data.test_windows
    tm = int(data.text_modality)
    groups = groups_for(original, data.window_info, tm)
    centroid, train_n = train_centroid(data.train_windows, tm)
    variants = {"aligned": original,
                "released_shuffle": shuffle_text_values_within_entities(
                    original, data.window_info, tm, seed=CONTROL_SEED)}
    maps = {}
    rng = random.Random(CONTROL_SEED)
    released = {}
    for entity in sorted(groups):
        p = list(range(len(groups[entity])))
        rng.shuffle(p)
        released[entity] = p
        for location, source in zip(groups[entity], p, strict=True):
            source_location = groups[entity][source]
            if not torch.equal(variants["released_shuffle"][location[0]].events[location[1]].value,
                               original[source_location[0]].events[source_location[1]].value):
                raise AssertionError("Released shuffle replay differs")
    maps["released_shuffle"] = released
    for fraction in DOSES:
        label = f"derangement_{round(100 * fraction)}"
        source_maps = dose_source_maps(groups, fraction, CONTROL_SEED)
        replacements = {}
        for entity, p in source_maps.items():
            for location, source in zip(groups[entity], p, strict=True):
                source_location = groups[entity][source]
                replacements[location] = original[source_location[0]].events[source_location[1]].value
        variants[label] = replace_values(original, replacements)
        maps[label] = source_maps
    variants["zero_content"] = zero_text_values(original, tm)
    variants["train_centroid"] = replace_values(
        original, {location: centroid for entries in groups.values() for location in entries})
    diagnostics = {}
    n = sum(len(g) for g in groups.values())
    for label, altered in variants.items():
        verify_invariants(original, altered, tm)
        identical = sum(torch.equal(original[wi].events[ei].value, altered[wi].events[ei].value)
                        for entries in groups.values() for wi, ei in entries)
        source_changed = (sum(i != p for values in maps[label].values()
                              for i, p in enumerate(values)) if label in maps else
                          0 if label == "aligned" else None)
        diagnostics[label] = {"note_positions": n, "identical_value_positions": identical,
                              "achieved_value_change_fraction": 1 - identical / n if n else None,
                              "changed_source_positions": source_changed,
                              "source_change_fraction": source_changed / n if source_changed is not None and n else None,
                              "singleton_entities": sum(len(g) == 1 for g in groups.values()),
                              "singleton_positions": sum(len(g) for g in groups.values() if len(g) == 1),
                              "entities_with_text": len(groups), "event_target_invariants": True,
                              "per_entity_multiset_preserved": label in maps or label == "aligned"}
    return variants, diagnostics, {"retained_train_note_positions": train_n,
                                  "vector_sha256": hashlib.sha256(centroid.numpy().tobytes()).hexdigest(),
                                  "norm": float(centroid.norm()), "fitted_split": "train only",
                                  "meaning": "position-weighted representation mean; not natural neutral text"}, maps


def sources_for(dataset, architecture):
    slug = SLUGS[dataset]
    if architecture == "selective_ssm":
        folder = (ROOT / "results/fnspid_untouched_gate/multimodal_models" if dataset == "FNSPID"
                  else ROOT / "results" / f"{slug}_multimodal_ln_fresh3_cuda")
        name = "selective_eventmamba_no_reliability"
    elif architecture == "immtsf_recavg_gradd":
        folder = ROOT / "results" / f"{slug}_immtsf_fusion_audit_fresh3"
        name = "immtsf_recavg_gradd_aligned_unit_consistent"
    else:
        folder = ROOT / "results" / f"{slug}_immtsf_xattn_audit_fresh3"
        name = "immtsf_t2v_xattn_aligned"
    return folder, name


def build_model(architecture, data, config, device):
    numeric = tuple(range(len(data.feature_names)))
    kwargs = dict(input_dims=data.input_dims, numeric_modalities=numeric,
                  text_modality=int(data.text_modality), target_dim=len(data.feature_names),
                  total_window=data.spec.total)
    if architecture == "immtsf_recavg_gradd":
        return IMMTSFRecAvgGRAddBaseline(**kwargs, time_mode="unit_consistent",
                                       d_txt=768, recency_sigma=1.0, dropout=0.1).to(device)
    if architecture == "immtsf_t2v_xattn":
        return IMMTSFT2VXAttnBaseline(**kwargs, d_txt=128, n_heads=1, dropout=0.1,
                                    kappa=1.0).to(device)
    mc = config["model"]
    base = SelectiveEventMamba(EventMambaConfig(
        input_dims=data.input_dims, event_width=int(mc["event_width"]),
        state_dim=int(mc["state_dim"]), readout_width=int(mc["readout_width"]),
        target_dim=len(data.feature_names), use_reliability=False))
    return ResidualEventForecast(base, data.input_dims, persistence_modalities=numeric).to(device)


@torch.inference_mode()
def evaluate(model, split, data, batch_size):
    model.eval()
    f = len(data.feature_names)
    squared = torch.zeros(f, dtype=torch.float64)
    absolute = torch.zeros(f, dtype=torch.float64)
    counts = torch.zeros(f, dtype=torch.float64)
    arrays = {key: np.zeros((len(split.windows), f), dtype=np.float64)
              for key in ("squared", "absolute", "counts")}
    for start in range(0, len(split.windows), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split.windows)),
                               dtype=torch.long, device=split.query_times.device)
        pred = model.forward_packed_queries(index_packed_grouped(split.packed, indices),
                                            split.query_times.index_select(0, indices))
        target = split.targets.index_select(0, indices)
        mask = split.target_masks.index_select(0, indices)
        if pred.shape != target.shape or not torch.isfinite(pred).all():
            raise AssertionError("Invalid prediction")
        error = pred - target
        s, a = error.square() * mask, error.abs() * mask
        squared += s.sum((0, 1)).double().cpu()
        absolute += a.sum((0, 1)).double().cpu()
        counts += mask.sum((0, 1)).double().cpu()
        arrays["squared"][start:start + len(indices)] = s.sum(1).double().cpu().numpy()
        arrays["absolute"][start:start + len(indices)] = a.sum(1).double().cpu().numpy()
        arrays["counts"][start:start + len(indices)] = mask.sum(1).double().cpu().numpy()
    valid = counts > 0
    metrics = {"mse": float((squared / counts.clamp_min(1))[valid].mean()),
               "mae": float((absolute / counts.clamp_min(1))[valid].mean()),
               "feature_mse": (squared / counts.clamp_min(1)).tolist(),
               "feature_mae": (absolute / counts.clamp_min(1)).tolist(),
               "windows": len(split.windows), "observed_values": int(counts.sum())}
    arrays.update(sample_ids=np.array([w.sample_id for w in split.windows]),
                  entity=np.array([data.window_info[w.sample_id].entity for w in split.windows]),
                  start_time=np.array([data.window_info[w.sample_id].start_time for w in split.windows]))
    metrics["entity_mse"] = entity_statistics(arrays)
    return metrics, arrays


def entity_statistics(arrays):
    result = {}
    for entity in sorted(set(arrays["entity"])):
        ix = np.where(arrays["entity"] == entity)[0]
        c = arrays["counts"][ix].sum(0)
        valid = c > 0
        result[str(entity)] = float(np.mean(arrays["squared"][ix].sum(0)[valid] / c[valid]))
    return result


def macro_loss(squared, counts):
    c = counts.sum(0)
    valid = c > 0
    return float(np.mean(squared.sum(0)[valid] / c[valid]))


def paired_bootstrap(aligned, altered, *, repetitions=2000, block_length=2, seed=194):
    """Two-stage paired entity/circular-window-block bootstrap, conditional on seeds.

    Arrays average squared error over matching training seeds BEFORE sampling;
    seeds are not treated as new entities. Features retain the historical
    macro-feature normalized-MSE weighting in every bootstrap replicate.
    """
    if not np.array_equal(aligned["sample_ids"], altered["sample_ids"]) or not np.array_equal(aligned["counts"], altered["counts"]):
        raise AssertionError("Unpaired bootstrap input")
    if block_length < 1:
        raise ValueError("Block length must be positive")
    groups = [np.where(aligned["entity"] == e)[0] for e in sorted(set(aligned["entity"]))]
    rng = np.random.default_rng(seed)
    delta = altered["squared"] - aligned["squared"]
    point = macro_loss(delta, aligned["counts"])
    values = np.empty(repetitions)
    for b in range(repetitions):
        sampled = []
        for entity_index in rng.integers(0, len(groups), size=len(groups)):
            group = groups[entity_index]
            group = group[np.argsort(aligned["start_time"][group], kind="stable")]
            n = len(group)
            length = min(block_length, n)
            starts = rng.integers(0, n, size=math.ceil(n / length))
            offsets = (starts[:, None] + np.arange(length)[None]) % n
            sampled.extend(group[offsets.ravel()[:n]])
        ix = np.asarray(sampled, dtype=np.int64)
        values[b] = macro_loss(delta[ix], aligned["counts"][ix])
    lo, hi = np.quantile(values, (0.025, 0.975))
    return {"delta_mse": point, "ci95": [float(lo), float(hi)],
            "block_length_windows": block_length, "bootstrap_repetitions": repetitions,
            "entity_clusters": len(groups), "bootstrap_seed": seed,
            "interpretation": "exploratory percentile interval conditional on selected models and inspected test sets"}


def aggregate(output, records, repetitions):
    summaries = []
    for dataset in sorted(set(r["dataset"] for r in records)):
        for architecture in sorted(set(r["architecture"] for r in records if r["dataset"] == dataset)):
            subset = [r for r in records if r["dataset"] == dataset and r["architecture"] == architecture]
            seeds = sorted(set(r["seed"] for r in subset))
            aligned_records = sorted([r for r in subset if r["condition"] == "aligned"], key=lambda r: r["seed"])
            def averaged(condition):
                rows = sorted([r for r in subset if r["condition"] == condition], key=lambda r: r["seed"])
                if [r["seed"] for r in rows] != seeds:
                    raise AssertionError("Incomplete seed pairing")
                loaded = [dict(np.load(output / r["paired_arrays"], allow_pickle=False)) for r in rows]
                result = dict(loaded[0])
                for key in ("squared", "absolute"):
                    result[key] = np.mean([a[key] for a in loaded], axis=0)
                for a in loaded:
                    if not np.array_equal(result["sample_ids"], a["sample_ids"]) or not np.array_equal(result["counts"], a["counts"]):
                        raise AssertionError("Seed windows/counts differ")
                return result, rows
            aligned, _ = averaged("aligned")
            baseline = float(np.mean([r["metrics"]["mse"] for r in aligned_records]))
            for condition in sorted(set(r["condition"] for r in subset) - {"aligned"}):
                altered, rows = averaged(condition)
                by_seed = [r["metrics"]["mse"] - a["metrics"]["mse"]
                           for r, a in zip(rows, aligned_records, strict=True)]
                intervals = [paired_bootstrap(aligned, altered, repetitions=repetitions,
                                             block_length=length) for length in (2, 4, 8)]
                summaries.append({"dataset": dataset, "architecture": architecture,
                                  "condition": condition, "training_seeds": seeds,
                                  "aligned_mse_mean": baseline,
                                  "altered_mse_mean": float(np.mean([r["metrics"]["mse"] for r in rows])),
                                  "delta_mse_mean": float(np.mean(by_seed)),
                                  "delta_mse_sample_sd_across_training_seeds": float(np.std(by_seed, ddof=1)) if len(seeds) > 1 else None,
                                  "delta_percent_of_aligned": 100 * float(np.mean(by_seed)) / baseline,
                                  "seed_deltas": dict(zip(map(str, seeds), by_seed)),
                                  "dependence_intervals": intervals,
                                  "achieved_value_change_fraction": rows[0]["coverage"]["achieved_value_change_fraction"],
                                  "entity_deltas": {entity: entity_statistics(altered)[entity] - value
                                                    for entity, value in entity_statistics(aligned).items()}})
    write_json(output / "paired_summary.json", summaries)
    with (output / "paired_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "architecture", "condition", "aligned_mse", "altered_mse", "delta_mse", "delta_percent", "value_change_fraction", "ci95_block2_low", "ci95_block2_high", "seed_delta_sd"])
        writer.writeheader()
        for r in summaries:
            ci = r["dependence_intervals"][0]["ci95"]
            writer.writerow(dict(dataset=r["dataset"], architecture=r["architecture"], condition=r["condition"], aligned_mse=r["aligned_mse_mean"], altered_mse=r["altered_mse_mean"], delta_mse=r["delta_mse_mean"], delta_percent=r["delta_percent_of_aligned"], value_change_fraction=r["achieved_value_change_fraction"], ci95_block2_low=ci[0], ci95_block2_high=ci[1], seed_delta_sd=r["delta_mse_sample_sd_across_training_seeds"]))
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS))
    p.add_argument("--architectures", nargs="+", default=["selective_ssm", "immtsf_recavg_gradd", "immtsf_t2v_xattn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[53, 67, 79])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap-repetitions", type=int, default=2000)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()
    output = args.output.resolve()
    (output / "paired_windows").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    started = time.perf_counter()
    manifest = {"status": "running", "analysis": "exploratory fixed aligned checkpoint test-only input perturbation",
                "historical_test_sets_already_inspected": True, "training_performed": False,
                "runtime": {"python": sys.version, "platform": platform.platform(),
                            "torch": str(torch.__version__), "device": str(device),
                            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
                "settings": vars(args) | {"output": str(output)}, "test_control_seed": CONTROL_SEED,
                "source_sha256": {str(f.relative_to(ROOT)): sha(f) for f in [ROOT / "eventmamba/timeimm.py", ROOT / "eventmamba/multimodal.py", ROOT / "eventmamba/timeimm_training.py", ROOT / "eventmamba/immtsf_baseline.py", ROOT / "eventmamba/selective.py"]},
                "runner_sha256": sha(__file__), "records": [], "coverage": {}, "failures": []}
    def save():
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)
    save()
    for dataset in args.datasets:
        for preprocessing in ("layer_norm", "none"):
            relevant = [a for a in args.architectures if (a == "selective_ssm") == (preprocessing == "layer_norm")]
            if not relevant:
                continue
            cache = GPT2_TEXT_CACHE_1024 if dataset == "RepoHealth" else GPT2_TEXT_CACHE
            data = load_timeimm(ROOT / "external/Time-IMM/data" / dataset, DATASET_SPECS[dataset],
                               normalization="train_only", text_cache_name=cache,
                               text_preprocessing=preprocessing)
            variants, diagnostics, centroid, source_maps = make_variants(data)
            manifest["coverage"][f"{dataset}/{preprocessing}"] = {"conditions": diagnostics, "centroid": centroid,
                                                                    "cache_name": cache, "windows": len(data.test_windows)}
            write_json(output / f"{SLUGS[dataset]}_{preprocessing}_source_maps.json", source_maps)
            for architecture in relevant:
                folder, checkpoint_name = sources_for(dataset, architecture)
                archived = json.loads((folder / "results.json").read_text(encoding="utf-8"))
                model = build_model(architecture, data, archived["config"], device)
                # Packed inputs depend on data, not checkpoint weights; reused across seeds.
                prepared = {condition: prepare_timeimm_split(model, windows, data.input_dims, len(data.feature_names))
                            for condition, windows in variants.items()}
                for seed in args.seeds:
                    checkpoint_path = folder / f"{checkpoint_name}_seed{seed}.pt"
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    if checkpoint["seed"] != seed or checkpoint["model"] != checkpoint_name:
                        raise AssertionError("Checkpoint identity mismatch")
                    historical = next(r for r in archived["results"] if r["seed"] == seed and r["model"] == checkpoint_name)
                    if historical["test_metrics"] != checkpoint["test_metrics"]:
                        raise AssertionError("Archive and checkpoint metrics differ")
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    weight_hash = state_digest(model)
                    checkpoint_hash = sha(checkpoint_path)
                    for condition, split in prepared.items():
                        tick = time.perf_counter()
                        metrics, arrays = evaluate(model, split, data, args.batch_size)
                        if condition == "aligned":
                            errors = {metric: abs(metrics[metric] - checkpoint["test_metrics"][metric])
                                      for metric in ("mse", "mae")}
                            if any(errors[m] > 1e-5 * max(1, abs(checkpoint["test_metrics"][m])) for m in errors):
                                raise AssertionError(f"Aligned replay failure {dataset}/{architecture}/{seed}: {errors}")
                            feature_error = float(np.max(np.abs(np.array(metrics["feature_mse"]) - np.array(checkpoint["test_metrics"]["feature_mse"]))))
                            if not np.allclose(metrics["feature_mse"], checkpoint["test_metrics"]["feature_mse"], rtol=1e-5, atol=1e-6):
                                raise AssertionError("Aligned feature replay failure")
                            replay = {"verified": True, "absolute_errors": errors,
                                      "max_feature_mse_absolute_error": feature_error,
                                      "tolerance": "rtol 1e-5; atol 1e-6 for feature MSE"}
                        if state_digest(model) != weight_hash:
                            raise AssertionError("Weights or buffers mutated during evaluation")
                        path = Path("paired_windows") / f"{SLUGS[dataset]}_{architecture}_seed{seed}_{condition}.npz"
                        np.savez_compressed(output / path, **arrays)
                        record = {"dataset": dataset, "architecture": architecture, "seed": seed,
                                  "condition": condition, "preprocessing": preprocessing,
                                  "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                                  "historical_best_epoch": checkpoint["best_epoch"],
                                  "historical_val_mse": checkpoint["val_metrics"]["mse"],
                                  "state_sha256": weight_hash, "weights_unchanged": True,
                                  "aligned_replay": replay, "metrics": metrics,
                                  "coverage": diagnostics[condition], "paired_arrays": path.as_posix(),
                                  "seconds": time.perf_counter() - tick}
                        manifest["records"].append(record)
                        save()
                        print(json.dumps({"dataset": dataset, "model": architecture, "seed": seed,
                                          "condition": condition, "mse": metrics["mse"],
                                          "seconds": record["seconds"]}), flush=True)
                    if sha(checkpoint_path) != checkpoint_hash:
                        raise AssertionError("Checkpoint file mutated")
                del model, prepared
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del data, variants
    print("Generating paired dependence-aware summaries", flush=True)
    summaries = aggregate(output, manifest["records"], args.bootstrap_repetitions)
    manifest["summary_cells"] = len(summaries)
    manifest["status"] = "complete"
    save()
    print(json.dumps({"status": "complete", "records": len(manifest["records"]),
                      "summary_cells": len(summaries), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
