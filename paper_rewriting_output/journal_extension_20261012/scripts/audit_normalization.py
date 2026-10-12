"""Paired persistence sensitivity to train-only versus full-entity scaling.

This isolates a metric/protocol change, not a new learned-model result. Comparisons
exclude windows with an unobserved history channel because a zero-normalized
fallback would imply a different physical prediction under the two scalers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def persistence_error(window, features: int):
    latest = np.zeros(features, dtype=np.float64)
    seen = np.zeros(features, dtype=bool)
    for event in window.events:
        if event.modality < features and event.arrival_time <= window.cutoff:
            latest[event.modality] = float(event.value.item())
            seen[event.modality] = True
    target = window.targets.numpy().astype(np.float64)
    masks = window.target_masks.numpy().astype(np.float64)
    return (((latest[None, :] - target) ** 2) * masks).sum(axis=0), masks.sum(axis=0), seen


def metric(sums, counts):
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("No observed targets")
    return float((sums[valid] / counts[valid]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "phase3_prototype")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.prototype_root.resolve()
    sys.path.insert(0, str(root))
    from eventmamba.timeimm import load_timeimm, DATASET_SPECS
    torch.set_num_threads(2)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for dataset, spec in DATASET_SPECS.items():
        train = load_timeimm(root / "external/Time-IMM/data" / dataset, spec,
                             normalization="train_only")
        full = load_timeimm(root / "external/Time-IMM/data" / dataset, spec,
                            normalization="official_full_entity")
        full_windows = {w.sample_id: w for w in full.test_windows}
        if set(full_windows) != {w.sample_id for w in train.test_windows}:
            raise AssertionError("Window set differs")
        f = len(train.feature_names)
        sums = np.zeros((2, f))
        raw = np.zeros((2, f))
        counts = np.zeros(f)
        max_relative_raw = 0.
        excluded = 0
        for window in train.test_windows:
            other = full_windows[window.sample_id]
            entity = train.window_info[window.sample_id].entity
            if not torch.equal(window.target_masks, other.target_masks):
                raise AssertionError("Target availability differs")
            e1, n1, seen1 = persistence_error(window, f)
            e2, n2, seen2 = persistence_error(other, f)
            if not np.array_equal(n1, n2) or not np.array_equal(seen1, seen2):
                raise AssertionError("Persistence inputs differ")
            if not seen1.all():
                excluded += 1
                continue
            r1 = e1 * train.scales[entity].std ** 2
            r2 = e2 * full.scales[entity].std ** 2
            relative = np.max(abs(r1 - r2) / np.maximum(np.maximum(r1, r2), 1e-8))
            max_relative_raw = max(max_relative_raw, float(relative))
            # Loader uses float32: the physical squared error agrees up to rounding.
            if not np.allclose(r1, r2, rtol=2e-4, atol=1e-5):
                raise AssertionError(f"Physical persistence errors differ: {dataset}/{entity}")
            sums += np.stack((e1, e2))
            raw += np.stack((r1, r2))
            counts += n1
            for channel in range(f):
                records.append(dict(dataset=dataset, entity=entity, sample_id=window.sample_id,
                                    channel=train.feature_names[channel], count=int(n1[channel]),
                                    train_sse=e1[channel], full_sse=e2[channel],
                                    train_raw_sse=r1[channel], full_raw_sse=r2[channel]))
        m1, m2 = metric(sums[0], counts), metric(sums[1], counts)
        rows.append(dict(dataset=dataset, train_only_normalized_mse=m1,
                         full_entity_normalized_mse=m2,
                         normalized_mse_change_percent=100*(m2-m1)/m1,
                         train_raw_feature_mse=(raw[0]/counts).tolist(),
                         full_raw_feature_mse=(raw[1]/counts).tolist(),
                         max_window_relative_raw_sse_difference=max_relative_raw,
                         total_windows=len(train.test_windows), analyzed_windows=len(train.test_windows)-excluded,
                         excluded_missing_history_windows=excluded,
                         observed_values=int(counts.sum()), features=list(train.feature_names)))
        print(json.dumps(rows[-1]), flush=True)
    payload = dict(status="complete", analysis="exploratory metric normalization sensitivity",
                   predictor="identical physical last-observation persistence",
                   window_rule="all channels have an observed history value; paired window IDs",
                   aggregation="micro-average observed values within channel, then equal available-channel average",
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    (out/"normalization_sensitivity.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (out/"normalization_windows.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
