"""Reconstruct all saved predictions without refitting and audit reported losses."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(path, x):
    z = np.load(path)
    return ((x - z["scaler_mean"]) / z["scaler_scale"]) @ z["coefficient"] + z["intercept"][0]


def main():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    for name, expected in freeze["implementation_hashes"].items():
        assert digest(HERE / name) == expected, f"Changed frozen source {name}"
    with (OUT / "entity_loss_results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(int(r["world_seed"]), float(r["strength"]), r["representation"], int(r["entity_id"])): r for r in rows}
    maximum_error = 0.0
    checks = 0
    for world in cfg["data_seeds"]:
        raw = np.load(OUT / "raw" / f"world_{world}.npz")
        for strength in cfg["strengths"]:
            common = raw["test_common"]
            y = raw["test_baseline"] + strength * raw["test_oracle"].ravel() + raw["test_noise"]
            ids = raw["test_entity"]
            for representation in ["oracle", "sentence"]:
                feature = raw[f"test_{representation}"]
                x = np.column_stack([common, feature])
                shuffled_x = np.column_stack([common, raw[f"test_{representation}_shuffle"]])
                zero_x = np.column_stack([common, np.zeros_like(feature)])
                losses = {key: [] for key in ["aligned", "fixed_shuffle", "fixed_zero", "retrained_shuffle", "timing"]}
                for seed in cfg["training_seeds"]:
                    aligned_path = OUT / "models" / f"w{world}_s{strength}_t{seed}_{representation}_aligned.npz"
                    other_path = OUT / "models" / f"w{world}_s{strength}_t{seed}_{representation}_retrained_shuffle.npz"
                    timing_path = OUT / "models" / f"w{world}_s{strength}_t{seed}_timing.npz"
                    preds = {"aligned": predict(aligned_path, x), "fixed_shuffle": predict(aligned_path, shuffled_x),
                             "fixed_zero": predict(aligned_path, zero_x), "retrained_shuffle": predict(other_path, shuffled_x),
                             "timing": predict(timing_path, common)}
                    for condition, p in preds.items():
                        loss = (p - y) ** 2
                        losses[condition].append(np.array([loss[ids == entity].mean() for entity in np.unique(ids)]))
                averaged = {condition: np.mean(value, axis=0) for condition, value in losses.items()}
                for entity in np.unique(ids):
                    record = lookup[(world, strength, representation, int(entity))]
                    for condition, values in averaged.items():
                        error = abs(values[entity] - float(record[condition]))
                        maximum_error = max(maximum_error, error)
                        assert error < 1e-10, (world, strength, representation, entity, condition, error)
                        checks += 1
    identities = json.loads((OUT / "model_training_identity.json").read_text(encoding="utf-8"))
    report = {"frozen_source_hashes_match": True, "checkpoint_prediction_loss_checks": checks,
              "maximum_absolute_loss_error": maximum_error, "fit_count": len(identities),
              "convergence_warning_fits": sum(bool(r["convergence_warning"]) for r in identities),
              "iteration_range": [min(r["n_iter"] for r in identities), max(r["n_iter"] for r in identities)],
              "scope": "All fixed-model variants use the exact same saved aligned parameters; loss values are independently recomputed without fitting."}
    (OUT / "VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
