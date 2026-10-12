"""Frozen semi-synthetic, irregular historical-text sensitivity calibration.

The empirical EPA-Air numerical/text timestamps are reused, but values, latent
directions, prose, arrival delays and targets are generated. This is not an
evaluation on the natural EPA-Air values or narratives. No GPU is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_times(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array(sorted(datetime.fromisoformat(r["date_time"].replace("Z", "+00:00"))
                           for r in rows), dtype="datetime64[s]")


def load_schedules(cfg: dict) -> list[dict]:
    result = []
    for d in sorted(Path(cfg["source_root"]).iterdir()):
        if not d.is_dir():
            continue
        nt = parse_times(d / "time_series.csv")
        tt = parse_times(d / "text.csv")
        base = nt[0]
        numeric = (nt - base).astype("timedelta64[s]").astype(float) / 86400
        text = (tt - base).astype("timedelta64[s]").astype(float) / 86400
        cutoff = np.arange(cfg["history_days"], numeric[-1] - cfg["forecast_horizon_days"],
                           cfg["stride_days"])[:cfg["max_queries_per_entity"]]
        # Every retained cutoff has a past numeric record; empty text windows remain.
        for c in cutoff:
            assert np.any((numeric > c - cfg["history_days"]) & (numeric <= c))
        result.append({"name": d.name, "numeric": numeric, "text": text, "cutoffs": cutoff,
                       "source_hashes": {p.name: sha(p) for p in (d / "time_series.csv", d / "text.csv")}})
    assert len(result) == 8
    return result


def schedule_identity(schedules: list[dict]) -> list[dict]:
    return [{"name": s["name"], "source_hashes": s["source_hashes"],
             "numeric_records": len(s["numeric"]), "text_records": len(s["text"]),
             "query_count": len(s["cutoffs"]),
             "text_interarrival_sd_days": float(np.std(np.diff(s["text"]))),
             "text_interarrival_min_days": float(np.min(np.diff(s["text"]))),
             "text_interarrival_max_days": float(np.max(np.diff(s["text"]))) }
            for s in schedules]


def encode_phrases(cfg: dict, out: Path) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(cfg["threads"])
    path = Path(cfg["encoder_path"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval().cpu()
    phrases = [x for pair in cfg["train_phrase_pairs"] + cfg["heldout_phrase_pairs"] for x in pair]
    with torch.no_grad():
        inputs = tokenizer(phrases, padding=True, truncation=True, max_length=128, return_tensors="pt")
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        emb = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
    train = emb[:8]
    center = train.mean(0)
    _, singular, vt = np.linalg.svd(train - center, full_matrices=False)
    ncomp = min(cfg["train_pca_max_components"], int(np.sum(singular > 1e-6)))
    components = vt[:ncomp].T.copy()
    encoded = (emb - center) @ components
    decoder = Ridge(alpha=0.01).fit(encoded[:8], np.array([1., -1.] * 4))
    probe = decoder.predict(encoded[8:])
    metadata = {"model": cfg["encoder_id"], "revision": cfg["encoder_revision"],
                "pooling": "attention-mask mean of last hidden states, L2 normalization",
                "tokenizer_max_length": 128, "device": "cpu", "embedding_dim": emb.shape[1],
                "PCA": "center and basis fitted only on the eight train phrases; no test labels",
                "retained_components": ncomp, "singular_values": singular.tolist(),
                "phrase_order": phrases, "heldout_phrase_sign_probe_predictions": probe.tolist(),
                "heldout_phrase_sign_probe_accuracy": float(np.mean((probe > 0) == (np.array([1, -1] * 4) > 0))),
                "probe_boundary": "eight templated heldout sentences, a diagnostic, not unrestricted semantic comprehension",
                "model_files": {str(p.relative_to(path)).replace("\\", "/"): sha(p)
                                for p in sorted(path.rglob("*")) if p.is_file()}}
    np.savez_compressed(out / "phrase_embeddings.npz", embeddings=emb, center=center,
                        components=components, projected=encoded)
    dump(out / "encoder_identity.json", metadata)
    return {"projected": encoded, "metadata": metadata}


def history_indices(event_time: np.ndarray, arrival_time: np.ndarray, cutoff: float, history: float) -> np.ndarray:
    # Both event occurrence and arrival are required. Future/unarrived notes never enter.
    return np.flatnonzero((event_time > cutoff - history) & (event_time <= cutoff) & (arrival_time <= cutoff))


def combine(values: np.ndarray, ages: np.ndarray, tau: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(values.shape[1], dtype=float)
    weights = np.exp(-ages / tau)
    return (weights[:, None] * values).sum(0) / np.sqrt(np.sum(weights ** 2))


def cyclic_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    # Cyclic rotation has zero index fixed points. Duplicate content may remain.
    return np.roll(np.arange(n), int(rng.integers(1, n)))


def generate_world(cfg: dict, schedules: list[dict], representation: dict, data_seed: int) -> tuple[dict, list[dict]]:
    splits = {}
    notes = []
    for split_index, (split, nentities) in enumerate(cfg["entities_per_split"].items()):
        numeric_features, oracle, oracle_shuffle, encoded, encoded_shuffle = [], [], [], [], []
        baseline, noise, ids, templates, counts, cutoff_values, changed, changed_content, l2dose = [], [], [], [], [], [], [], [], []
        for entity in range(nentities):
            schedule_index = entity % len(schedules)
            s = schedules[schedule_index]
            # Streams separate values/signs/delay/template/control so nuisance draws
            # cannot reveal signs or change when signal strength changes.
            streams = np.random.SeedSequence([data_seed, split_index, entity]).spawn(6)
            numeric_rng, sign_rng, delay_rng, phrase_rng, control_rng, noise_rng = [np.random.default_rng(z) for z in streams]
            nt, tt = s["numeric"], s["text"]
            x = np.empty(len(nt))
            x[0] = numeric_rng.normal()
            gaps = np.maximum(np.diff(nt), 0)
            unit = np.median(gaps[gaps > 0])
            phi = cfg["numeric_ar1_phi"] ** (gaps / unit)
            for i, p in enumerate(phi, start=1):
                x[i] = p * x[i - 1] + np.sqrt(max(0, 1 - p * p)) * numeric_rng.normal()
            signs = sign_rng.choice(np.array([-1., 1.]), len(tt))
            delays = delay_rng.exponential(cfg["simulated_arrival_delay_mean_days"], len(tt))
            arrival = tt + delays
            pair = phrase_rng.integers(0, 4, len(tt)) + (0 if split == "train" else 4)
            phrase_indices = pair * 2 + (signs < 0).astype(int)
            text_values = representation["projected"][phrase_indices]
            for j, (t, a, sg, pi) in enumerate(zip(tt, arrival, signs, phrase_indices)):
                notes.append({"world_seed": data_seed, "split": split, "entity_id": entity,
                              "schedule_id": schedule_index, "event_id": j, "event_time_days": float(t),
                              "simulated_arrival_time_days": float(a), "direction_truth": float(sg),
                              "phrase_index": int(pi)})
            for query, cutoff in enumerate(s["cutoffs"]):
                idx = np.flatnonzero((nt > cutoff - cfg["history_days"]) & (nt <= cutoff))
                values, times = x[idx], nt[idx]
                last, mean, sd = values[-1], values.mean(), values.std()
                slope = (values[-1] - values[0]) / max(times[-1] - times[0], 1 / 24)
                numeric_features.append([last, mean, sd, slope, (cutoff - times[-1]) / 7])
                baseline.append(0.8 * last + 0.2 * mean)
                noise.append(noise_rng.normal(0, cfg["target_noise_sd"]))
                avail = history_indices(tt, arrival, cutoff, cfg["history_days"])
                age = cutoff - tt[avail]
                perm = cyclic_permutation(len(avail), control_rng)
                a = signs[avail, None]
                e = text_values[avail]
                oracle.append(combine(a, age, cfg["semantic_decay_days"]))
                oracle_shuffle.append(combine(a[perm], age, cfg["semantic_decay_days"]))
                encoded.append(combine(e, age, cfg["semantic_decay_days"]))
                encoded_shuffle.append(combine(e[perm], age, cfg["semantic_decay_days"]))
                l2dose.append(float(np.linalg.norm(encoded[-1] - encoded_shuffle[-1])))
                counts.append(len(avail))
                cutoff_values.append(float(cutoff))
                ids.append(entity)
                templates.append(schedule_index)
                changed.append(int(np.sum(perm != np.arange(len(avail)))))
                changed_content.append(int(np.sum(signs[avail] != signs[avail][perm])))
        counts_arr = np.array(counts)
        # Numerical+event-process features are common to all representations.
        timing = np.stack([counts_arr / 7, np.array(cutoff_values) / 300, (counts_arr > 0).astype(float)], 1)
        common = np.column_stack([np.array(numeric_features), timing])
        splits[split] = {"common": common, "numeric": np.array(numeric_features),
                         "oracle": np.array(oracle), "oracle_shuffle": np.array(oracle_shuffle),
                         "sentence": np.array(encoded), "sentence_shuffle": np.array(encoded_shuffle),
                         "baseline": np.array(baseline), "noise": np.array(noise),
                         "entity": np.array(ids), "schedule": np.array(templates),
                         "event_count": counts_arr, "cutoff": np.array(cutoff_values),
                         "changed_positions": np.array(changed), "changed_directions": np.array(changed_content),
                         "embedding_l2_dose": np.array(l2dose)}
    return splits, notes


def paired(values: np.ndarray, alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    effect = float(values.mean())
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(1 - alpha / 2, n - 1))
    if sem == 0:
        p = 0.0 if effect > 0 else 1.0
    else:
        p = float(stats.t.sf(effect / sem, n - 1))
    return {"n_independent_entities": n, "mean_loss_difference": effect,
            "ci95": [effect - critical * sem, effect + critical * sem],
            "one_sided_p": p, "reject": bool(p < alpha)}


def entity_means(loss: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.array([loss[ids == k].mean() for k in np.unique(ids)])


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, variants: dict[str, np.ndarray], seed: int, cfg: dict) -> tuple[dict, dict, dict]:
    scaler = StandardScaler().fit(train_x)
    reg = SGDRegressor(loss="squared_error", penalty="l2", alpha=cfg["sgd"]["alpha"],
                       max_iter=cfg["sgd"]["max_iter"], tol=cfg["sgd"]["tol"],
                       learning_rate="invscaling", eta0=cfg["sgd"]["eta0"],
                       power_t=cfg["sgd"]["power_t"], average=True, random_state=seed,
                       shuffle=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        reg.fit(scaler.transform(train_x), train_y)
    predictions = {k: reg.predict(scaler.transform(v)) for k, v in variants.items()}
    assert all(np.isfinite(v).all() for v in predictions.values())
    checkpoint = {"coefficient": reg.coef_.copy(), "intercept": reg.intercept_.copy(),
                  "scaler_mean": scaler.mean_.copy(), "scaler_scale": scaler.scale_.copy()}
    identity = {"seed": seed, "n_iter": int(reg.n_iter_), "convergence_warning": [str(x.message) for x in caught],
                "train_shape": list(train_x.shape), "model": "sklearn.SGDRegressor, averaged stochastic SGD, L2 linear head"}
    return predictions, identity, checkpoint


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate_ci(k: int, n: int) -> list[float]:
    z = stats.norm.ppf(0.975)
    phat = k / n
    denominator = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return [float(center - half), float(center + half)]


def freeze(cfg: dict, schedules: list[dict], output: Path) -> dict:
    path = output / "FROZEN_PROTOCOL.json"
    source = {"config.json": sha(HERE / "config.json"), "run_calibration.py": sha(Path(__file__)),
              "test_calibration.py": sha(HERE / "test_calibration.py")}
    if path.exists():
        identity = json.loads(path.read_text(encoding="utf-8"))
        assert identity["implementation_hashes"] == source, "Frozen implementation changed"
        assert identity["source_schedule_identity"] == schedule_identity(schedules), "Source changed"
        return identity
    assert not (output / "world_results.jsonl").exists(), "Never retrospectively freeze evaluated results"
    identity = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": cfg["protocol_id"],
                "implementation_hashes": source, "config": cfg,
                "source_schedule_identity": schedule_identity(schedules),
                "registration_status": "local time-stamped protocol freeze before generated outcome evaluation; not externally preregistered",
                "past_exposure": "The source dataset and original manuscript test results have already been examined; new generated numeric/text/noise draws are distinct.",
                "analysis_unit": "generated entity conditional on empirical schedules; training seeds averaged within entity; detection rates use independently generated worlds",
                "availability": "event_time <= cutoff AND simulated_arrival_time <= cutoff within seven-day history; simulated independent exponential latency is not observed real publication latency",
                "primary": "fixed aligned SGD checkpoint: aligned vs within-current-available-history cyclic content permutation; one-sided paired entity-level t test alpha=.05; positive delta means permutation harmed prediction",
                "secondary": "fixed zero-content test control, retrained shuffled model, numerical-plus-timing retrained comparator; validation capability gate requires positive entity-level lower CI and at least 1% relative improvement over timing baseline",
                "multiplicity": "each strength/representation is a pre-specified calibration condition; rate intervals reported descriptively; no pooled claim from selecting significant cells",
                "target": "0.8 last_numeric + 0.2 history_mean + strength * variance-normalized exponentially weighted available random text direction + independent N(0, .35²) noise",
                "scope": "independent recency-pooled linear prediction pipeline on controlled templated prose; oracle direction vectors alone do not validate language representations or Time-IMM fusion architectures"}
    dump(path, identity)
    return identity


def run(cfg: dict, output: Path, only_worlds: int | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    schedules = load_schedules(cfg)
    frozen = freeze(cfg, schedules, output)
    if (output / "world_results.jsonl").exists():
        raise RuntimeError("Existing evaluated run preserved. Use --summarize or a new output directory.")
    rep = encode_phrases(cfg, output)
    results, seed_rows, unit_rows, diagnostics, identities = [], [], [], [], []
    data_seeds = cfg["data_seeds"][:only_worlds] if only_worlds else cfg["data_seeds"]
    start = time.time()
    (output / "raw").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    for world_index, data_seed in enumerate(data_seeds):
        splits, notes = generate_world(cfg, schedules, rep, data_seed)
        raw = {f"{s}_{k}": v for s, d in splits.items() for k, v in d.items()}
        np.savez_compressed(output / "raw" / f"world_{data_seed}.npz", **raw)
        write_csv(output / "raw" / f"text_events_{data_seed}.csv", notes)
        for split, d in splits.items():
            total = d["event_count"].sum()
            diagnostics.append({"world_seed": data_seed, "split": split, "entities": int(len(np.unique(d["entity"]))),
                                "queries": len(d["entity"]), "zero_note_queries": int((d["event_count"] == 0).sum()),
                                "one_note_queries": int((d["event_count"] == 1).sum()),
                                "multi_note_queries": int((d["event_count"] >= 2).sum()),
                                "text_positions": int(total), "changed_positions": int(d["changed_positions"].sum()),
                                "changed_directions": int(d["changed_directions"].sum()),
                                "median_embedding_aggregate_l2_change": float(np.median(d["embedding_l2_dose"]))})
        for strength in cfg["strengths"]:
            ys = {s: d["baseline"] + strength * d["oracle"].ravel() + d["noise"] for s, d in splits.items()}
            collected = {r: {k: [] for k in ["aligned", "fixed_shuffle", "fixed_zero", "retrained_shuffle", "timing"]}
                         for r in ["oracle", "sentence"]}
            collected_val = {r: {k: [] for k in ["aligned", "timing"]} for r in ["oracle", "sentence"]}
            for training_seed in cfg["training_seeds"]:
                timing_preds, identity, checkpoint = fit_predict(splits["train"]["common"], ys["train"],
                          {s: d["common"] for s, d in splits.items() if s != "train"}, training_seed, cfg)
                identities.append({"world_seed": data_seed, "strength": strength, "representation": "timing",
                                   "condition": "retrained", **identity})
                np.savez_compressed(output / "models" / f"w{data_seed}_s{strength}_t{training_seed}_timing.npz", **checkpoint)
                timing_loss = (timing_preds["test"] - ys["test"]) ** 2
                timing_val = (timing_preds["val"] - ys["val"]) ** 2
                for representation in ["oracle", "sentence"]:
                    def design(split, condition="aligned"):
                        d = splits[split]
                        z = d[representation + ("_shuffle" if condition == "shuffle" else "")]
                        if condition == "zero":
                            z = np.zeros_like(z)
                        return np.column_stack([d["common"], z])
                    variants = {"test": design("test"), "val": design("val"),
                                "shuffle": design("test", "shuffle"), "zero": design("test", "zero")}
                    pred, identity, checkpoint = fit_predict(design("train"), ys["train"], variants, training_seed, cfg)
                    identities.append({"world_seed": data_seed, "strength": strength, "representation": representation,
                                       "condition": "aligned", **identity})
                    np.savez_compressed(output / "models" / f"w{data_seed}_s{strength}_t{training_seed}_{representation}_aligned.npz", **checkpoint)
                    retrain, identity, checkpoint = fit_predict(design("train", "shuffle"), ys["train"],
                          {"test": design("test", "shuffle")}, training_seed, cfg)
                    identities.append({"world_seed": data_seed, "strength": strength, "representation": representation,
                                       "condition": "shuffled_train_and_test", **identity})
                    np.savez_compressed(output / "models" / f"w{data_seed}_s{strength}_t{training_seed}_{representation}_retrained_shuffle.npz", **checkpoint)
                    losses = {"aligned": (pred["test"] - ys["test"]) ** 2,
                              "fixed_shuffle": (pred["shuffle"] - ys["test"]) ** 2,
                              "fixed_zero": (pred["zero"] - ys["test"]) ** 2,
                              "retrained_shuffle": (retrain["test"] - ys["test"]) ** 2, "timing": timing_loss}
                    for k, loss in losses.items():
                        collected[representation][k].append(entity_means(loss, splits["test"]["entity"]))
                    collected_val[representation]["aligned"].append(entity_means((pred["val"] - ys["val"]) ** 2, splits["val"]["entity"]))
                    collected_val[representation]["timing"].append(entity_means(timing_val, splits["val"]["entity"]))
                    single = paired(entity_means(losses["fixed_shuffle"] - losses["aligned"], splits["test"]["entity"]))
                    seed_rows.append({"world_seed": data_seed, "strength": strength, "representation": representation,
                                      "training_seed": training_seed, **{k: float(v.mean()) for k, v in losses.items()},
                                      "fixed_shuffle_delta": single["mean_loss_difference"], "p": single["one_sided_p"]})
            for representation in ["oracle", "sentence"]:
                avg = {k: np.mean(v, axis=0) for k, v in collected[representation].items()}
                val = {k: np.mean(v, axis=0) for k, v in collected_val[representation].items()}
                capability = paired(val["timing"] - val["aligned"])
                relative_gain = float((val["timing"].mean() - val["aligned"].mean()) / val["timing"].mean())
                qualified = capability["ci95"][0] > 0 and relative_gain > cfg["validation_capability_min_relative_gain"]
                primary = paired(avg["fixed_shuffle"] - avg["aligned"], cfg["primary_alpha_one_sided"])
                result = {"world_seed": data_seed, "strength": strength, "representation": representation,
                          "test_entity_MSE": {k: float(v.mean()) for k, v in avg.items()},
                          "primary_fixed_shuffle": primary,
                          "fixed_zero": paired(avg["fixed_zero"] - avg["aligned"]),
                          "retrained_shuffle": paired(avg["retrained_shuffle"] - avg["aligned"]),
                          "timing": paired(avg["timing"] - avg["aligned"]),
                          "validation_capability": capability, "validation_relative_gain": relative_gain,
                          "capability_qualified": bool(qualified), "gated_detection": bool(qualified and primary["reject"])}
                results.append(result)
                with (output / "world_results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                for entity in range(cfg["entities_per_split"]["test"]):
                    unit_rows.append({"world_seed": data_seed, "strength": strength, "representation": representation,
                                      "entity_id": entity, "schedule_id": entity % len(schedules),
                                      **{k: float(v[entity]) for k, v in avg.items()}})
        print(f"world {world_index + 1}/{len(data_seeds)} seed={data_seed} elapsed={time.time()-start:.1f}s", flush=True)
        write_csv(output / "seed_results.csv", seed_rows)
        write_csv(output / "entity_loss_results.csv", unit_rows)
        write_csv(output / "input_diagnostics.csv", diagnostics)
        dump(output / "model_training_identity.json", identities)
    summarize(cfg, output)
    import sklearn
    dump(output / "run_identity.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
         "frozen_protocol_sha256": sha(output / "FROZEN_PROTOCOL.json"), "elapsed_seconds": time.time()-start,
         "python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
         "platform": platform.platform(), "device": "cpu", "threads": cfg["threads"],
         "worlds_executed": len(data_seeds), "fit_count": len(identities),
         "results_sha256": sha(output / "world_results.jsonl"),
         "raw_data_sha256": {p.name: sha(p) for p in sorted((output / "raw").glob("*"))},
         "model_checkpoint_count": len(list((output / "models").glob("*.npz")))})


def summarize(cfg: dict, output: Path) -> None:
    results = [json.loads(z) for z in (output / "world_results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = []
    for representation in ["oracle", "sentence"]:
        for strength in cfg["strengths"]:
            subset = [r for r in results if r["representation"] == representation and r["strength"] == strength]
            n = len(subset)
            detected = sum(r["primary_fixed_shuffle"]["reject"] for r in subset)
            gated = sum(r["gated_detection"] for r in subset)
            delta = np.array([r["primary_fixed_shuffle"]["mean_loss_difference"] for r in subset])
            mean_ci = paired(delta)["ci95"] if n > 1 else [None, None]
            summary.append({"representation": representation, "strength": strength, "worlds": n,
                            "test_entities_per_world": cfg["entities_per_split"]["test"],
                            "training_seeds_per_world": len(cfg["training_seeds"]),
                            "detected_worlds": int(detected), "detection_rate": detected / n,
                            "detection_rate_wilson95": rate_ci(detected, n),
                            "capability_qualified_worlds": sum(r["capability_qualified"] for r in subset),
                            "gated_detected_worlds": int(gated), "gated_detection_rate": gated / n,
                            "gated_rate_wilson95": rate_ci(gated, n),
                            "mean_fixed_shuffle_delta": float(delta.mean()), "mean_delta_world_ci95": mean_ci,
                            "mean_relative_shuffle_degradation": float(np.mean([r["primary_fixed_shuffle"]["mean_loss_difference"] / r["test_entity_MSE"]["fixed_shuffle"] for r in subset])),
                            "mean_test_MSE": {k: float(np.mean([r["test_entity_MSE"][k] for r in subset]))
                                              for k in subset[0]["test_entity_MSE"]}})
    dump(output / "summary.json", {"protocol_id": cfg["protocol_id"], "conditions": summary,
           "limits": ["Controlled template language and synthetic targets, not real-world semantic utility.",
                      "Empirical schedules reused across generated entities; inference is conditional on those fixed schedules.",
                      "SGD training seeds are averaged, not counted as independent domains or data replications.",
                      "Single-note history windows cannot support within-history content permutation; these remain in analysis.",
                      "Null rejection rate from 20 worlds has limited resolution and broad uncertainty.",
                      "These linear pipeline results do not establish sensitivity of the original Time-IMM fusion models."]})
    flat = [{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))},
             "rate_ci_low": r["detection_rate_wilson95"][0], "rate_ci_high": r["detection_rate_wilson95"][1],
             "delta_ci_low": r["mean_delta_world_ci95"][0], "delta_ci_high": r["mean_delta_world_ci95"][1],
             **{f"mse_{k}": v for k, v in r["mean_test_MSE"].items()}} for r in summary]
    write_csv(output / "summary.csv", flat)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--output", type=Path, default=HERE / "results")
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--only-worlds", type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with threadpool_limits(limits=config["threads"]):
        if args.freeze_only:
            args.output.mkdir(parents=True, exist_ok=True)
            print(json.dumps(freeze(config, load_schedules(config), args.output), indent=2))
        elif args.summarize:
            summarize(config, args.output)
        else:
            run(config, args.output, args.only_worlds)
