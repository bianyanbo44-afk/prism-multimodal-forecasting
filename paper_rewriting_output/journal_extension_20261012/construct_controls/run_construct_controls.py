"""Separate controlled proposition dependence from arbitrary source proxies.

Synthetic constructs/values/targets reuse real event clocks. The sentence
encoder is actual pinned MiniLM; d/p are never supplied to the text predictor.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE.parent / "calibration/run_calibration.py"
sys.path.insert(0, str(HELPER_PATH.parent))
from run_calibration import load_schedules, schedule_identity, history_indices, combine, cyclic_permutation, paired, entity_means, rate_ci


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(p, rows):
    with Path(p).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sentences(cfg):
    phrases, directions, proxies = [], [], []
    for pairs in [cfg["train_direction_pairs"], cfg["test_direction_pairs"], cfg["paraphrase_direction_pairs"]]:
        for pair in pairs:
            for di, proposition in enumerate(pair):
                for pi, suffix in enumerate([cfg["proxy_positive_suffix"], cfg["proxy_negative_suffix"]]):
                    phrases.append(proposition + suffix)
                    directions.append(1 - 2 * di)
                    proxies.append(1 - 2 * pi)
    return phrases, np.array(directions), np.array(proxies)


def phrase_index(pair_index, direction, proxy):
    return pair_index * 4 + (direction < 0).astype(int) * 2 + (proxy < 0).astype(int)


def make_encoder(cfg, output):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(cfg["threads"])
    phrases, d, p = sentences(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_path"], local_files_only=True)
    model = AutoModel.from_pretrained(cfg["encoder_path"], local_files_only=True).cpu().eval()
    with torch.no_grad():
        tok = tokenizer(phrases, padding=True, truncation=True, max_length=128, return_tensors="pt")
        hidden = model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1)
        embedding = (hidden * mask).sum(1) / mask.sum(1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).numpy()
    center = embedding[:16].mean(0)
    _, singular, vt = np.linalg.svd(embedding[:16] - center, full_matrices=False)
    n = min(cfg["pca_max_components"], int(np.sum(singular > 1e-6)))
    components = vt[:n].T
    projected = (embedding - center) @ components
    probe_results = {}
    for name, labels in [("direction", d), ("source_proxy", p)]:
        probe = Ridge(alpha=.01).fit(projected[:16], labels[:16])
        predictions = probe.predict(projected)
        probe_results[name] = {"train_accuracy": float(np.mean((predictions[:16] > 0) == (labels[:16] > 0))),
                              "test_accuracy": float(np.mean((predictions[16:32] > 0) == (labels[16:32] > 0))),
                              "paraphrase_accuracy": float(np.mean((predictions[32:] > 0) == (labels[32:] > 0))),
                              "predictions": predictions.tolist()}
    encoder_path = Path(cfg["encoder_path"])
    dump(output / "encoder_identity.json", {"model": "sentence-transformers/all-MiniLM-L6-v2",
         "revision": cfg["encoder_revision"], "pooling": "attention-mask mean, L2 normalization",
         "device": "cpu", "max_token_length": 128, "pca_training_sentence_count": 16,
         "pca_components": n, "singular_values": singular.tolist(),
         "phrase_order": phrases, "direction_truth": d.tolist(), "proxy_truth": p.tolist(),
         "probe_results": probe_results, "probe_limit": "Finite template direction and arbitrary metadata constructs; no universal semantic competence claim.",
         "model_files": {p.relative_to(encoder_path).as_posix(): sha(p) for p in sorted(encoder_path.rglob("*")) if p.is_file()}})
    np.savez_compressed(output / "encoder_arrays.npz", embedding=embedding, projected=projected,
                        center=center, components=components, direction=d, proxy=p)
    return projected


def generate(cfg, schedules, enc, world_seed):
    data, notes = {}, []
    for split_index, (split, nentities) in enumerate(cfg["entities_per_split"].items()):
        common, baseline, noise, dsignal, psignal, ids, counts, clocks = [], [], [], [], [], [], [], []
        represented = {k: [] for k in cfg["conditions"]}
        for entity in range(nentities):
            s = schedules[entity % len(schedules)]
            seeds = np.random.SeedSequence([world_seed, split_index, entity]).spawn(7)
            nrng, drng, prng, arng, trng, crng, yrng = [np.random.default_rng(z) for z in seeds]
            nt, tt = s["numeric"], s["text"]
            values = np.empty(len(nt))
            values[0] = nrng.normal()
            dt = np.maximum(0., np.diff(nt))
            unit = np.median(dt[dt > 0])
            phis = cfg["numeric_ar1_phi"] ** (dt / unit)
            for j, phi in enumerate(phis, start=1):
                values[j] = phi * values[j-1] + np.sqrt(max(0., 1-phi*phi)) * nrng.normal()
            d = drng.choice(np.array([-1., 1.]), len(tt))
            p = prng.choice(np.array([-1., 1.]), len(tt))
            arrival = tt + arng.exponential(cfg["arrival_delay_mean_days"], len(tt))
            template = trng.integers(0, 4, len(tt)) + (0 if split == "train" else 4)
            idx = phrase_index(template, d, p)
            payload = enc[idx]
            for event, (event_time, arrived, direction, proxy, phrase) in enumerate(zip(tt, arrival, d, p, idx)):
                notes.append({"world_seed": world_seed, "split": split, "entity_id": entity, "event_id": event,
                              "schedule_id": entity % len(schedules), "event_time": float(event_time),
                              "simulated_arrival": float(arrived), "direction": float(direction),
                              "proxy": float(proxy), "phrase_index": int(phrase)})
            for cutoff in s["cutoffs"]:
                ni = np.flatnonzero((nt > cutoff-cfg["history_days"]) & (nt <= cutoff))
                x, times = values[ni], nt[ni]
                avail = history_indices(tt, arrival, cutoff, cfg["history_days"])
                age = cutoff - tt[avail]
                rotation = cyclic_permutation(len(avail), crng)
                row = [x[-1], x.mean(), x.std(), (x[-1]-x[0])/max(times[-1]-times[0], 1/24),
                       (cutoff-times[-1])/7, np.log1p(len(avail)), (age.min()/7 if len(age) else 1), float(len(age)>0)]
                common.append(row)
                baseline.append(.8*x[-1]+.2*x.mean())
                noise.append(yrng.normal(0, cfg["noise_sd"]))
                dsignal.append(combine(d[avail,None], age, cfg["semantic_decay_days"])[0])
                psignal.append(combine(p[avail,None], age, cfg["semantic_decay_days"])[0])
                forms = {"aligned": payload[avail], "whole_permutation": payload[avail][rotation],
                         "meaning_flip": enc[idx[avail] ^ 2], "proxy_flip": enc[idx[avail] ^ 1],
                         "meaning_preserving_paraphrase": enc[idx[avail] + (32 if split == "train" else 16)],
                         "zero": np.zeros_like(payload[avail])}
                for condition, z in forms.items():
                    represented[condition].append(combine(z, age, cfg["semantic_decay_days"]))
                ids.append(entity)
                counts.append(len(avail))
                clocks.append(float(cutoff))
        data[split] = {"common": np.array(common), "baseline": np.array(baseline), "noise": np.array(noise),
                       "direction_signal": np.array(dsignal), "proxy_signal": np.array(psignal),
                       "entity": np.array(ids), "event_count": np.array(counts), "cutoff": np.array(clocks),
                       **{k: np.array(v) for k, v in represented.items()}}
    train_logcounts = data["train"]["common"][:, 5]
    timing_center, timing_scale = train_logcounts.mean(), train_logcounts.std()
    for split in data:
        data[split]["timing_signal"] = (data[split]["common"][:, 5] - timing_center) / timing_scale
    return data, notes, {"timing_center_train_only": timing_center, "timing_scale_train_only": timing_scale}


def select_ridge(train_x, train_y, val_x, val_y, val_ids, cfg):
    scaler = StandardScaler().fit(train_x)
    tx, vx = scaler.transform(train_x), scaler.transform(val_x)
    candidates, models = [], []
    for alpha in cfg["ridge_alphas"]:
        reg = Ridge(alpha=alpha, solver="svd").fit(tx, train_y)
        mse = entity_means((reg.predict(vx)-val_y)**2, val_ids).mean()
        candidates.append({"alpha": alpha, "validation_entity_MSE": float(mse)})
        models.append(reg)
    best = int(np.argmin([x["validation_entity_MSE"] for x in candidates]))
    model = models[best]
    checkpoint = {"coef": model.coef_, "intercept": np.array([model.intercept_]),
                  "scale_mean": scaler.mean_, "scale_std": scaler.scale_}
    return model, scaler, {"candidates": candidates, "selected_alpha": candidates[best]["alpha"]}, checkpoint


def predict_saved(path, x):
    c = np.load(path)
    return ((x-c["scale_mean"])/c["scale_std"]) @ c["coef"] + c["intercept"][0]


def freeze(cfg, schedules, output):
    identities = {"config.json": sha(HERE / "config.json"), "run_construct_controls.py": sha(Path(__file__)),
                  "test_construct_controls.py": sha(HERE / "test_construct_controls.py"),
                  "read_only_E1_helper": sha(HELPER_PATH)}
    path = output / "FROZEN_PROTOCOL.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        assert old["implementation_hashes"] == identities
        assert old["source_schedule_identity"] == schedule_identity(schedules)
        return old
    assert not (output / "world_results.jsonl").exists()
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "implementation_hashes": identities, "source_schedule_identity": schedule_identity(schedules),
              "registration": "Local freeze before generated outcomes; not an external preregistration.",
              "past_exposure": "E1 calibration and original source/manuscript results already viewed; new world seeds and all d/p/noise draws fixed prospectively here.",
              "generator_constructs": "Semantic target uses available direction truth; proxy target uses available arbitrary Cedar/Maple source truth; timing target uses train-standardized log count. All are constructed, not natural causal claims.",
              "availability": "Both occurrence and independently simulated arrival by cutoff; seven-day nonoverlapping histories.",
              "primary_comparisons": "Fixed aligned checkpoint vs whole permutation / meaning flip / proxy flip; no fitting between test controls.",
              "paraphrase": "Same generated directional truth and source cue, different heldout proposition wording; representation equivalence is not assumed.",
              "model_selection": "Train-only scaling/PCA; ridge alpha selected on mean entity loss in validation, never test.",
              "inference": "128 independent generated entities conditional on empirical clocks; 20 worlds for rate/effect intervals; deterministic ridge has no artificial training-seed replications.",
              "multiplicity": "Mechanism/control cells fixed beforehand, all reported; rejection-rate comparisons are descriptive calibration, no selected-cell global inference.",
              "scope": "Generator-defined two-proposition and source-proxy constructs in finite templated language; no universal natural-language semantic detector claim."}
    dump(path, record)
    return record


def run(cfg, output):
    output.mkdir(parents=True, exist_ok=True)
    schedules = load_schedules(cfg)
    freeze(cfg, schedules, output)
    if (output / "world_results.jsonl").exists():
        raise RuntimeError("Scored run preserved; select a fresh output path.")
    enc = make_encoder(cfg, output)
    (output / "raw").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    started = time.time()
    units, model_records, diagnostics, world_results = [], [], [], []
    maximum_error, verification_count = 0., 0
    for number, seed in enumerate(cfg["data_seeds"]):
        data, notes, timing_identity = generate(cfg, schedules, enc, seed)
        np.savez_compressed(output / "raw" / f"world_{seed}.npz", **{f"{s}_{k}": v for s,d in data.items() for k,v in d.items()})
        write_csv(output / "raw" / f"text_events_{seed}.csv", notes)
        for split,d in data.items():
            diagnostics.append({"world_seed": seed, "split": split, "queries": len(d["entity"]),
                                "zero_note": int((d["event_count"]==0).sum()), "one_note": int((d["event_count"]==1).sum()),
                                "multiple_notes": int((d["event_count"]>=2).sum()), **timing_identity})
        for mechanism in cfg["mechanisms"]:
            field = {"semantic_only": "direction_signal", "proxy_only": "proxy_signal", "timing_only": "timing_signal"}[mechanism]
            y = {s: d["baseline"] + d[field] + d["noise"] for s,d in data.items()}
            def design(split, condition="aligned"):
                return np.column_stack([data[split]["common"], data[split][condition]])
            reg, scaler, selected, checkpoint = select_ridge(design("train"), y["train"], design("val"), y["val"], data["val"]["entity"], cfg)
            path = output / "models" / f"w{seed}_{mechanism}_aligned.npz"
            np.savez_compressed(path, **checkpoint)
            val_aligned = entity_means((reg.predict(scaler.transform(design("val")))-y["val"])**2, data["val"]["entity"])
            model_records.append({"world_seed": seed, "mechanism": mechanism, "model": "aligned_text", **selected})
            baseline_reg, baseline_scaler, selected, baseline_checkpoint = select_ridge(data["train"]["common"], y["train"], data["val"]["common"], y["val"], data["val"]["entity"], cfg)
            baseline_path = output / "models" / f"w{seed}_{mechanism}_numeric_timing.npz"
            np.savez_compressed(baseline_path, **baseline_checkpoint)
            model_records.append({"world_seed": seed, "mechanism": mechanism, "model": "numeric_timing", **selected})
            val_baseline = entity_means((baseline_reg.predict(baseline_scaler.transform(data["val"]["common"]))-y["val"])**2, data["val"]["entity"])
            cap = paired(val_baseline-val_aligned)
            relative_gain = float((val_baseline.mean()-val_aligned.mean())/val_baseline.mean())
            gate = cap["ci95"][0]>0 and relative_gain>cfg["capability_min_relative_validation_gain"]
            losses = {}
            for condition in cfg["conditions"]:
                pred = reg.predict(scaler.transform(design("test", condition)))
                reconstructed = predict_saved(path, design("test", condition))
                error = float(np.max(np.abs(pred-reconstructed)))
                maximum_error = max(maximum_error, error)
                assert error < 1e-12
                verification_count += len(pred)
                losses[condition] = entity_means((pred-y["test"])**2, data["test"]["entity"])
            bp = baseline_reg.predict(baseline_scaler.transform(data["test"]["common"]))
            losses["numeric_timing"] = entity_means((bp-y["test"])**2, data["test"]["entity"])
            losses["oracle_readout"] = entity_means(data["test"]["noise"]**2, data["test"]["entity"])
            comparisons = {k: paired(losses[k]-losses["aligned"], cfg["paired_alpha_one_sided"]) for k in cfg["conditions"] if k!="aligned"}
            result = {"world_seed": seed, "mechanism": mechanism, "test_entity_MSE": {k: float(v.mean()) for k,v in losses.items()},
                      "paired_comparisons": comparisons, "validation_capability": cap,
                      "validation_relative_gain": relative_gain, "capability_qualified": bool(gate),
                      "gated_whole_permutation_detection": bool(gate and comparisons["whole_permutation"]["reject"])}
            world_results.append(result)
            with (output / "world_results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result)+"\n")
            for entity in range(cfg["entities_per_split"]["test"]):
                units.append({"world_seed": seed, "mechanism": mechanism, "entity_id": entity,
                              "schedule_id": entity%len(schedules), **{k: float(v[entity]) for k,v in losses.items()}})
        print(f"world {number+1}/{len(cfg['data_seeds'])} seed={seed} elapsed={time.time()-started:.1f}s", flush=True)
        write_csv(output / "entity_losses.csv", units)
        write_csv(output / "input_diagnostics.csv", diagnostics)
        dump(output / "model_selection.json", model_records)
    summary = []
    for mechanism in cfg["mechanisms"]:
        records = [r for r in world_results if r["mechanism"]==mechanism]
        n = len(records)
        for condition in cfg["conditions"]:
            if condition == "aligned":
                continue
            delta = np.array([r["paired_comparisons"][condition]["mean_loss_difference"] for r in records])
            k = sum(r["paired_comparisons"][condition]["reject"] for r in records)
            summary.append({"mechanism": mechanism, "condition": condition, "worlds": n,
                            "mean_delta_MSE": float(delta.mean()), "delta_world_mean_ci95": paired(delta)["ci95"],
                            "rejected_worlds": k, "rejection_rate": k/n, "rate_wilson95": rate_ci(k,n),
                            "capability_qualified_worlds": sum(r["capability_qualified"] for r in records),
                            "mean_aligned_MSE": float(np.mean([r["test_entity_MSE"]["aligned"] for r in records])),
                            "mean_control_MSE": float(np.mean([r["test_entity_MSE"][condition] for r in records])),
                            "mean_numeric_timing_MSE": float(np.mean([r["test_entity_MSE"]["numeric_timing"] for r in records])),
                            "mean_oracle_readout_MSE": float(np.mean([r["test_entity_MSE"]["oracle_readout"] for r in records]))})
    dump(output / "summary.json", {"conditions": summary, "scope": "Generated construct validation, not unrestricted semantics detection."})
    write_csv(output / "summary.csv", [{**{k:v for k,v in r.items() if not isinstance(v,list)},
                                       "delta_ci_low": r["delta_world_mean_ci95"][0], "delta_ci_high": r["delta_world_mean_ci95"][1],
                                       "rate_ci_low": r["rate_wilson95"][0], "rate_ci_high": r["rate_wilson95"][1]} for r in summary])
    dump(output / "VERIFICATION.json", {"checkpoint_prediction_checks": verification_count,
         "max_absolute_prediction_error": maximum_error, "fixed_checkpoint_shared_for_all_controls": True,
         "model_fits": len(model_records)*len(cfg["ridge_alphas"]), "saved_selected_checkpoints": len(model_records),
         "all_candidates_use_only_train_and_validation": True})
    dump(output / "run_identity.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
         "elapsed_seconds": time.time()-started, "device": "cpu", "threads": cfg["threads"],
         "frozen_protocol_sha256": sha(output / "FROZEN_PROTOCOL.json"), "results_sha256": sha(output / "world_results.jsonl"),
         "raw_sha256": {p.name: sha(p) for p in sorted((output / "raw").glob("*"))}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE/"results")
    p.add_argument("--freeze-only", action="store_true")
    args = p.parse_args()
    cfg = json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=cfg["threads"]):
        if args.freeze_only:
            args.output.mkdir(parents=True, exist_ok=True)
            print(json.dumps(freeze(cfg, load_schedules(cfg), args.output), indent=2))
        else:
            run(cfg,args.output)
