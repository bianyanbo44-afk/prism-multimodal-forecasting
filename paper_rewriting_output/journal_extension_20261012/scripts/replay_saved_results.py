"""Offline review of saved PRISM evidence; stdlib and NumPy only.

No original benchmark, encoder, sklearn, Torch or workspace absolute path is
needed. Frozen runner files are included for provenance, but are not imported.
This checks the saved experiment, rather than asserting a public data release.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parents[1]
PACKAGE = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if error > tolerance:
        raise AssertionError(("Numerical replay discrepancy", error, tolerance))
    return error


def entity_means(loss, ids):
    return np.array([loss[ids == e].mean() for e in np.unique(ids)])


def linear(path, x, kind):
    with np.load(path, allow_pickle=False) as z:
        if kind == "calibration":
            return ((x-z["scaler_mean"])/z["scaler_scale"]) @ z["coefficient"] + z["intercept"][0]
        if kind == "construct":
            return ((x-z["scale_mean"])/z["scale_std"]) @ z["coef"] + z["intercept"][0]
        return ((x-z["scaler_mean"])/z["scaler_scale"]) @ z["coef"].T + z["intercept"]


def verify_manifest():
    manifest = read(PACKAGE / "REPLAY_MANIFEST.json")
    for row in manifest["files"]:
        p = PACKAGE / row["path"]
        assert p.is_file(), f"Missing package file: {row['path']}"
        assert p.stat().st_size == row["bytes"], f"Size mismatch: {row['path']}"
        assert sha(p) == row["sha256"], f"SHA-256 mismatch: {row['path']}"
    return {"files_verified": len(manifest["files"]), "all_file_sha256_match": True,
            "manifest_self_hash_excluded": True}


def replay_calibration():
    folder = HERE / "calibration"
    cfg = read(folder / "config.json")
    rows = csv_rows(folder / "results/entity_loss_results.csv")
    lookup = {(int(r["world_seed"]), float(r["strength"]), r["representation"], int(r["entity_id"])): r for r in rows}
    error, checks = 0., 0
    for world in cfg["data_seeds"]:
        with np.load(folder / "results/raw" / f"world_{world}.npz", allow_pickle=False) as raw:
            common, ids = raw["test_common"], raw["test_entity"]
            for strength in cfg["strengths"]:
                y = raw["test_baseline"] + strength*raw["test_oracle"].ravel() + raw["test_noise"]
                for rep in ("oracle", "sentence"):
                    x = np.column_stack([common, raw[f"test_{rep}"]])
                    dx = np.column_stack([common, raw[f"test_{rep}_shuffle"]])
                    zx = np.column_stack([common, np.zeros_like(raw[f"test_{rep}"])])
                    losses = defaultdict(list)
                    for seed in cfg["training_seeds"]:
                        a = folder / "results/models" / f"w{world}_s{strength}_t{seed}_{rep}_aligned.npz"
                        d = folder / "results/models" / f"w{world}_s{strength}_t{seed}_{rep}_retrained_shuffle.npz"
                        t = folder / "results/models" / f"w{world}_s{strength}_t{seed}_timing.npz"
                        pred = {"aligned": linear(a,x,"calibration"), "fixed_shuffle": linear(a,dx,"calibration"),
                                "fixed_zero": linear(a,zx,"calibration"), "retrained_shuffle": linear(d,dx,"calibration"),
                                "timing": linear(t,common,"calibration")}
                        for condition, values in pred.items():
                            losses[condition].append(entity_means((values-y)**2, ids))
                    for condition, values in losses.items():
                        expected = [float(lookup[(world,strength,rep,int(e))][condition]) for e in np.unique(ids)]
                        error = max(error,close(np.mean(values,axis=0),expected))
                        checks += len(expected)
    return {"entity_loss_checks": checks, "max_absolute_error": error, "models_refit": 0}


def replay_construct():
    folder = HERE / "construct_controls"
    cfg = read(folder / "config.json")
    rows = csv_rows(folder / "results/entity_losses.csv")
    lookup = {(int(r["world_seed"]),r["mechanism"],int(r["entity_id"])):r for r in rows}
    error, checks = 0., 0
    for world in cfg["data_seeds"]:
        with np.load(folder / "results/raw" / f"world_{world}.npz", allow_pickle=False) as raw:
            common, ids = raw["test_common"], raw["test_entity"]
            for mechanism in cfg["mechanisms"]:
                field = {"semantic_only":"direction_signal", "proxy_only":"proxy_signal", "timing_only":"timing_signal"}[mechanism]
                y = raw["test_baseline"] + raw[f"test_{field}"] + raw["test_noise"]
                a = folder / "results/models" / f"w{world}_{mechanism}_aligned.npz"
                losses = {}
                for condition in cfg["conditions"]:
                    x = np.column_stack([common,raw[f"test_{condition}"]])
                    losses[condition] = entity_means((linear(a,x,"construct")-y)**2,ids)
                t = folder / "results/models" / f"w{world}_{mechanism}_numeric_timing.npz"
                losses["numeric_timing"] = entity_means((linear(t,common,"construct")-y)**2,ids)
                losses["oracle_readout"] = entity_means(raw["test_noise"]**2,ids)
                for condition, values in losses.items():
                    expected = [float(lookup[(world,mechanism,int(e))][condition]) for e in np.unique(ids)]
                    error = max(error,close(values,expected)); checks += len(expected)
    return {"entity_loss_checks": checks, "max_absolute_error": error, "models_refit":0}


def macro(squared, counts):
    total = counts.sum(0); valid = total > 0
    return float(np.mean(squared.sum(0)[valid]/total[valid]))


def replay_fixed():
    folder = HERE / "fixed_model/results"
    records = read(folder / "manifest.json")["records"]
    grouped, error, reduction_error = defaultdict(dict), 0., 0.
    for r in records:
        with np.load(folder / r["paired_arrays"], allow_pickle=False) as z:
            mse = macro(z["squared"],z["counts"])
            # The frozen runner reduced float32 errors over an entire batch for
            # metrics, and over each window for its saved arrays. Float64 storage
            # does not undo the different first-stage float32 rounding.
            reduction_error = max(reduction_error,close(mse,r["metrics"]["mse"],tolerance=1e-6))
            reduction_error = max(reduction_error,close(macro(z["absolute"],z["counts"]),r["metrics"]["mae"],tolerance=1e-6))
            grouped[(r["dataset"],r["architecture"])][(r["seed"],r["condition"])] = (r["metrics"]["mse"],{k:z[k] for k in z.files})
    summary = read(folder / "paired_summary.json")
    for s in summary:
        group = grouped[(s["dataset"],s["architecture"])]; seeds = s["training_seeds"]
        aa = [group[(seed,"aligned")] for seed in seeds]
        dd = [group[(seed,s["condition"])] for seed in seeds]
        a, d = np.mean([v[0] for v in aa]),np.mean([v[0] for v in dd])
        error = max(error,close(a,s["aligned_mse_mean"]),close(d,s["altered_mse_mean"]),close(d-a,s["delta_mse_mean"]))
        if "fixed_checkpoint_gain_percent_control_denominator" in s:
            error=max(error,close(100*(d-a)/d,s["fixed_checkpoint_gain_percent_control_denominator"]))
        for (_,left),(_,right) in zip(aa,dd):
            for key in ("sample_ids","entity","start_time","counts"):
                np.testing.assert_array_equal(left[key],right[key])
    identities = read(PACKAGE / "CHECKPOINT_PATHS.json")
    assert len(identities)==45
    for r in identities:
        assert sha(PACKAGE/r["package_path"]) == r["checkpoint_sha256"]
    return {"evaluation_metrics_replayed":len(records), "paired_summary_cells_replayed":len(summary),
            "paired_input_identifiers_verified":True,"checkpoint_hashes_verified":len(identities),
            "max_summary_arithmetic_error":error,"max_absolute_float32_reduction_discrepancy":reduction_error,
            "float32_reduction_absolute_tolerance":1e-6,"intervals_refit":False,
            "scope":"Checks per-window aggregates within disclosed float32 batch/window reduction tolerance; verifies summary arithmetic using saved per-seed metrics. Neural prediction reconstruction requires the separately acquired original inputs."}


def replay_external():
    index = read(HERE / "replay_inputs/external_index.json")
    error, observations = 0.,0
    for r in index:
        with np.load(HERE/r["input"],allow_pickle=False) as z:
            pred = z["baseline"]+linear(HERE/r["checkpoint"],z["x"],"external")
            squared = (pred-z["target"])**2
            error=max(error,close(pred[z["mask"]],z["csv_prediction"]),close(z["target"][z["mask"]],z["csv_target"]))
            error=max(error,close(squared[z["mask"]].mean(),r["expected_mse"]))
            observations+=int(z["mask"].sum())
    csv_checks=0
    for name in ("EPA-Air","FNSPID"):
        folder=HERE/"external_validation/results"/name
        records=read(folder/"results.json")["results"]
        for r in records:
            if r["representation"]!="none":continue
            rows=csv_rows(folder/(r["training_condition"]+".csv"))
            squared=np.array([(float(v["prediction"])-float(v["target"]))**2 for v in rows])
            error=max(error,close(squared,[float(v["squared_error"]) for v in rows]),close(squared.mean(),r["test"]["mse"]))
            assert len(rows)==r["test"]["observations"]
            csv_checks+=1
    return {"saved_model_conditions_replayed":len(index),"additional_baseline_csv_conditions_recomputed":csv_checks,"scalar_prediction_checks":observations,
            "max_absolute_error":error,"models_refit":0}


def replay_repeats():
    index=read(HERE / "replay_inputs/repeat_index.json")
    lookup={(r["dataset"],r["representation"],int(r["base_seed"])):r for r in csv_rows(HERE / "intervention_repeats/results/map_aggregate.csv")}
    error, checks = 0.,0
    grouped=defaultdict(list)
    for r in index:
        with np.load(HERE/r["aligned_input"],allow_pickle=False) as z, np.load(HERE/r["deltas"],allow_pickle=False) as delta, np.load(HERE/r["checkpoint"],allow_pickle=False) as state:
            ap=z["baseline"]+linear(HERE/r["checkpoint"],z["x"],"external")
            start,stop=r["content_start"],r["content_stop"]
            for j,seed in enumerate(delta["seeds"]):
                pred=ap+((delta["component_delta"][j]/state["scaler_scale"][start:stop])@state["coef"][:,start:stop].T)[delta["repeat_index"]]
                loss=((pred-z["target"])**2)[z["mask"]].mean()
                row=lookup[(r["dataset"],r["representation"],int(seed))]
                error=max(error,close(loss,float(row["map_mse"])))
                grouped[(r["dataset"],r["representation"])].append(float(loss))
                checks+=1
    summary=csv_rows(HERE / "intervention_repeats/results/between_map_summary.csv")
    for r in summary:
        values=np.array(grouped[(r["dataset"],r["representation"])])-float(r["aligned_mse"])
        for name, val in (("mean_loss_difference",values.mean()),("empirical_q025_loss_difference",np.quantile(values,.025)),("empirical_q975_loss_difference",np.quantile(values,.975))):
            error=max(error,close(val,float(r[name])))
    return {"map_effects_replayed":checks,"between_map_summary_cells_replayed":len(summary),
            "max_absolute_error":error,"models_refit":0,"scope":"Empirical variation across saved cyclic maps, not sampling confidence intervals."}


def end_to_end_example():
    """Refit one stipulated generated mechanism using only packaged arrays."""
    folder=HERE/"construct_controls";cfg=read(folder/"config.json");world=cfg["data_seeds"][0]
    with np.load(folder/"results/raw"/f"world_{world}.npz",allow_pickle=False) as raw:
        x={s:np.column_stack([raw[f"{s}_common"],raw[f"{s}_aligned"]]) for s in ("train","val","test")}
        y={s:raw[f"{s}_baseline"]+raw[f"{s}_direction_signal"]+raw[f"{s}_noise"] for s in x}
        mean,scale=x["train"].mean(0),x["train"].std(0);scale[scale==0]=1
        tx=(x["train"]-mean)/scale;offset=tx.mean(0);tx=tx-offset;ym=y["train"].mean()
        candidates=[]
        for alpha in cfg["ridge_alphas"]:
            coef=np.linalg.solve(tx.T@tx+alpha*np.eye(tx.shape[1]),tx.T@(y["train"]-ym))
            intercept=ym-offset@coef
            vp=((x["val"]-mean)/scale)@coef+intercept
            score=entity_means((vp-y["val"])**2,raw["val_entity"]).mean()
            candidates.append((score,alpha,coef,intercept))
        _,alpha,coef,intercept=min(candidates,key=lambda z:z[0])
        prediction=((x["test"]-mean)/scale)@coef+intercept
        expected=linear(folder/"results/models"/f"w{world}_semantic_only_aligned.npz",x["test"],"construct")
        discrepancy=close(prediction,expected,tolerance=1e-9)
        selection=next(r for r in read(folder/"results/model_selection.json") if r["world_seed"]==world and r["mechanism"]=="semantic_only" and r["model"]=="aligned_text")
        assert alpha==selection["selected_alpha"]
        return {"world_seed":world,"mechanism":"semantic_only","training_rows":len(tx),
                "validation_rows":len(x["val"]),"test_rows":len(prediction),"candidate_fits":len(candidates),
                "selected_alpha_matches":True,"max_absolute_prediction_difference":discrepancy,
                "test_entity_MSE":float(entity_means((prediction-y["test"])**2,raw["test_entity"]).mean()),
                "scope":"Train/validation-selected/test evaluation from a saved generated world; does not regenerate the empirical event clocks or encoder."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example",action="store_true",help="Also run one genuine NumPy train/select/test example.")
    parser.add_argument("--output",type=Path,default=PACKAGE/"REPLAY_RESULT.json")
    args=parser.parse_args();started=time.perf_counter()
    result={"manifest":verify_manifest()}
    for name,function in (("calibration",replay_calibration),("construct_controls",replay_construct),
                          ("fixed_model",replay_fixed),("external_validation",replay_external),("intervention_repeats",replay_repeats)):
        result[name]=function();print(name,json.dumps(result[name]),flush=True)
    if args.example:
        result["end_to_end_example"]=end_to_end_example();print("end_to_end_example",json.dumps(result["end_to_end_example"]),flush=True)
    result.update({"all_checks_passed":True,"runtime":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform()},"seconds":time.perf_counter()-started})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("All offline replay checks passed.",flush=True)


if __name__=="__main__":
    main()
