"""Append fixed-checkpoint gains and paired intervals with control denominator.

Keeps the completed inference runner and its original hash unchanged. The
retrospective inference input/output arrays remain the sole numeric evidence.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def joint_control_bootstrap(aligned, controls, *, block_length, repetitions=2000, seed=194):
    labels = sorted(controls)
    control_squared = np.stack([controls[label]["squared"] for label in labels])
    for label in labels:
        if not np.array_equal(aligned["sample_ids"], controls[label]["sample_ids"]):
            raise AssertionError("Sample pairing differs")
        if not np.array_equal(aligned["counts"], controls[label]["counts"]):
            raise AssertionError("Observation counts differ")
    groups = [np.where(aligned["entity"] == e)[0] for e in sorted(set(aligned["entity"]))]
    groups = [g[np.argsort(aligned["start_time"][g], kind="stable")] for g in groups]
    rng = np.random.default_rng(seed)
    values = np.empty((repetitions, len(labels)))
    for b in range(repetitions):
        sampled = []
        for entity_index in rng.integers(0, len(groups), size=len(groups)):
            g = groups[entity_index]
            n, length = len(g), min(block_length, len(g))
            starts = rng.integers(0, n, size=math.ceil(n / length))
            offsets = (starts[:, None] + np.arange(length)[None]) % n
            sampled.extend(g[offsets.ravel()[:n]])
        ix = np.asarray(sampled, dtype=np.int64)
        counts = aligned["counts"][ix].sum(0)
        valid = counts > 0
        a_loss = np.mean(aligned["squared"][ix].sum(0)[valid] / counts[valid])
        c_loss = np.mean(control_squared[:, ix].sum(1)[:, valid] / counts[None, valid], axis=1)
        if np.any(c_loss <= 0):
            raise ValueError("Nonpositive control loss: percentage gain undefined")
        values[b] = 100 * (c_loss - a_loss) / c_loss
    ci = np.quantile(values, [0.025, 0.975], axis=0)
    return {label: [float(ci[0, i]), float(ci[1, i])] for i, label in enumerate(labels)}


def main():
    start = time.perf_counter()
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((RESULTS / "paired_summary.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for r in manifest["records"]:
        grouped[r["dataset"], r["architecture"]].append(r)
    for (dataset, architecture), records in grouped.items():
        condition_arrays = {}
        for condition in sorted(set(r["condition"] for r in records)):
            records_for_condition = sorted([r for r in records if r["condition"] == condition], key=lambda r: r["seed"])
            arrays = [dict(np.load(RESULTS / r["paired_arrays"], allow_pickle=False)) for r in records_for_condition]
            averaged = dict(arrays[0])
            averaged["squared"] = np.mean([a["squared"] for a in arrays], axis=0)
            condition_arrays[condition] = averaged
        controls = {label: array for label, array in condition_arrays.items() if label != "aligned"}
        by_length = {length: joint_control_bootstrap(condition_arrays["aligned"], controls, block_length=length)
                     for length in (2, 4, 8)}
        for r in rows:
            if r["dataset"] != dataset or r["architecture"] != architecture:
                continue
            r["fixed_checkpoint_gain_percent_control_denominator"] = 100 * r["delta_mse_mean"] / r["altered_mse_mean"]
            r["fixed_checkpoint_gain_definition"] = "100 * (control MSE - aligned MSE) / control MSE; fixed aligned checkpoint, test-only perturbation"
            for interval in r["dependence_intervals"]:
                interval["fixed_checkpoint_gain_percent_control_denominator_ci95"] = by_length[interval["block_length_windows"]][r["condition"]]
            # SAG and CBTG labels here explicitly refer to fixed-checkpoint
            # analogues, not the historical condition-retraining estimands.
            r["fixed_checkpoint_sag_percent"] = (r["fixed_checkpoint_gain_percent_control_denominator"]
                                                  if r["condition"] == "released_shuffle" or r["condition"].startswith("derangement_") else None)
            r["fixed_checkpoint_cbtg_percent"] = (r["fixed_checkpoint_gain_percent_control_denominator"]
                                                   if r["condition"] == "zero_content" else None)
        print(f"Appended control-denominator paired intervals: {dataset}/{architecture}", flush=True)
    (RESULTS / "paired_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (RESULTS / "paired_summary_control_denominator.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = ["dataset", "architecture", "condition", "aligned_mse", "control_mse", "delta_percent_aligned_denominator", "gain_percent_control_denominator", "gain_control_ci95_block2_low", "gain_control_ci95_block2_high", "gain_control_ci95_block4_low", "gain_control_ci95_block4_high", "gain_control_ci95_block8_low", "gain_control_ci95_block8_high"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {"dataset": r["dataset"], "architecture": r["architecture"], "condition": r["condition"], "aligned_mse": r["aligned_mse_mean"], "control_mse": r["altered_mse_mean"], "delta_percent_aligned_denominator": r["delta_percent_of_aligned"], "gain_percent_control_denominator": r["fixed_checkpoint_gain_percent_control_denominator"]}
            for interval in r["dependence_intervals"]:
                length = interval["block_length_windows"]
                lo, hi = interval["fixed_checkpoint_gain_percent_control_denominator_ci95"]
                out[f"gain_control_ci95_block{length}_low"] = lo
                out[f"gain_control_ci95_block{length}_high"] = hi
            writer.writerow(out)
    details = {"analysis": "control-denominator percentage gain supplement to fixed-checkpoint inference",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_manifest_sha256": hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest(),
               "elapsed_seconds": time.perf_counter() - start, "comparisons": len(rows),
               "repetitions": 2000, "bootstrap_seed": 194, "block_lengths": [2, 4, 8]}
    (RESULTS / "control_denominator_analysis.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    lines = ["# SemAudit 固定检查点百分比：分母一致的追加分析", "",
             "原报告 Δ% = 100 × (control−aligned)/aligned，分母为 aligned。本文追加固定检查点的 SAG/CBTG 数值对应式：gain% = 100 × (control−aligned)/control，分母为当前对照。旧区间没有改名；以下区间逐次从同一配对实体/时间块 bootstrap 的 aligned 和 control 损失计算 gain%，与该分母一致。", "",
             "本分析仍是历史数据、历史验证集所选择检查点的回顾性诊断。回放成功建立数值复现，不建立模型对已知语义的检出能力；本工作包没有开展独立能力校准。不能把这些模型的近零置换响应推广为所有模型或文本本身没有作用。", "",
             "历史 best_epoch=0 的 4/45 个单元精确为：CESNET / IMM-TSF T2V-XAttn / seeds 53、67、79；FNSPID / IMM-TSF T2V-XAttn / seed 53。其余模型身份及 best_epoch 在 manifest 中逐项保留。", "",
             "| 数据集 | 模型 | 全剂量 fixed SAG%（control 分母） | 95% gain 区间，块2 | 零内容 fixed CBTG%（control 分母） |",
             "|---|---|---:|---|---:|"]
    for dataset in ("ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"):
        for architecture in ("selective_ssm", "immtsf_recavg_gradd", "immtsf_t2v_xattn"):
            cells = {r["condition"]: r for r in rows if r["dataset"] == dataset and r["architecture"] == architecture}
            r = cells["derangement_100"]
            lo, hi = r["dependence_intervals"][0]["fixed_checkpoint_gain_percent_control_denominator_ci95"]
            lines.append(f"| {dataset} | {architecture} | {r['fixed_checkpoint_sag_percent']:+.6f}% | [{lo:+.6f}%, {hi:+.6f}%] | {cells['zero_content']['fixed_checkpoint_cbtg_percent']:+.3f}% |")
    lines += ["", "所有 105 个比较的 aligned 分母 Δ%、control 分母 gain% 与 2/4/8 块长区间见 paired_summary_control_denominator.csv 及扩充后的 paired_summary.json。正式推理 runner、其 SHA 以及原始 manifest 完全保留；追加分析有独立脚本哈希、输入 manifest 哈希和计时。"]
    (HERE / "CONTROL_DENOMINATOR_REPORT_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    main()
