"""Measure input intervention strength; this does not measure forecasting quality.

Reconstructs the released within-entity permutation and validates it against
the original control implementation. A new derangement changes source indices
where possible, but identical representations can still leave content unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def make_permutation(n: int, rng: random.Random, *, derange: bool) -> list[int]:
    if n < 0:
        raise ValueError("n must be nonnegative")
    permutation = list(range(n))
    if derange and n < 2:
        return permutation
    for _ in range(10000 if derange else 1):
        rng.shuffle(permutation)
        if not derange or all(i != source for i, source in enumerate(permutation)):
            return permutation
    raise RuntimeError("Unable to draw a derangement")


def vector_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pair_diagnostics(values: list[torch.Tensor], times: list[float],
                     permutation: list[int]) -> dict:
    n = len(values)
    if len(times) != n or sorted(permutation) != list(range(n)):
        raise ValueError("Every source must appear exactly once")
    digests = [vector_digest(value) for value in values]
    fixed = sum(i == source for i, source in enumerate(permutation))
    unchanged = sum(digests[i] == digests[source] for i, source in enumerate(permutation))
    cosines = []
    for start in range(0, n, 1024):
        indices = list(range(start, min(start + 1024, n)))
        original = torch.stack([values[i] for i in indices]).to(dtype=torch.float64)
        replaced = torch.stack([values[permutation[i]] for i in indices]).to(dtype=torch.float64)
        if not torch.isfinite(original).all() or not torch.isfinite(replaced).all():
            raise ValueError("Nonfinite representation")
        original_norm = torch.linalg.vector_norm(original, dim=1)
        replaced_norm = torch.linalg.vector_norm(replaced, dim=1)
        valid = (original_norm > 0) & (replaced_norm > 0)
        similarity = (original[valid] * replaced[valid]).sum(dim=1)
        similarity = similarity / (original_norm[valid] * replaced_norm[valid])
        cosines.extend(similarity.clamp(-1, 1).tolist())
    return {
        "positions": n,
        "fixed_positions": fixed,
        "identical_representation_positions": unchanged,
        "duplicate_representation_excess": n - len(set(digests)),
        "cosines": cosines,
        "absolute_time_displacements": [abs(times[i] - times[source])
                                         for i, source in enumerate(permutation)],
    }


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {"mean": float(np.mean(values)), "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def audit_split(windows, window_info, text_modality, seed: int, *, derange: bool,
                official_shuffle) -> dict:
    groups = defaultdict(list)
    for window_index, window in enumerate(windows):
        info = window_info[window.sample_id]
        for event_index, event in enumerate(window.events):
            if event.modality == text_modality:
                groups[info.entity].append((window_index, event_index, event.value,
                                           info.start_time + event.event_time))
    # The released control initializes one RNG per split and processes sorted entities.
    rng = random.Random(seed)
    official = None if derange else official_shuffle(
        windows, window_info, text_modality, seed=seed)
    totals = {"positions": 0, "fixed_positions": 0,
              "identical_representation_positions": 0,
              "duplicate_representation_excess": 0}
    cosines, displacements = [], []
    for entity in sorted(groups):
        entries = groups[entity]
        permutation = make_permutation(len(entries), rng, derange=derange)
        diagnostics = pair_diagnostics([entry[2] for entry in entries],
                                       [entry[3] for entry in entries], permutation)
        for key in totals:
            totals[key] += diagnostics[key]
        cosines.extend(diagnostics["cosines"])
        displacements.extend(diagnostics["absolute_time_displacements"])
        if official is not None:
            for entry, source in zip(entries, permutation, strict=True):
                actual = official[entry[0]].events[entry[1]].value
                if not torch.equal(actual, entries[source][2]):
                    raise AssertionError("Permutation reconstruction differs from released control")
    if official is not None:
        for before, after in zip(windows, official, strict=True):
            if (before.sample_id, before.cutoff) != (after.sample_id, after.cutoff):
                raise AssertionError("Forecast unit changed")
            for attr in ("targets", "target_masks", "query_horizons"):
                if not torch.equal(getattr(before, attr), getattr(after, attr)):
                    raise AssertionError(f"Changed {attr}")
            for a, b in zip(before.events, after.events, strict=True):
                if (a.event_time, a.arrival_time, a.modality, a.event_id, a.quality) != (
                        b.event_time, b.arrival_time, b.modality, b.event_id, b.quality):
                    raise AssertionError("Event process changed")
                if a.modality != text_modality and not torch.equal(a.value, b.value):
                    raise AssertionError("Numerical input changed")
    n = totals["positions"]
    return {
        **totals, "windows": len(windows), "entities_with_text": len(groups),
        "singleton_entities": sum(len(entries) == 1 for entries in groups.values()),
        "fixed_position_fraction": totals["fixed_positions"] / n if n else None,
        "identical_representation_fraction": totals["identical_representation_positions"] / n if n else None,
        "duplicate_representation_excess_fraction": totals["duplicate_representation_excess"] / n if n else None,
        "cosine": quantiles(cosines), "undefined_cosines": n - len(cosines),
        "absolute_time_displacement": quantiles(displacements),
        "released_control_reconstruction_verified": official is not None,
        "representation_multiset_preserved_per_entity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "phase3_prototype")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"])
    parser.add_argument("--seed", type=int, default=9101)
    args = parser.parse_args()
    root = args.prototype_root.resolve()
    sys.path.insert(0, str(root))
    from eventmamba.timeimm import DATASET_SPECS, GPT2_TEXT_CACHE, GPT2_TEXT_CACHE_1024, load_timeimm
    from eventmamba.multimodal import shuffle_text_values_within_entities
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running", "analysis": "exploratory input intervention diagnostics",
        "seed": args.seed, "normalization": "train_only", "text_preprocessing": "none",
        "counting_unit": "retained text-event position; not an independent forecasting replicate",
        "interpretation_boundary": "Cosine is representation similarity, not semantic equivalence; no forecasting losses are measured.",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [root / "eventmamba/timeimm.py", root / "eventmamba/multimodal.py"]},
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [], "failures": [],
    }
    result_file = output / "control_strength.json"
    def save():
        result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    for dataset in args.datasets:
        try:
            spec = DATASET_SPECS[dataset]
            cache = GPT2_TEXT_CACHE_1024 if dataset == "RepoHealth" else GPT2_TEXT_CACHE
            data = load_timeimm(root / "external/Time-IMM/data" / dataset, spec,
                               normalization="train_only", text_cache_name=cache,
                               text_preprocessing="none")
            if data.text_modality is None:
                raise ValueError("No text modality")
            for split_index, split in enumerate(("train", "val", "test")):
                windows = getattr(data, f"{split}_windows")
                for derange in (False, True):
                    row = audit_split(windows, data.window_info, data.text_modality,
                                      args.seed + split_index, derange=derange,
                                      official_shuffle=shuffle_text_values_within_entities)
                    row.update(dataset=dataset, split=split,
                               condition="derangement" if derange else "released_shuffle",
                               cache=cache, time_displacement_unit=spec.time_unit)
                    payload["rows"].append(row)
                    save()
            print(json.dumps({"completed": dataset, "rows": len(payload["rows"])}), flush=True)
            del data
        except Exception as exc:
            payload["failures"].append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(payload["failures"][-1]), flush=True)
            save()
    payload["status"] = "complete" if not payload["failures"] else "incomplete"
    save()
    lines = ["# SemAudit 输入干预强度诊断", "", "本表来自历史数据的新增探索性诊断，未训练模型，也未测量预测损失。",
             "计数单位是保留窗口中的文本事件位置，不能视为独立样本数。数值相同指缓存表示完全相同，不等同于原文重复。",
             "余弦高低只描述当前表示的几何距离，不能单独证明语义是否被打乱。", "",
             "| 数据集（测试集） | 对照 | 文本位置 | 原位置比例 | 表示完全不变比例 | 余弦中位数 | 时间位移中位数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        if row["split"] == "test" and row["positions"]:
            median = row["cosine"]["p50"]
            median_text = f"{median:.4f}" if median is not None else "NA"
            lines.append(f'| {row["dataset"]} | {row["condition"]} | {row["positions"]} | '
                         f'{row["fixed_position_fraction"]:.2%} | {row["identical_representation_fraction"]:.2%} | '
                         f'{median_text} | {row["absolute_time_displacement"]["p50"]:.4f} {row["time_displacement_unit"]} |')
    lines += ["", "完整 train/val/test 结果、源代码哈希及失败记录见 control_strength.json。",
              "历史随机置换经过原控制函数核对，并检查事件过程、数值输入、目标及窗口一致性。",
              "无固定点置换只在至少两个文本位置的实体中可行；即使没有固定点，重复表示仍可导致输入内容不变。",
              f'本轮状态：{payload["status"]}。']
    (output / "CONTROL_STRENGTH_REPORT_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
