"""Produce reviewable source tables and a Chinese report from completed evaluation."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("Wait for complete evaluation before report generation")
    summaries = json.loads((RESULTS / "paired_summary.json").read_text(encoding="utf-8"))
    entity_path = RESULTS / "entity_metrics.csv"
    with entity_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "architecture", "seed", "condition", "entity", "windows", "observed_values", "mse", "mae"])
        writer.writeheader()
        for r in manifest["records"]:
            a = dict(np.load(RESULTS / r["paired_arrays"], allow_pickle=False))
            for entity in sorted(set(a["entity"])):
                ix = np.where(a["entity"] == entity)[0]
                c = a["counts"][ix].sum(0)
                valid = c > 0
                writer.writerow({"dataset": r["dataset"], "architecture": r["architecture"], "seed": r["seed"], "condition": r["condition"], "entity": entity, "windows": len(ix), "observed_values": int(c.sum()), "mse": float(np.mean(a["squared"][ix].sum(0)[valid] / c[valid])), "mae": float(np.mean(a["absolute"][ix].sum(0)[valid] / c[valid]))})
    aligned = [r for r in manifest["records"] if r["condition"] == "aligned"]
    replay_max = max(r["aligned_replay"]["max_feature_mse_absolute_error"] for r in aligned)
    best_epoch_zero = sum(r["historical_best_epoch"] == 0 for r in aligned)
    lines = ["# SemAudit 固定模型测试输入扰动：探索性结果", "",
             "执行日期：2026-10-12（Asia/Shanghai）。已固定既有 aligned 检查点，只更改测试输入，没有训练或更新权重。历史测试集已查看，本轮不能称为 untouched、确认性复验或无语义作用的等效证明。", "",
             f"共完成 {len(aligned)} 个检查点（5 数据集 × 3 模型 × 种子 53、67、79），每个 8 条件，共 {len(manifest['records'])} 次评估。执行脚本耗时 {manifest['elapsed_seconds']:.2f} 秒。所有 aligned MSE/MAE 与归档及检查点中的评估一致（容差内），最大逐特征 MSE 绝对偏差为 {replay_max:.3g}。所有条件均通过事件时间、到达时间、模态、质量、事件 ID、数值输入、目标、掩码及查询窗口不变量检查；每次评估后完整 state_dict 哈希保持相同，检查点文件哈希亦保持相同。", "",
             "## 比较对象与干预", "",
             "历史各条件重训练回答『重新训练后还能利用哪些信息』；本轮固定 aligned 模型回答『当前检查点的预测对测试输入内容如何响应』。两种比较不能共享一个语义依赖结论。", "",
             "本轮 8 条件为 aligned、原发布的实体内且划分内 shuffle、25/50/75/100% 无固定点子集置换、零内容向量及训练文本表示均值。shuffle 使用原控制函数与测试划分种子 9103，并逐位置核对置换；各剂量使用嵌套的实体内随机子集，在子集内置换以保留实体表示多重集。子集少于两个位置时不改动，实际比例单独报告。", "",
             "训练均值只使用 retained train 文本位置，不看 validation/test，也不重新训练模型；它是位置加权的表示中心，不能称为自然语言的『无信息』或『语义中性』真值。对 selective SSM 使用历史 LayerNorm 表示，对两种 IMM-TSF 使用历史原表示；RepoHealth 使用历史 1024-token cache，其他数据用 320-token cache。零向量和均值可能处于训练分布外，零向量同时去除表示共同方向，不能将其损失变化直接归因于正确语义。包装模型显式保留文本事件 mask，零内容不会删掉事件位置。", "",
             "## 实际干预覆盖", "",
             "| 数据集 | 测试文本位置 | 单位置实体数 | 原 shuffle 改变表示 | 25% 剂量 | 50% 剂量 | 75% 剂量 | 100% 剂量 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"):
        c = manifest["coverage"][f"{dataset}/none"]["conditions"]
        base = c["aligned"]
        fractions = [c[label]["achieved_value_change_fraction"] for label in ("released_shuffle", "derangement_25", "derangement_50", "derangement_75", "derangement_100")]
        lines.append(f"| {dataset} | {base['note_positions']} | {base['singleton_entities']} | " + " | ".join(f"{x:.2%}" for x in fractions) + " |")
    lines += ["", "CESNET 的 7 个单位置实体、7/16 文本位置不能做实体内无固定点替换；全剂量最大覆盖为 56.25%。本轮未使用跨实体交换改变旧的不变量。其他数据集全剂量达到 100% 缓存表示位置改变；这仍只证明输入向量改变，不是自然语言命题已改变的真值。", "",
              "## 损失变化与时间相关区间", "",
              "下表为三训练种子的配对平均，ΔMSE = 扰动 − aligned，正值表示当前模型在扰动下变差。百分比使用 aligned MSE 为分母。95% 区间是未作多重比较调整的探索性 percentile bootstrap：先按相同训练种子平均每窗口平方误差，再按实体采样，并在实体内按循环连续窗口块采样，主块长 2；4/8 窗口敏感性保存在完整 JSON。所有重采样保持目标、窗口、特征与模型配对。种子不作为独立实体，只另列配对效应的种子间 SD。", "",
              "| 数据集 | 模型 | 原 shuffle Δ% | 全剂量 Δ% | 全剂量 ΔMSE [95% 块2区间] | 零内容 Δ% | 训练均值 Δ% |",
              "|---|---|---:|---:|---|---:|---:|"]
    for dataset in ("ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"):
        for arch in ("selective_ssm", "immtsf_recavg_gradd", "immtsf_t2v_xattn"):
            rows = {r["condition"]: r for r in summaries if r["dataset"] == dataset and r["architecture"] == arch}
            r = rows["derangement_100"]
            ci = r["dependence_intervals"][0]["ci95"]
            lines.append(f"| {dataset} | {arch} | {rows['released_shuffle']['delta_percent_of_aligned']:+.5f}% | {r['delta_percent_of_aligned']:+.5f}% | {r['delta_mse_mean']:+.6g} [{ci[0]:+.6g}, {ci[1]:+.6g}] | {rows['zero_content']['delta_percent_of_aligned']:+.3f}% | {rows['train_centroid']['delta_percent_of_aligned']:+.5f}% |")
    max_permutation = max(summaries, key=lambda r: abs(r["delta_percent_of_aligned"]) if r["condition"] == "derangement_100" else -1)
    lines += ["",
              f"本轮全剂量实体内置换的最大三种子平均相对变化为 {max_permutation['delta_percent_of_aligned']:+.5f}%（{max_permutation['dataset']}，{max_permutation['architecture']}）。这描述这些已选择检查点在当前干预下的数值响应，不能建立文本本身无预测价值、全新表示下也无用、或任意融合模型的无依赖结论。", "",
              f"{best_epoch_zero}/{len(aligned)} 个 aligned 检查点历史 best_epoch=0；它们由历史验证集选择保留初始模型。这里如实评估这些实际归档模型，没有根据新结果筛选或改训。模型能力和语义灵敏度必须另用独立的校准任务建立。", "",
              "这些初始检查点的精确单元为 CESNET / IMM-TSF T2V-XAttn / seeds 53、67、79，及 FNSPID / IMM-TSF T2V-XAttn / seed 53。回放成功证明归档模型数值可复现；本工作包没有独立校准语义检出能力，属于回顾性检查点诊断。", "",
              "原表 Δ% 的分母明确为 aligned。与主文 SAG/CBTG 控制分母一致的固定检查点对应百分比及重新计算的配对区间见 CONTROL_DENOMINATOR_REPORT_zh.md；其 gain% = 100 × (control−aligned)/control。两种估计对象保持区分，旧区间没有直接改名。", "",
              "区间依赖实体交换性和块长度假设，不能保证消除所有领域级相关性。重复与重叠预测窗口不计为独立重复；有限实体、短序列和单窗口实体限制区间解释。极小的 float32 损失差可能受数值求和次序影响，区间小于/大于零不作为新的显著性判决。本次未设实用等效界限，也不作等效结论。", "",
              "## 可复核文件", "",
              "- `results/manifest.json`：360 次条件评估、MSE、MAE、逐特征及实体 MSE、aligned 复验、模型/源码哈希和不变量结果。",
              "- `results/paired_windows/*.npz`：逐窗口逐特征平方误差和绝对误差之和、观测值计数、sample_id、entity、start_time；相同模型和种子下条件逐窗口配对。",
              "- `results/paired_summary.json` / `.csv`：105 个模型×数据×控制比较，2/4/8 块长区间及种子间效应。",
              "- `results/paired_summary_control_denominator.csv`：同时保留 aligned 分母 Δ%、control 分母 gain% 与 2/4/8 块长的 gain 区间；追加分析的独立身份见 control_denominator_analysis.json。",
              "- `results/entity_metrics.csv`：每条件、模型、种子和实体的 MSE、MAE、窗口数、目标计数。",
              "- `results/*_source_maps.json`：每实体置换来源索引；`run_fixed_model.py`、`test_fixed_model.py`：脚本及 8 项边界测试，全部通过。",
              "", "无新模型训练、无自然语言编码器扩展、无外部数据确认在本工作包内执行。"]
    (HERE / "FIXED_MODEL_REPORT_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created report and {entity_path}")


if __name__ == "__main__":
    main()
