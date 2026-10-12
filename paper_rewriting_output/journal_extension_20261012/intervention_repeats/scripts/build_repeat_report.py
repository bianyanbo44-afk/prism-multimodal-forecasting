import csv,json,hashlib
from pathlib import Path
import numpy as np
from run_repeats import ROOT,EXT

def main():
    result=json.loads((ROOT/'results'/'results.json').read_text(encoding='utf8'))
    summaries=result['summaries'];units=result['unit_summaries']
    lines=['# SemAudit 固定模型重复置换：映射敏感性','',
      '在原有固定 aligned ridge 模型上，增加 20 个预先保存的独立 RNG 基础种子（120001–120020），每个实体/切分内产生非零循环位移。该设计来自已有候选槽位，保持 recipient 文本时间、数值输入、事件结构、目标和检查点；没有重新训练、重新选择超参数或重新拟合 PCA。FNSPID/EPA-Air 每个数据集的同一 20 个映射在三种表示间复用。Fidel 的同一映射在九个站点间共享。','',
      '**下列 2.5–97.5% 范围是 20 个循环置换映射之间的经验分位数范围，不是采样置信区间、置换检验的零分布或训练模型的不确定性。** 循环位移族也不等同于全部可能置换的均匀采样。历史测试结果已经查看，所有比较都是探索性的条件敏感性检查。','',
      '| 数据 | 表示 | Aligned MSE | 平均置换 MSE | 平均 loss(map)−loss(aligned) | 映射间经验 2.5–97.5% 范围 | 正差映射数 |','|---|---|---:|---:|---:|---|---:|']
    for r in summaries:
        lines.append(f"| {r['dataset']} | {r['representation']} | {r['aligned_mse']:.8f} | {r['mean_map_mse']:.8f} | {r['mean_loss_difference']:+.8f} | [{r['empirical_q025_loss_difference']:+.8f}, {r['empirical_q975_loss_difference']:+.8f}] | {r['positive_map_count']}/{r['map_draws']} |")
    lines.extend(['','FNSPID 的 GPT-2 在 20 个映射中损失差均为负；MiniLM 和 FinBERT 各有 3 个映射为正。EPA-Air 三种表示的经验分位数范围都跨越零，正差映射数分别为 2、6、8。单个置换图上的方向不能被当成稳健的一般语义结论。原先 FNSPID 的验证能力不足、EPA 测试泛化失败和 latest-four 线性表示截取限制继续适用。','',
      '## Fidel：规则外部任务的跨映射点效应',''])
    fidel=next(r for r in summaries if r['dataset']=='Fidel');plantrows=[r for r in units if r['dataset']=='Fidel']
    lines.extend([f"Fidel 的 20 个映射全部增加固定模型损失；平均损失差 {fidel['mean_loss_difference']:.8f}，经验分位数范围 [{fidel['empirical_q025_loss_difference']:.8f}, {fidel['empirical_q975_loss_difference']:.8f}]，按每次映射损失归一化后，对齐相对收益平均 {fidel['mean_relative_aligned_gain']:.2%}。九个站点的逐映射效果另列下表。它们共享 Calgary 天气流，不能视为九个独立天气实验。",'',
      '| 站点 | 平均损失差 | 映射间经验 2.5–97.5% 范围 | 正差映射数 |','|---|---:|---|---:|'])
    for r in plantrows:lines.append(f"| {r['unit']} | {r['mean_loss_difference']:.8f} | [{r['empirical_q025_loss_difference']:.8f}, {r['empirical_q975_loss_difference']:.8f}] | {r['positive_map_count']}/{r['map_draws']} |")
    lines.extend(['','## 实际扰动剂量和时间语义','',
      '| 数据/表示 | 选择 recipient 槽位 | 可置换槽位 | 改变向量槽位范围 | 改变池化窗口范围 | donor 时间晚于 recipient cutoff 的比例范围 |','|---|---:|---:|---|---|---|'])
    for r in summaries:
        lines.append(f"| {r['dataset']}/{r['representation']} | {r['selected_recipient_slots']} | {r['eligible_recipient_slots']} | {r['min_changed_vector_slots']}–{r['max_changed_vector_slots']} | {r['min_changed_pooled_windows']}–{r['max_changed_pooled_windows']} | {r['min_donor_after_cutoff_fraction']:.2%}–{r['max_donor_after_cutoff_fraction']:.2%} |")
    unique_maps={}
    for dataset in ('FNSPID','EPA-Air','Fidel'):
        maps=json.loads((ROOT/'results'/f'{dataset}_map_shifts.json').read_text(encoding='utf8'))
        unique_maps[dataset]=len({json.dumps(v,sort_keys=True) for v in maps.values()})
    lines.extend(['',f"三个任务各自保存的完整映射均有 {unique_maps['FNSPID']}/{unique_maps['EPA-Air']}/{unique_maps['Fidel']} 个不同配置。槽位指实际模型使用的 recipient 输入位置；Fidel 的 3,285 个 plant/day 位置包含共享的 365 个时间戳，候选 donor 来自测试切分内的完整报告时间序列，而非独立站点事件。",'',
      '所有测试 recipient 位置在这里都可做非固定点替换；20 个映射均改变全部选中的向量以及全部池化窗口。原文改变计数在 Time-IMM 的逐映射记录中单独报告，Fidel 原始天气文字不可用，不假定原文命题改变已核验。','',
      'Recipient 时间保持原始历史槽位。Donor 内容可以来自同一测试切分中的未来日期；Fidel donor 晚于 recipient cutoff 的比例跨映射约 1.10%–98.90%，完整时间对照保存在 `map_slot_timing.csv`。这是离线内容错位干预，不能解释为实际可用的替代预测输入。Fidel 原始表示时间戳本身也未被证明是发布/到达时间，所以结果不能升级为实时自然事件正对照。','',
      '## 不变量与保存范围','',
      f"共执行 {len(summaries)*20} 个 cell/map 对比，导出 1,260 条 entity/plant 映射效应和 94,000 条 donor/recipient 时间记录。七个固定模型的 aligned 重放损失均复现已有结果；target/mask 与标识的散列、数值和非内容特征、检查点文件散列通过核验。预测用保存的标准化与线性系数直接计算，没有模型更新。源脚本和旧 checkpoint 的散列在操作前后相同。",'',
      '本实验只量化七个既有独立 ridge 检验器对循环错位映射的敏感性，没有重复 15 个神经模型的置换图，也没有给那些模型新增校准或置信区间。','',
      '所有权重保持条件、任务边界、种子和分位数规则见 `protocol.json`；其散列、执行脚本散列和旧输入散列在新评分前保存于 `FROZEN_PROTOCOL_MANIFEST.json`。验证测试和便携复现依赖见 `REPRODUCTION.md`。'])
    (ROOT/'REPEAT_MAP_REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    table=[]
    for r in summaries:table.append(f"{r['dataset']} & {r['representation'].replace('_', ' ')} & {r['mean_loss_difference']:.6f} & [{r['empirical_q025_loss_difference']:.6f}, {r['empirical_q975_loss_difference']:.6f}] & {r['positive_map_count']}/20 \\\\")
    block=r'''\paragraph{Sensitivity to the intervention map.}
We replayed the seven saved aligned residual-ridge states under 20 additional seeded cyclic derangements, preserving each task's original eligible recipient slots, timestamps, numeric inputs, targets, and fitted parameters. The maps were shared across representations within each Time-IMM dataset and across the nine Fidel plants. No model or PCA was refitted. We report the mean paired scalar-target loss difference and empirical 2.5th--97.5th percentiles across these specified map draws; these are between-map ranges, not sampling confidence intervals, permutation-test null distributions, or uncertainty over trained neural checkpoints.
'''
    block+='\n'+r'\begin{tabular}{llrrr}'+'\n'+r'\toprule'+'\n'+r'Task & Representation & Mean loss difference & Between-map range & Positive maps \\'+'\n'+r'\midrule'+'\n'+'\n'.join(table)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n\n'
    block+=r'''The Fidel loss increase was positive for all 20 maps and all nine plants, whereas the EPA-Air ranges crossed zero for all representations. The FNSPID sentence and finance representations also changed sign for some maps. These checks expose dependence on a single map while retaining the earlier linear-capacity and generalization limitations. All recipient timestamps were retained, but donor content could come from later in the same split; this is an offline semantic-misalignment intervention and does not validate publication-time availability. In particular, Fidel remains a regular, timestamp-restricted published-embedding experiment rather than a verified real-time irregular positive control. The repeated-map analysis covers these seven ridge states and does not calibrate the 15 neural-model states.
'''
    (ROOT/'MANUSCRIPT_BLOCK.tex').write_text(block,encoding='utf8')
    (ROOT/'results'/'distinct_map_counts.json').write_text(json.dumps(unique_maps,indent=2),encoding='utf8')
    print('Repeated-map report and manuscript block saved.',flush=True)
if __name__=='__main__':main()
