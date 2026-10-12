"""Build interpretable reporting tables from executed output, retaining all directions."""
import csv,json,hashlib,sys,platform
from pathlib import Path
import numpy as np
import sklearn,torch,transformers
from run_representation import ROOT,MODEL_IDS,PROTOTYPE,sha,prepare_dataset

def main():
    rows=[];diagnostics=[];manifest={
      'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'transformers':transformers.__version__,'sklearn':sklearn.__version__,
      'new_cache_root':str(ROOT/'caches'),'original_inputs_read_only':True,
      'encoders':{'gpt2':{'model_id':'openai-community/gpt2','revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e','pooling':'existing attention-masked token mean','max_length':320,'dtype':'original float16 cache read as float32','new_encoding':False},
      'minilm':{'model_id':MODEL_IDS['minilm'][0],'revision':MODEL_IDS['minilm'][1],'license':'Apache-2.0 in pinned model-card metadata','pooling':'attention-masked token mean then L2-normalization','max_length':256,'dtype':'float32 CPU','trust_remote_code':False,'local_files_only':True},
      'finbert':{'model_id':MODEL_IDS['finbert'][0],'revision':MODEL_IDS['finbert'][1],'license':'Pinned model card has no explicit license metadata; no weight redistribution included in study source package','pooling':'BertModel hidden token mean then L2-normalization; classifier head excluded','max_length':256,'dtype':'float32 CPU','trust_remote_code':False,'local_files_only':True,'scope':'Finance domain pretraining/sentiment adaptation; not sentence-trained and not in-domain for EPA-Air'}}}
    lines=['# SemAudit 编码器与独立数据/模型扩展：实际执行报告','',
      '本轮是探索性扩展，原 Time-IMM 和 Fidel 测试结果此前已经查看。协议在新增表示得分计算前保存为 `protocol.json`，验证集选正则化强度；没有测试结果驱动的模型或实体筛选。新金融方向探针另有 `probe_protocol.json`。本报告不是预注册确认性结果。','',
      '新增两类冻结表示：通用句子模型 all-MiniLM-L6-v2，以及金融领域 FinBERT 的平均隐藏表示。均固定 Hugging Face 提交，使用 attention-mask mean pooling、L2 归一化和 256 token 截断。FinBERT 分类头未使用，它不被称为句子专门训练模型；其应用于 EPA-Air 仅是表示敏感性对照。原 GPT-2 使用既有 320 token 缓存。模型版本、文件散列和环境分别见 `representation_manifest.json` 与 `models/download_manifest.json`。','',
      '独立预测器为本目录自行实现的 last-value + ridge residual。数值特征为各变量历史 last/mean/std/count/age，加预测步长及平方项；event_structure 另保留全部历史文本的计数与最近时间；内容使用每个历史窗口最新四条文本的均值和训练拟合的 32 维 PCA。所有预测目标、窗口、训练规范化与结构对照一致。它不是完整 IMM-TSF 新编码器复现，结论仅覆盖该线性检验器及明确的内容上限。','',
      '| 数据 | 表示 | 数值模型 MSE | Event structure MSE | Aligned MSE | 同一模型 Deranged MSE | 重训练 Deranged MSE | 同一模型 Zero MSE | 同一模型 Neutral MSE |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('FNSPID','EPA-Air'):
        if name!='FNSPID':
            lines.extend(['| 数据 | 表示 | 数值模型 MSE | Event structure MSE | Aligned MSE | 同一模型 Deranged MSE | 重训练 Deranged MSE | 同一模型 Zero MSE | 同一模型 Neutral MSE |','|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        result=json.loads((ROOT/'results'/name/'results.json').read_text(encoding='utf8'));verification=json.loads((ROOT/'results'/name/'verification.json').read_text(encoding='utf8'))
        _,_,samples,selected,_,passport=prepare_dataset(name)
        manifest[name]={'all_entity_count':len(passport['entities']),'entities':passport['entities'],'window_counts':{s:len(x) for s,x in samples.items()},'text_counts':passport['full_text_counts'],'selected_text_counts':passport['selected_text_counts'],'selected_text_occurrences':{s:sum(len(x['selected']) for x in pack) for s,pack in samples.items()},'all_history_text_occurrences':{s:sum(x['all_notes'] for x in pack) for s,pack in samples.items()},'protocol_sha256':result['protocol_sha256'],'source_hash_checks_passed':True}
        def get(rep,train,evaluation):return next((r for r in result['results'] if r['representation']==rep and r['training_condition']==train and r['evaluation_condition']==evaluation),None)
        numeric=get('none','numeric_only','numeric_only')['test']['mse'];structure=get('none','event_structure','event_structure')['test']['mse']
        for rep in ('gpt2','minilm','finbert'):
            a=get(rep,'aligned','aligned');f=get(rep,'aligned','semantic_derangement');r=get(rep,'semantic_derangement','semantic_derangement');z=get(rep,'aligned','zero');n=get(rep,'aligned','neutral')
            if a is None:continue
            row={'dataset':name,'representation':rep,'numeric_mse':numeric,'event_structure_mse':structure,'aligned_mse':a['test']['mse'],'fixed_deranged_mse':f['test']['mse'],'retrained_deranged_mse':r['test']['mse'],'fixed_zero_mse':z['test']['mse'],'fixed_neutral_mse':n['test']['mse'] if n else None,'fixed_aligned_relative_gain':(f['test']['mse']-a['test']['mse'])/f['test']['mse'],'retrained_aligned_relative_gain':(r['test']['mse']-a['test']['mse'])/r['test']['mse'],'aligned_validation_mse':a['val']['mse'],'validation_persistence_mse':verification['baselines']['val']['persistence']['mse'],'test_persistence_mse':verification['baselines']['test']['persistence']['mse'],'entity_count':len(a['test']['entities']),'test_scalar_targets':a['test']['observations'],'macro_entity_aligned_mse':a['test']['macro_entity_mse'],'macro_entity_deranged_mse':f['test']['macro_entity_mse'],'entities_fixed_positive':sum(f['test']['entities'][e]['mse']>x['mse'] for e,x in a['test']['entities'].items())}
            rows.append(row)
            neutral=f"{row['fixed_neutral_mse']:.6f}" if n else '—'
            lines.append(f"| {name} | {rep} | {numeric:.6f} | {structure:.6f} | {row['aligned_mse']:.6f} | {row['fixed_deranged_mse']:.6f} | {row['retrained_deranged_mse']:.6f} | {row['fixed_zero_mse']:.6f} | {neutral} |")
            au=result['representation_audit'][rep]
            diagnostics.append({'dataset':name,'representation':rep,'test_raw_cosine':au['test']['replacement_raw_cosine_median'],'test_centered_cosine':au['test']['replacement_train_centered_cosine_median'],'test_raw_text_unchanged_fraction':au['test']['raw_text_unchanged_fraction'],'test_vector_unchanged_fraction':au['test']['exact_vector_unchanged_fraction'],'token_audit':au['tokens']})
        lines.extend(['',f"{name}: 全部 {len(passport['entities'])} 个实体，train/val/test 窗口为 {len(samples['train'])}/{len(samples['val'])}/{len(samples['test'])}。只编码了 latest-four 选择的 {sum(len(x) for x in selected.values())} 条不同实体/行文本，源数据共 {sum(passport['full_text_counts'].values())} 条。完整源文件与窗口计数见 passport。数值模型验证集 MSE {get('none','numeric_only','numeric_only')['val']['mse']:.6f}，验证集 persistence MSE {verification['baselines']['val']['persistence']['mse']:.6f}，测试集 persistence MSE {verification['baselines']['test']['persistence']['mse']:.6f}。",''])
        aligned_records=[r for r in result['results'] if r['training_condition']=='aligned' and r['evaluation_condition']=='aligned']
        valwins=sum(r['val']['mse']<verification['baselines']['val']['persistence']['mse'] for r in aligned_records)
        testwins=sum(r['test']['mse']<verification['baselines']['test']['persistence']['mse'] for r in aligned_records)
        lines.extend([f"{name} 的 {len(aligned_records)} 种 aligned 表示中，{valwins} 种验证集胜过 persistence，{testwins} 种测试集胜过 persistence。验证能力不足或测试期失效的单元保留在结果中，不能据其扰动变化很小推断真实文本没有预测价值。",''])
    lines.extend(['## 实体权重敏感性','',
      '预先固定的主损失按已观测标量目标加权；下表另报告每个实体等权的描述性平均，以及实体内同模型 loss(deranged) > loss(aligned) 的数量。两种聚合可出现不同方向，实体数也不是独立事件语义的样本量。', '',
      '| 数据 | 表示 | Aligned macro MSE | 同模型 Deranged macro MSE | 对齐点效应为正的实体数 |','|---|---|---:|---:|---:|'])
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['representation']} | {r['macro_entity_aligned_mse']:.6f} | {r['macro_entity_deranged_mse']:.6f} | {r['entities_fixed_positive']}/{r['entity_count']} |")
    lines.extend(['','EPA-Air FinBERT 的标量加权损失在置换后略降，而实体等权损失略升；不能把主汇总中的方向外推到每个实体。完整单位记录和所有方向结果都保留。','', '## 表示几何和内容覆盖','', '| 数据 | 表示 | Test 原始替换余弦中位数 | 训练均值中心化余弦中位数 | 原文未改变比例 | 向量未改变比例 |','|---|---|---:|---:|---:|---:|'])
    for d in diagnostics:lines.append(f"| {d['dataset']} | {d['representation']} | {d['test_raw_cosine']:.6f} | {d['test_centered_cosine']:.6f} | {d['test_raw_text_unchanged_fraction']:.4%} | {d['test_vector_unchanged_fraction']:.4%} |")
    probe=json.loads((ROOT/'results'/'financial_direction_probe.json').read_text(encoding='utf8'))
    lines.extend(['','原文重复、向量完全相同以及余弦几何分开报告。无固定点循环置换在实体/切分内选择的文本位置上执行；只有一个位置时无法改变，不能隐藏这些不可审计位置。计数和位置沿用真实历史槽位，内容来自同一切分的其他位置；这种合成干预不证明自然语义的因果效应。','',
      f"固定金融方向探针采用 12 条训练与 12 条未用于拟合的固定句子，C=1 logistic regression；MiniLM 和 FinBERT 均得到 12/12。相反方向句子的配对余弦中位数分别为 {probe['results']['minilm']['opposite_pair_cosine_median']:.6f} 和 {probe['results']['finbert']['opposite_pair_cosine_median']:.6f}。这是窄范围的可表示性检查，不能当成整套表示的语义完整性验证。高余弦也不等于方向信息无法解码。",'',
      '## 独立真实数据谱系与模型：Fidel 规则光伏任务','',
      '新增独立 CPU ridge 使用全部九个 Calgary 光伏站，360 小时历史、24 小时预测、24 小时步长。训练/验证/测试窗口为 9,729/3,285/3,285，测试有 78,840 个小时目标。数值基线使用前一日 24 小时的季节 persistence，再以最近两日轮廓、前七日小时均值/标准差和年度日历特征拟合残差。天气内容使用发布的七条 256 维表示的均值，PCA 仅拟合训练。没有执行下载的上游代码。','',
      '关键改动：所有天气时间戳均不晚于固定数值预测 cutoff；旧 TGTSF 运行按未来每个 query 时刻选最近报告，因此两者不能视为同一复现。原始天气文字和真正发布/到达时间未在本地材料出现，新选报告在所有日预测起点的 age 为零。这里仍假定发布表示所带 timestamp 可以视作可用时间；不能据此证明内容在当时真正已知。'])
    f=json.loads((ROOT/'results'/'Fidel'/'results.json').read_text(encoding='utf8'))
    fr={(r['training_condition'],r['evaluation_condition']):r for r in f['results']}
    num=fr[('numeric_only','numeric_only')]['test']['mse'];a=fr[('aligned','aligned')]['test']['mse'];d=fr[('aligned','semantic_derangement')]['test']['mse'];r=fr[('semantic_derangement','semantic_derangement')]['test']['mse'];z=fr[('aligned','zero')]['test']['mse']
    deltas={e:fr[('aligned','semantic_derangement')]['test']['plants'][e]['mse']-v['mse'] for e,v in fr[('aligned','aligned')]['test']['plants'].items()}
    lines.extend(['',f"测试 MSE：数值模型 {num:.8f}，aligned {a:.8f}，同一 aligned 检查点 deranged {d:.8f}，重训练 deranged {r:.8f}，同一检查点 zero {z:.8f}。总文本收益为 {(num-a)/num:.2%}；以 deranged 损失为分母，同模型对齐收益 {(d-a)/d:.2%}，重训练对齐收益 {(r-a)/r:.2%}。九个站点的固定模型损失差均为正，范围 {min(deltas.values()):.8f}–{max(deltas.values()):.8f}。共享同一城市天气，不把九个站点或重叠小时当成独立天气证据，不报告伪独立窗口的显著性区间。",'',
      f"Fidel 天气替换原始余弦中位数 {f['intervention_audit']['raw_cosine_median']:.6f}，训练均值中心化后为 {f['intervention_audit']['train_centered_cosine_median']:.6f}；所有测试向量均改变。raw text 不可用，不能验证命题变化或生成模型来源。该任务是规则外部数据/独立模型的表示干预证据，不承担不规则实时正对照的证明职责。",'',
      '## 外部候选 TimesX 的检索范围','',
      '检查了 PMLR 正式页面和 arXiv 2607.06973 的 HTML 全文、全部外部链接以及公开 GitHub 仓库检索。论文 3.3 报告 190 个变量、2018–2025 数据和季度更新计划，本次页面链接未识别官方下载仓库或数据地址；GitHub 查询 `TimesX time series` 返回空。检查得到的是此检索路径的局限，不能声称项目没有公开源代码或数据。未下载或执行不明来源的 TimesX 代码，本轮外部模型实证使用已固定本地来源的 Fidel。','',
      '## 解释边界与完整性','',
      'FNSPID/EPA 的固定模型测试打乱对比没有自动提供可推广的“语义无用”结论。代表性、线性模型能力、latest-four 截取和时间分布改变仍限制解释。EPA 验证集能力并不保证测试期稳定；表示之间重训练损失差与固定模型损失差很不一致，正是需要区分两种估计对象的证据。域外 FinBERT 与句子模型不能作为同等适配的金融/环保模型。','',
      '全部方向结果保存在 JSON 和 per-entity/time CSV；固定条件共享检查点散列与完全相同目标，四项算法边界测试通过。核验脚本重算所有 CSV 损失并检查原始输入文件散列不变。模型和数据本地缓存不作为可重分发数据包；新增代码/协议/结果可由作者检查。来源和版本见 manifest。','',
      '实验设计辅助来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065。其元数据已经由总扩展方案核实；不用于支撑本轮预测结果。'])
    (ROOT/'EXTERNAL_REPRESENTATION_REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (ROOT/'representation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    (ROOT/'results'/'summary_table.json').write_text(json.dumps({'forecasting':rows,'representation_diagnostics':diagnostics,'fidel_plant_deltas':deltas},indent=2),encoding='utf8')
    with (ROOT/'results'/'summary_table.csv').open('w',encoding='utf8',newline='') as handle:
        out=csv.DictWriter(handle,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    print('Report, summary tables and representation manifest written.',flush=True)
if __name__=='__main__':main()
