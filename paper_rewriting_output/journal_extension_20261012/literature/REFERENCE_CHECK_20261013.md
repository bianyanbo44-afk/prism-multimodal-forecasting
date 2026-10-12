# PRISM 全部参考文献真实性核查（2026-10-13）

**结论：40 条全部找到本次联网核查的官方来源，未发现虚构作品或 DOI 指向其他作品。** 当前 PDF 实际引用 25 条，另有 15 条仅存在于 BibTeX 库。库中已有的 20 个 DOI 全部可解析，返回题名、作者和作品身份相符。

本次重新查阅会议/期刊官网、arXiv、官方模型仓库及 Crossref，并保存响应、时间和哈希。此前 2026-10-12 报告只用作问题线索，不作为本次结论的替代证据。核查日期按北京时间计；证据中的 UTC 2026-10-12 17:xx 对应北京时间 2026-10-13 01:xx。

## 已修正

| 条目 | 问题 | 处理 | 对正文的影响 |
|---|---|---|---|
| AIR / `seo2025air` | 第 7 作者姓名拼写错误：Junhyuk Kang | 改为 **Junhyeok Kang**；[arXiv摘要](https://arxiv.org/abs/2512.10229)与[v3全文](https://arxiv.org/html/2512.10229v3)一致 | 未引用库条目；25 条正文参考文献不变 |
| MM-SHAP / `parcalabescu2023mmshap` | 题名末尾使用 and，官网使用 &；仅属符号差异 | 规范为 **Models & Tasks**；[ACL 官方记录](https://aclanthology.org/2023.acl-long.223/) | 未引用库条目；25 条正文参考文献不变 |

未新增文献，未删除文献。`references_before.bib` 保留核查前版本，`references_verified.bib` 保存修正后版本。

## 来源间的页码差异

以下六处是数据库与出版方记录的差异，不属于文献虚构。按出版方官网及原PDF裁定，避免把正确页码改成数据库中的另一组页码。

| 条目 | 出版方页码 | Crossref 页码 | 当前处理 |
|---|---|---|---|
| [reimers2019sentencebert](https://aclanthology.org/D19-1410/) | 3982–3992 | 3980–3990 | 采用ACL官方页码；当前值正确。 |
| [peng2022ogm](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html) | 8238–8247 | 8228–8237 | 采用CVF官方页码；当前值正确。 |
| [dancette2021vqa](https://openaccess.thecvf.com/content/ICCV2021/html/Dancette_Beyond_Question-Based_Biases_Assessing_Multimodal_Shortcut_Learning_in_Visual_Question_ICCV_2021_paper.html) | 1574–1583 | 1554–1563 | 采用CVF官网及原PDF页码；当前值正确。 |
| [wang2020hard](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html) | 12695–12705 | 12692–12702 | 采用CVF官网及原PDF页码；当前值正确。 |
| [zhang2024mla](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html) | 27456–27466 | 27446–27456 | 采用CVF官网及原PDF页码；当前值正确。 |
| [moebus2025contimask](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4eb5daabc45b45a9a312aa2c8fca8a74-Abstract-Conference.html) | 54779–54804 | 61017–61042 | 两个正式来源存在冲突；当前条目不列页码，保留此做法。 |

## 关键记录进一步确认

- **VoT、TaTS 的 ICLR 2026 归属已确认。** OpenReview正文被浏览器验证页阻挡，但本次取得[VoT官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a1fc7890141f1ada3d8bc6199cce07-Abstract-Conference.html)和[TaTS官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/29de5722d7d2a6946c08d0c0162a1c71-Abstract-Conference.html)，同时下载同站正式PDF确认首页题名、完整作者和ICLR 2026标记。VoT官网导出误写第5作者为Guo/Guo Guo；正式PDF及arXiv均为Chenjuan Guo，当前条目保留正确完整姓名。
- **Künsch 1989 的页码已补足新证据。** Crossref缺页码；[作者ETH主页](https://people.math.ethz.ch/~hkuensch/papers/)明确列出17卷、1217–1241，与当前条目一致。
- **Time-IMM、Contimask 的2025年无需改为2026年。** 会议及官方BibTeX归属2025/第38卷；2026-04-23是官网上线日期。
- **Fidel-TS引用v4有明确版本边界。** 初次公开于2025年，v4修订于2026-05-07，当前版本URL和说明一致。[版本记录](https://arxiv.org/abs/2509.24789v4)
- **TFB作者Buang Zhang有原PDF支持。** 没有根据名字是否常见来臆测或改写。[PVLDB原文](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf)
- **两条模型引用是真实版本资源。** FinBERT固定SHA为`4556d13015211d73dccd3fdd39d39232506f3e43`；MiniLM固定SHA为`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。两次官方revision API请求均返回匹配仓库与完整SHA。

## 逐条结果：当前PDF的25条

“确认”表示作品存在且当前书目信息与主要来源相符；页码来源冲突另列于上表。完整作者顺序、字段对比、各来源URL与本次证据路径在同目录的结构化JSON中。

| 正文编号 | 文献与主要核查来源 | 年份 | 载体 | 结果 |
|---|---|---|---|---|
| 1 | [Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8e7768122f3eeec6d77cd2b424b72413-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | Advances in Neural Information Processing Systems | 确认 |
| 2 | [Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html) | 2025 | Advances in Neural Information Processing Systems | 确认 |
| 3 | [Context is Key: A Benchmark for Forecasting with Essential Textual Information](https://proceedings.mlr.press/v267/williams25a.html) | 2025 | Proceedings of the 42nd International Conference on Machine Learning | 确认 |
| 4 | [Fidel-TS: A High-Fidelity Multimodal Benchmark for Time Series Forecasting](https://arxiv.org/abs/2509.24789v4) | 2025 |  | 确认 |
| 5 | [Rethinking Multimodal Time-Series Forecasting Evaluation](https://proceedings.mlr.press/v306/liu26go.html) | 2026 | Proceedings of the 43rd International Conference on Machine Learning | 确认 |
| 6 | [MoTime: A Dataset Suite for Multimodal Time Series Forecasting](https://arxiv.org/abs/2505.15072) | 2025 |  | 确认 |
| 7 | [TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts](https://proceedings.mlr.press/v306/lin26al.html) | 2026 | Proceedings of the 43rd International Conference on Machine Learning | 确认 |
| 8 | [Unlocking the Value of Text: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a1fc7890141f1ada3d8bc6199cce07-Abstract-Conference.html) | 2026 | International Conference on Learning Representations | 确认 |
| 9 | [Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative](https://proceedings.iclr.cc/paper_files/paper/2026/hash/29de5722d7d2a6946c08d0c0162a1c71-Abstract-Conference.html) | 2026 | International Conference on Learning Representations | 确认 |
| 10 | [Rethinking Multimodal Fusion for Time Series: Text Modalities Need Constrained Fusion](https://arxiv.org/abs/2603.22372) | 2026 |  | 确认 |
| 11 | [Shortcut learning in deep neural networks](https://www.nature.com/articles/s42256-020-00257-z) | 2020 | Nature Machine Intelligence | 确认 |
| 12 | [Characterizing and Overcoming the Greedy Nature of Learning in Multi-Modal Deep Neural Networks](https://proceedings.mlr.press/v162/wu22d.html) | 2022 | Proceedings of the 39th International Conference on Machine Learning | 确认 |
| 13 | [Balanced Multimodal Learning via On-the-Fly Gradient Modulation](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html) | 2022 | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition | 确认 |
| 14 | [A Closer Look at Multimodal Representation Collapse](https://proceedings.mlr.press/v267/chaudhuri25a.html) | 2025 | Proceedings of the 42nd International Conference on Machine Learning | 确认 |
| 15 | [When Does Multimodality Lead to Better Time Series Forecasting?](https://arxiv.org/abs/2506.21611) | 2025 |  | 确认 |
| 16 | [Are Language Models Actually Useful for Time Series Forecasting?](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) | 2024 | Advances in Neural Information Processing Systems | 确认 |
| 17 | [All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously](https://jmlr.org/papers/v20/18-760.html) | 2019 | Journal of Machine Learning Research | 确认 |
| 18 | [Conditional variable importance for random forests](https://link.springer.com/article/10.1186/1471-2105-9-307) | 2008 | BMC Bioinformatics | 确认 |
| 19 | [Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4eb5daabc45b45a9a312aa2c8fca8a74-Abstract-Conference.html) | 2025 | Advances in Neural Information Processing Systems | 确认；未填争议页码 |
| 20 | [A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction](https://www.sciencedirect.com/science/article/pii/S0167947317302384) | 2018 | Computational Statistics and Data Analysis | 确认 |
| 21 | [The Jackknife and the Bootstrap for General Stationary Observations](https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary-Observations/10.1214/aos/1176347265.full) | 1989 | The Annals of Statistics | 确认 |
| 22 | [The Stationary Bootstrap](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870) | 1994 | Journal of the American Statistical Association | 确认 |
| 23 | [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) | 2019 | Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing | 确认 |
| 24 | [all-MiniLM-L6-v2 model release](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 未虚构年份 | 模型版本资源 | 确认 |
| 25 | [FinBERT model release](https://huggingface.co/ProsusAI/finbert) | 未虚构年份 | 模型版本资源 | 确认 |

## 逐条结果：未引用的15条库记录

| BibTeX key | 文献与主要核查来源 | 年份 | 载体 | 结果 |
|---|---|---|---|---|
| `liu2026posttime` | [Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting](https://arxiv.org/abs/2605.29401) | 2026 |  | 确认 |
| `seo2025air` | [Adaptive Information Routing for Multimodal Time Series Forecasting](https://arxiv.org/abs/2512.10229) | 2025 |  | 确认；作者拼写已修正 |
| `zhang2024tpatchgnn` | [Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach](https://proceedings.mlr.press/v235/zhang24bw.html) | 2024 | Proceedings of the 41st International Conference on Machine Learning | 确认 |
| `hessel2020emap` | [Does My Multimodal Model Learn Cross-Modal Interactions? It's Harder to Tell Than You Might Think!](https://aclanthology.org/2020.emnlp-main.62/) | 2020 | Proceedings of EMNLP | 确认 |
| `frank2021visionlanguage` | [Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers](https://aclanthology.org/2021.emnlp-main.775/) | 2021 | Proceedings of EMNLP | 确认 |
| `dancette2021vqa` | [Beyond Question-Based Biases: Assessing Multimodal Shortcut Learning in Visual Question Answering](https://openaccess.thecvf.com/content/ICCV2021/html/Dancette_Beyond_Question-Based_Biases_Assessing_Multimodal_Shortcut_Learning_in_Visual_Question_ICCV_2021_paper.html) | 2021 | Proceedings of the IEEE/CVF International Conference on Computer Vision | 确认 |
| `parcalabescu2023mmshap` | [MM-SHAP: A Performance-Agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks](https://aclanthology.org/2023.acl-long.223/) | 2023 | Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics | 确认；题名符号已规范 |
| `wang2020hard` | [What Makes Training Multi-Modal Classification Networks Hard?](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html) | 2020 | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition | 确认 |
| `zhang2024mla` | [Multimodal Representation Learning by Alternating Unimodal Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html) | 2024 | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition | 确认 |
| `liang2021multibench` | [MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract-round1.html) | 2021 | Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks | 确认 |
| `chen2023causal` | [Causal Intervention and Counterfactual Reasoning for Multi-Modal Fake News Detection](https://aclanthology.org/2023.acl-long.37/) | 2023 | Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics | 确认 |
| `lin2023counterfactual` | [Counterfactual Augmentation for Multimodal Learning Under Presentation Bias](https://aclanthology.org/2023.findings-emnlp.43/) | 2023 | Findings of the Association for Computational Linguistics: EMNLP 2023 | 确认 |
| `hewamalage2023forecast` | [Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices](https://link.springer.com/article/10.1007/s10618-022-00894-5) | 2023 | Data Mining and Knowledge Discovery | 确认 |
| `kaufman2012leakage` | [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://dl.acm.org/doi/10.1145/2382577.2382579) | 2012 | ACM Transactions on Knowledge Discovery from Data | 确认 |
| `qiu2024tfb` | [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) | 2024 | Proceedings of the VLDB Endowment | 确认 |

## 版本及证据

- 输入BibTeX SHA256：`bf2c09a58eeb4d0c6585967af1e715b1b21cec5172a281339c4257fd1d10b791`
- 修正后BibTeX SHA256：`11ebdd1550221aac62d9d2200a77d748ee1ef35926b6a274d41c6ab084c2d1a6`
- `FINAL_REFERENCE_AUDIT.json`：40条完整字段及来源；`crossref_results.json`：20个原有DOI的新请求；`primary_results.json`：ICLR和模型固定版本证据。原始响应保存在作者本地`qa/reference_verification_20261013/`；公开仓库提供报告和结构化书目核查结果，第三方原文PDF未再分发。
- 本次范围为真实性与书目信息核查；没有逐句审核本稿的观点是否得到各原文支持。
