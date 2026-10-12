# Data and code access

| Evidence | Public location or acquisition route |
|---|---|
| Generated known-signal calibration | `calibration/results/` under the extension directory: paired entity losses, summaries, fixed fitting configuration and 1,200 saved linear states. Larger generated feature/target arrays are in `PRISM_Synthetic_Evidence.zip` at release `v0.1.0`. |
| Generated direction/source/timing constructs | `construct_controls/results/`: all fifteen predefined outcomes, model selection, paired entity losses and 120 fitted states. The release asset contains all twenty generated worlds. |
| Eight figures and tables | Editable PDF/SVG/PNG, source CSV and saved aggregate reporting inputs under `manuscript/figures/`; two table generators are under `scripts/`. |
| Historical irregular checkpoints | Public code, aggregate effects, checkpoint identities and validation summaries are included. Original natural inputs, fitted neural checkpoint binaries and full per-window replay material remain outside this public subset. |
| Independent natural-data ridge and repeated maps | Public runners, protocol, reported losses and aggregate map effects. Natural feature/target matrices, report vectors and cached representations are acquired or regenerated locally from the upstream inputs. |

The generated arrays contain synthetic numerical histories, propositions, features, targets and loss records under fixed empirical event-time templates. They do not contain natural EPA-Air values or narratives. The arrays replay the saved representation pipeline; rebuilding event clocks and re-encoding phrases requires the upstream resources below.

## Upstream resources

- Time-IMM: https://github.com/blacksnail789521/Time-IMM, recorded commit `e934cef23d526817dcf0d1a66c6f82274c96c04b`. Obtain the five processed datasets through the provider and retain its terms. Place data under `phase3_prototype/external/Time-IMM/data/` for the retained adapter layout.
- Fidel-TS photovoltaic inputs: https://huggingface.co/datasets/VEWOXIC/Canada_photovoltaics_plants, recorded revision `752799306917192f05f0846ffc897ec70efb0998`. The retained layout is `phase3_prototype/data/fidel_ts/Canada_photovoltaics_plants/` with `time_series` and the provider's weather report embeddings.
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- FinBERT: https://huggingface.co/ProsusAI/finbert, revision `4556d13015211d73dccd3fdd39d39232506f3e43`. The experiment uses pooled hidden states without the classifier head.
- GPT-2 representations are the historical benchmark's masked-mean caches; reconstruction code is included in `phase3_prototype/scripts/`.

The study has not established public redistribution terms for all upstream datasets, weather embeddings or FinBERT weights. These resources and their natural-data derivatives are therefore accessed through the providers rather than redistributed here. MiniLM's model-card metadata records Apache-2.0; its weights are also obtained separately. No new permission is asserted for any upstream material.

The larger local review/replay ZIP contains additional benchmark-derived arrays and is not a public release asset. Public generated-data verification is performed by `verify_public_release.py`; it does not claim a reconstruction of the natural neural forecasts or re-estimation of bootstrap intervals.

中文核对：公开位置、版本和获取路线均已列出；未虚构 DOI、模型/数据许可或真实作者名单。上游再分发条件及自有代码的最终复用许可仍由作者按实际权属确定。
