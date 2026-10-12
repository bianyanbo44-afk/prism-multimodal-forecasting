"""Assemble reporting tables from saved results; perform no fitting or new tests."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript"
DATASETS = {"ClusterTrace": "clustertrace", "CESNET": "cesnet", "EPA-Air": "epa_air",
            "FNSPID": "fnspid", "RepoHealth": "repohealth"}
MECHANISMS = {"selective_ssm": "Selective", "immtsf_recavg_gradd": "RecAvg", "immtsf_t2v_xattn": "XAttn"}
REPRESENTATIONS = {"gpt2": "GPT-2", "minilm": "MiniLM", "finbert": "FinBERT"}


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def table(head, rows, caption, label, columns=None, size="small", position="htbp"):
    body = "\n".join((row if isinstance(row, str) else " & ".join(map(str, row))) + "\\\\" for row in rows)
    return ("\\begin{table}[" + position + "]\n\\centering\\" + size + "\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\begin{tabular}{" + (columns or "l" * len(head)) + "}\n\\toprule\n"
            + " & ".join(head) + "\\\\\n\\midrule\n" + body
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def sci(value):
    mantissa, exponent = f"{value:+.3e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def representation_text(reports, forecasting):
    rows, missing = [], []
    for dataset in ("EPA-Air", "FNSPID"):
        lookup = {(r["representation"], r["training_condition"], r["evaluation_condition"]): r
                  for r in reports[dataset]["results"]}
        p = next(r for r in forecasting if r["dataset"] == dataset)
        nt = lookup[("none", "event_structure", "event_structure")]["test"]["mse"]
        rows.append(r"\multicolumn{6}{l}{" + dataset
                    + f": persistence {p['test_persistence_mse']:.6f}; numerical/timing {nt:.6f}" + "}")
        for rep, name in REPRESENTATIONS.items():
            needed = [(rep, "aligned", "aligned"), (rep, "aligned", "semantic_derangement"),
                      (rep, "semantic_derangement", "semantic_derangement")]
            if not all(k in lookup for k in needed):
                missing.append(f"{dataset}/{rep}")
                continue
            a, s, r = [lookup[k]["test"]["mse"] for k in needed]
            rows.append([dataset, name, f"{a:.6f}", f"{s:.6f}", f"{s-a:+.6f}", f"{r:.6f}"])
    full_table = table(["Dataset", "Representation", "Aligned", "Fixed shuffle", r"$\Delta_{\rm fix}$", "Retrained shuffle"], rows,
        r"Independent ridge representation comparisons, with task baselines shown above each dataset. Loss is pooled observed-scalar standardized test MSE; $\Delta_{\rm fix}=L_{\rm fixed\ shuffle}-L_A$ is the primary effect scale. Fixed shuffle retains the aligned checkpoint, whereas retrained shuffle has its own validation-selected checkpoint. All numerical-adapter windows remain, with at most four historical notes in the content summary. Validation evidence is reported in Table~\ref{tab:ridge_capability_details}; these scores differ from the historical feature-macro endpoint.",
        "tab:representations", size="footnotesize")
    content = r"""\begin{figure}[!htbp]
\centering\includegraphics[width=\textwidth]{fig8_transfer}
\caption{Validation comparisons and later test transfer of independent ridge models. Pooled observed-scalar MSE is divided by persistence MSE from the same split; one is the baseline and lower is better. The vertical axis is logarithmic. EPA-Air and FNSPID include numerical, numerical/timing and three aligned text configurations; Fidel-TS includes numerical-only and aligned weather configurations relative to seasonal persistence. Each line connects one validation-selected state across splits. Ratios are descriptive transformations of saved means. Exact losses and penalties appear in Table~\ref{tab:ridge_capability_details}; perturbation contrasts appear in Table~\ref{tab:representations}.}
\label{fig:transfer}
\end{figure}

Validation comparisons and later test transfer are shown in Figure~\ref{fig:transfer}; fixed-model and retrained contrasts are reported in Table~\ref{tab:representations}.

"""
    content += r"""
A training-only probe on twelve financial direction sentences allowed MiniLM and FinBERT to classify all twelve differently phrased held-out sentences correctly. The probed direction is therefore recoverable within this vocabulary; forecasting performance requires separate evidence.

Replacement geometry also depends on a shared representation component. On FNSPID, GPT-2's median replacement cosine falls from 0.998964 to 0.188214 after subtracting the training centroid; MiniLM's falls from 0.454059 to 0.142177. On EPA-Air, FinBERT's median falls from 0.768842 to $-0.013962$. Raw cosine can thus be dominated by a shared direction and does not determine whether a task-relevant proposition changed.

On EPA-Air, all three content models improve the validation mean over persistence (1.075034). Only FinBERT improves over numerical/timing ridge (0.901160 versus 0.906853, a descriptive 0.63\% benefit). Every model loses to test persistence (0.751407), including numerical-only and numerical/timing ridge with MSEs of 27.056535 and 24.127104. Negative fixed-shuffle differences therefore accompany poor test-period transfer.

On FNSPID, persistence validation MSE is 0.648214, below numerical ridge (0.675381), numerical/timing ridge (0.675222) and all three content models (0.673852, 0.679578 and 0.680205). GPT-2's 0.20\% mean improvement over numerical/timing ridge thus provides no overall capability advantage. Later gains over test persistence (0.320936) cannot retrospectively qualify validation capability. No formal paired validation screen was applied to these representation comparisons.

Aggregation can reverse the effect's direction. For EPA-Air FinBERT, fixed shuffle reduces scalar-pooled MSE from 1.950746 to 1.945247 but increases entity-macro MSE from 2.388571 to 2.398929. Three of eight entities have a positive alignment point effect. FNSPID has 4/10 such entities for GPT-2, 2/10 for MiniLM and 4/10 for FinBERT. The pooled endpoint therefore need not represent a common entity-level direction.
"""
    if missing:
        content += "\n\\paragraph{Internal build status.} Unfinished comparisons: " + ", ".join(missing) + ".\n"
    return content, missing, full_table


def historical_validation(historical_root, manifest):
    rows, inputs = [], []
    for dataset, slug in DATASETS.items():
        multimodal = historical_root / "results" / (f"{slug}_multimodal_ln_fresh3_cuda/results.json" if slug != "fnspid"
                                                      else "fnspid_untouched_gate/multimodal_models/results.json")
        semantic = historical_root / "results" / (f"{slug}_semantic_controls_fresh3_cuda/results.json" if slug != "fnspid"
                                                    else "fnspid_untouched_gate/semantic_controls/results.json")
        persistence = next(r for r in read(multimodal)["results"] if r["model"] == "last_value")
        inputs.extend([multimodal, semantic])
        for architecture, name in MECHANISMS.items():
            selected = sorted((r for r in manifest["records"] if r["dataset"] == dataset
                               and r["architecture"] == architecture and r["condition"] == "aligned"),
                              key=lambda r: r["seed"])
            if [r["seed"] for r in selected] != [53, 67, 79]:
                raise ValueError(f"Incomplete historical aligned records: {dataset}/{architecture}")
            if architecture == "selective_ssm":
                records = read(multimodal)["results"] + read(semantic)["results"]
                aligned_name, timing_name = "selective_eventmamba_no_reliability", "selective_ssm_time_only_zero_text"
            else:
                family = "fusion" if name == "RecAvg" else "xattn"
                source = historical_root / "results" / f"{slug}_immtsf_{family}_audit_fresh3/results.json"
                inputs.append(source)
                records = read(source)["results"]
                suffix = "_unit_consistent" if name == "RecAvg" else ""
                aligned_name, timing_name = architecture + "_aligned" + suffix, architecture + "_time_only" + suffix
            aligned = sorted((r for r in records if r["model"] == aligned_name), key=lambda r: r["seed"])
            timing = sorted((r for r in records if r["model"] == timing_name), key=lambda r: r["seed"])
            if [r["seed"] for r in aligned] != [53, 67, 79] or [r["seed"] for r in timing] != [53, 67, 79]:
                raise ValueError(f"Incomplete archived validation comparison: {dataset}/{name}")
            for original, replay in zip(aligned, selected):
                if abs(original["val_metrics"]["mse"] - replay["historical_val_mse"]) > 1e-12:
                    raise ValueError("Archived/replay validation-loss mismatch")
                if original["best_epoch"] != replay["historical_best_epoch"]:
                    raise ValueError("Archived/replay selected-epoch mismatch")
            for record in aligned + timing:
                for key in ("windows", "query_rows", "observed_values"):
                    if record["val_metrics"][key] != persistence["val_metrics"][key]:
                        raise ValueError(f"Validation population mismatch: {dataset}/{name}/{key}")
            rows.append([dataset, name, f"{persistence['val_metrics']['mse']:.6f}",
                         f"{fmean(r['val_metrics']['mse'] for r in timing):.6f}",
                         f"{fmean(r['historical_val_mse'] for r in selected):.6f}",
                         "/".join(str(r["historical_best_epoch"]) for r in selected)])
    return rows, list(dict.fromkeys(inputs))


def capability_text(reports, forecasting, historical_rows):
    calibration = read("calibration/results/summary.json")["conditions"]
    constructs = read("construct_controls/results/summary.json")["conditions"]
    # Check the saved counts used by the concise batch descriptions.
    if {r["capability_qualified_worlds"] for r in calibration if r["strength"] == 0} != {0}:
        raise ValueError("Calibration null-screen count changed")
    if {r["capability_qualified_worlds"] for r in calibration if r["strength"] > 0} != {20}:
        raise ValueError("Calibration nonzero-screen count changed")
    construct_counts = {r["mechanism"]: r["capability_qualified_worlds"] for r in constructs}
    if construct_counts != {"semantic_only": 20, "proxy_only": 20, "timing_only": 0}:
        raise ValueError("Construct-screen counts changed")
    if sum(row[-1].split("/").count("0") for row in historical_rows) != 4:
        raise ValueError("Historical initialized-selection count changed")
    overview_rows = [
        [r"Historical fixed neural models; restricted random permutation and full derangement",
         r"Archived persistence, time-only and aligned validation losses (Table~\ref{tab:historical_validation}). Formal screen: \emph{Not assessed}; 4/45 selections at epoch zero.",
         r"Derangement: 56.25\% on CESNET; 100\% elsewhere.",
         r"Very small fixed effects; interpretation remains conditional on these checkpoints and maps."],
        [r"Linear known-signal calibration; nonzero within-history rotation",
         r"Fixed fitting settings. Incremental-benefit screen: 0/20 null worlds; 20/20 at each nonzero strength, for both representations.",
         r"97.27\% of source positions; 48.52\% of direction values.",
         r"Known signal detected in 20/20 worlds per nonzero cell; unscreened null detections 3/20 and 2/20."],
        [r"Construct ridge models; nonzero within-history rotation and cue edits",
         r"Same-sample ridge selection and post-selection nominal screen: direction/source 20/20; timing 0/20.",
         r"Empty and singleton histories retained; alternative cue preserved in each targeted edit.",
         r"Whole-content effects occur for both cues; targeted flips distinguish the generated cue used."],
        [r"EPA-Air ridge; nonzero entity/split circular shift",
         r"Persistence validation MSE 1.075034; aligned 0.901160--0.968275. Only FinBERT improves the numerical/timing mean. Formal screen: \emph{Not assessed}.",
         r"All selected representations change; latest-four-note summary.",
         r"All models lose to test persistence; fixed pooled effects are negative."],
        [r"FNSPID ridge; nonzero entity/split circular shift",
         r"Persistence validation MSE 0.648214; aligned 0.673852--0.680205. No overall mean improvement. Formal screen: \emph{Not assessed}.",
         r"All selected representations change; latest-four-note summary.",
         r"Fixed pooled effects are negative; later test gains do not qualify validation capability."],
        [r"Fidel ridge; nonzero shift of shared split report sequence",
         r"Validation numerical 0.221023; aligned 0.208860; persistence 0.350711. Descriptive stream utility 5.50\%. Formal screen: \emph{Not assessed}.",
         r"All recipient weather vectors change; one common map across nine plants.",
         r"Fixed $\Delta$ MSE +0.044461; retrained +0.017906. The two effects answer different questions."],
    ]
    overview = table(["Evaluation and intervention", "Validation evidence and screening status", "Achieved intervention", "Test evidence"], overview_rows,
        r"Capability evidence and interpretation by batch. Real-data validation comparisons are descriptive; no formal screen was applied. Persistence addresses overall prediction, while a numerical/event comparator addresses incremental content benefit. Fidel's numerical-only comparison measures complete-stream utility. Repeated-map draws retain the aligned model and use the ridge rows' circular-shift families.",
        "tab:capability_overview", columns=r"@{}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}@{}", size="footnotesize", position="!ht")
    overview += r"""
The batches resolve different questions: EPA-Air separates validation improvement from later transfer; FNSPID separates incremental content benefit from overall prediction; Fidel demonstrates complete-stream utility in a regular task. Historical validation losses contextualize fixed-checkpoint effects without adding a screening decision. Full losses and selected epochs appear in Tables~\ref{tab:historical_validation} and~\ref{tab:ridge_capability_details}.
"""
    appendix = r"""\section{Saved validation comparisons and their interpretation}
\label{sec:capability_records}

The following tables report existing validation records. Historical checkpoints remain \emph{Not assessed} by a formal screen, and the independent real-data ridge batch did not calculate a paired validation lower bound for screening. The summaries provide comparator context without changing checkpoint selection, introducing test-driven qualifications or excluding outcomes.

"""
    appendix += table(["Dataset", "Mechanism", "Persistence", "Time-only", "Aligned", "Epochs"], historical_rows,
        r"Historical selected-model validation MSE and epochs. Persistence is the archived last-observation baseline. Time-only is a separately fitted and validation-selected condition with zero text vectors and retained event metadata. Train-only standardized loss uses the original feature-macro endpoint; stochastic-model losses are means over seeds 53/67/79. Epochs list the aligned selections in that order, with zero indicating initialization. The validation window/query/observed-value counts match within each dataset comparison. All 45 checkpoints retain formal screening status \emph{Not assessed}; retained test populations appear in Table~\ref{tab:fixed_population}.",
        "tab:historical_validation", size="footnotesize")
    appendix += r"""
These validation means answer distinct descriptive questions: persistence contextualizes overall prediction, while the condition-retrained time-only model contextualizes content beyond retained occurrence information. An epoch-zero selection identifies a checkpoint state; by itself it does not establish inferiority to persistence. The historical seed average is not an independent sample count or a new confidence estimate.

"""
    ridge_rows = []
    for dataset in ("EPA-Air", "FNSPID"):
        records = reports[dataset]["results"]
        p = next(r for r in forecasting if r["dataset"] == dataset)
        ridge_rows.append([dataset, "Persistence", "--", f"{p['validation_persistence_mse']:.6f}", f"{p['test_persistence_mse']:.6f}"])
        for rep, condition, name in [("none", "numeric_only", "Numerical"), ("none", "event_structure", "Numerical/timing")]:
            r = next(r for r in records if r["representation"] == rep and r["training_condition"] == condition and r["evaluation_condition"] == condition)
            ridge_rows.append([dataset, name, f"{r['selected_alpha']:g}", f"{r['val']['mse']:.6f}", f"{r['test']['mse']:.6f}"])
        for rep, name in REPRESENTATIONS.items():
            r = next(r for r in records if r["representation"] == rep and r["training_condition"] == "aligned" and r["evaluation_condition"] == "aligned")
            ridge_rows.append([dataset, name + " aligned", f"{r['selected_alpha']:g}", f"{r['val']['mse']:.6f}", f"{r['test']['mse']:.6f}"])
    for condition, name in [("persistence", "Seasonal persistence"), ("numeric_only", "Numerical"), ("aligned", "Aligned weather")]:
        r = next(r for r in reports["Fidel"]["results"] if r["training_condition"] == condition and r["evaluation_condition"] == condition)
        ridge_rows.append(["Fidel-TS", name, f"{r['selected_alpha']:g}" if "selected_alpha" in r else "--", f"{r['val']['mse']:.6f}", f"{r['test']['mse']:.6f}"])
    appendix += table(["Dataset", "Configuration", r"Selected $\lambda$", "Validation MSE", "Test MSE"], ridge_rows,
        r"Saved independent ridge validation and test losses. All losses pool observed standardized scalar targets; $\lambda$ is selected from 1, 10, 100 and 1000 using validation MSE. Numerical/timing adds available-note counts and recency to the numerical design; it supplies the matched comparator for incremental content benefit on Time-IMM. Fidel numerical features include calendar and lag summaries, with no separate weather-event comparator fitted. Validation means are reported after selection and confer no formal screen pass. Test losses contextualize transfer and did not select configurations.",
        "tab:ridge_capability_details", size="footnotesize")
    appendix += r"""
For EPA-Air, all three aligned configurations improve the persistence validation mean, but only FinBERT improves the numerical/timing mean (0.63\%). For FNSPID, none improves the persistence mean; only GPT-2 improves the numerical/timing mean (0.20\%). These point comparisons do not supply the paired lower bound of Eq.~\ref{eq:validation_content_screen}. For Fidel, the aligned validation mean improves by 5.50\% over numerical-only ridge. That comparison measures descriptive complete-stream utility; it does not isolate incremental content benefit with an event-preserving comparator. Seasonal persistence repeats the preceding daily profile, and all plants share weather context.

Time-IMM validation comprises 41 EPA-Air windows and 317 FNSPID windows, with 8,048 and 40,590 observed scalar targets, respectively. Test counts are 43/321 windows and 7,508/41,220 targets. These targets and overlapping windows are scoring units rather than independent replications. Fidel has 3,285 validation and 3,285 test windows, each with 24 hourly targets, across nine plants. Condition-retrained controls have their own validation selection and remain separate from fixed-model effects. Entity-macro results are reported alongside pooled MSE where their directions differ.
"""
    return overview, appendix


def generated_supplement_text():
    calrows = []
    for r in read("calibration/results/summary.json")["conditions"]:
        ms = r["mean_test_MSE"]
        calrows.append(["Oracle" if r["representation"] == "oracle" else "MiniLM", f"{r['strength']:.2f}",
                        f"{ms['aligned']:.6f}", f"{ms['fixed_shuffle']:.6f}", f"{ms['retrained_shuffle']:.6f}", f"{r['detected_worlds']}/20"])
    supplement = table(["Input", "Strength", "Aligned", "Fixed shuffle", "Retrained shuffle", "Detected"], calrows,
        r"Full known-signal calibration means over twenty generated worlds after optimizer-seed and entity averaging. Detected means the unscreened within-world one-sided paired test rejects. Zero-strength rows retain false detections. Worlds are independent conditional on eight empirical timing templates; strengths reuse inputs within worlds.", "tab:calibration_full")
    with (ROOT / "fixed_model/results/paired_summary_control_denominator.csv").open(encoding="utf-8-sig", newline="") as f:
        fixed = list(csv.DictReader(f))
    fixedrows = []
    for r in fixed:
        if r["condition"] == "derangement_100":
            fixedrows.append([r["dataset"], MECHANISMS[r["architecture"]], sci(float(r["control_mse"]) - float(r["aligned_mse"])),
                              f"{float(r['gain_percent_control_denominator']):+.6f}",
                              f"[{float(r['gain_control_ci95_block2_low']):+.6f}, {float(r['gain_control_ci95_block2_high']):+.6f}]"])
    supplement += table(["Dataset", "Mechanism", r"$\Delta$ MSE", r"Fixed SAG (\%)", r"SAG 95\% range (\%)"], fixedrows,
        r"Full-derangement fixed effects: raw feature-macro $\Delta$ MSE and secondary control-denominator SAG. Ranges use the paired block-length-two bootstrap; length-four/eight records are retained separately. These are exploratory uncertainty descriptions under the specified entity/block resampling, conditional on the selected checkpoints and realized maps. Small entity counts constrain their wider-population interpretation (Table~\ref{tab:fixed_population}). Each cell averages three aligned checkpoints; initialized selections and coverage remain visible.",
        "tab:fixed_full", size="footnotesize")
    supplement += r"""
\section{Calibration null behaviour and validation screening}

The unscreened null rejection rates are 3/20 for oracle and 2/20 for MiniLM, with Wilson 95\% intervals [5.24,36.04]\% and [2.79,30.10]\%. The incremental-benefit screen retains no null world, giving screened detections 0/20 for each representation and Wilson interval [0,16.11]\%. These are joint screening-and-detection outcomes over all worlds. No retained null world is available to estimate a conditional rejection rate among screen-positive null worlds.

The screen uses the two-sided 95\% paired-entity lower bound and the more-than-1\% relative improvement rule in Eq.~\ref{eq:validation_content_screen}. Calibration fitting settings are fixed and training-only; the three optimizer seeds are averaged within each of 32 validation entities before the comparison. The construct batch uses the same algebra after ridge selection on those validation entities, so its bound is post-selection nominal and its screening interpretation descriptive. Neither batch establishes general 5\% error control, and the chosen margin may exclude weaker real signals.

\section{Independent model and encoder identities}

MiniLM revision is \nolinkurl{1110a243fdf4706b3f48f1d95db1a4f5529b4d41}; FinBERT revision is \nolinkurl{4556d13015211d73dccd3fdd39d39232506f3e43}. Pooling, maximum token length, cache file hashes and original CSV hashes are in the representation manifest. FinBERT's pinned model card does not supply explicit licence metadata, so the study package includes retrieval instructions rather than redistributing those weights. The two encoders have distinct domain adaptation and pooling provenance.

The independent photovoltaic ridge model selects its penalty from 1, 10, 100 and 1000 using 2021 validation data. Numerical-only features include daily and weekly lag summaries and annual sine/cosine; the content condition adds the most recent published weather-vector summary and training-only PCA. Forecast origin stride is 24 hours; the recorded weather timestamp is exactly at the daily cutoff in this retained population. Neither report issuance nor future-report knowability follows from that equality. Saved validation comparisons appear in Table~\ref{tab:ridge_capability_details}.
"""
    return supplement, len(calrows), len(fixedrows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true", help="Internal build only; never a final manuscript")
    parser.add_argument("--historical-root", type=Path, default=ROOT.parents[1] / "phase3_prototype",
                        help="Archived prototype root containing the saved historical validation records")
    args = parser.parse_args()
    reports = {d: read(f"external_validation/results/{d}/results.json") for d in ("EPA-Air", "FNSPID", "Fidel")}
    forecasting = read("external_validation/results/summary_table.json")["forecasting"]
    representation, missing, representation_table = representation_text(reports, forecasting)
    if missing and not args.allow_incomplete:
        raise RuntimeError("Unfinished representation comparisons: " + ", ".join(missing))
    historical_rows, historical_inputs = historical_validation(args.historical_root, read("fixed_model/results/manifest.json"))
    overview, capability_supplement = capability_text(reports, forecasting, historical_rows)
    supplement, calibration_count, fixed_count = generated_supplement_text()
    for name, content in {"representation_results.tex": representation, "representation_table_supplement.tex": representation_table,
                          "generated_supplement.tex": supplement,
                          "capability_results.tex": overview, "capability_supplement.tex": capability_supplement}.items():
        (PAPER / name).write_text(content, encoding="utf-8")
    inputs = [ROOT / rel for rel in ["calibration/results/summary.json", "construct_controls/results/summary.json",
                                     "fixed_model/results/paired_summary_control_denominator.csv",
                                     "fixed_model/results/manifest.json", "external_validation/results/summary_table.json",
                                     "calibration/run_calibration.py", "construct_controls/run_construct_controls.py"]]
    inputs += [ROOT / f"external_validation/results/{d}/results.json" for d in reports]
    inputs += historical_inputs
    source_hashes = {}
    for p in inputs:
        try:
            identity = str(p.relative_to(ROOT))
        except ValueError:
            identity = "phase3_prototype/" + str(p.relative_to(args.historical_root))
        source_hashes[identity] = hashlib.sha256(p.read_bytes()).hexdigest()
    (PAPER / "EVIDENCE_TABLE_STATE.json").write_text(json.dumps({"complete": not missing, "missing": missing,
        "source_sha256": source_hashes, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
    print(f"Assembled saved-result tables: representation cells={6-len(missing)}, calibration rows={calibration_count}, "
          f"fixed-model cells={fixed_count}, historical validation rows={len(historical_rows)}; complete={not missing}")


if __name__ == "__main__":
    main()
