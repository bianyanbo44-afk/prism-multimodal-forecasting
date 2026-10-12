"""Integrate executed construct and map results without changing frozen experiments."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'manuscript'

def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))

def table(head, rows, caption, label):
    return ('\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
        '\\begin{tabular}{'+'l'*len(head)+'}\n\\toprule\n'+' & '.join(head)+'\\\\\n\\midrule\n'
        +'\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)
        +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

def initialize_prose(name, content):
    """Seed a missing manuscript component without replacing edited prose."""
    path = PAPER/name
    if not path.exists():
        path.write_text(content, encoding='utf-8')

def refresh_table_rows(name, label, generated):
    """Refresh frozen-result rows while preserving edited captions and narrative."""
    path = PAPER/name
    if not path.exists():
        path.write_text(generated, encoding='utf-8')
        return
    existing = path.read_text(encoding='utf-8')
    pattern = re.compile(r'\\begin\{table\}.*?\\end\{table\}', re.DOTALL)
    source = next((m.group(0) for m in pattern.finditer(generated)
                   if '\\label{'+label+'}' in m.group(0)), None)
    target = next((m for m in pattern.finditer(existing)
                   if '\\label{'+label+'}' in m.group(0)), None)
    if source is None or target is None:
        raise ValueError(f'Missing table {label} in {name}')
    rows = source.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
    block = target.group(0)
    before, after = block.split('\\midrule', 1)
    _, tail = after.split('\\bottomrule', 1)
    replacement = before+'\\midrule'+rows+'\\bottomrule'+tail
    path.write_text(existing[:target.start()]+replacement+existing[target.end():], encoding='utf-8')

def main():
    cal=read('construct_controls/results/summary.json')['conditions']
    by={(x['mechanism'],x['condition']):x for x in cal}
    initialize_prose('construct_results.tex', r'''\subsection{Construct controls distinguish a directional proposition from a source cue}

Whole-content dependence does not uniquely identify reliance on an intended proposition. We generated independent direction signs and arbitrary Cedar/Maple source cues in twenty irregular worlds, assigning the target to direction alone, source alone or available-note count alone. The MiniLM ridge predictor passes the validation capability screen in all directional and source-only worlds. Whole permutation increases source-only MSE by 0.770870, with an across-world 95\% interval [0.758012,0.783728], and rejects in 20/20 worlds. Flipping the directional proposition while retaining the source suffix instead gives $-0.000228$ [$-0.000640$,0.000185]; flipping only the source gives 2.872553 [2.840252,2.904853]. The directional task shows the complementary pattern, with direction flip 2.698159 [2.667529,2.728789] and source flip $-0.000071$ [$-0.001688$,0.001546]. These controls discriminate the two stipulated constructs. Whole permutation alone correctly measures predictive content dependence in both but does not distinguish them.

Meaning-preserving paraphrases also change error. Direction/source probes recover all sixteen test sentences and sixteen paraphrases, but paraphrasing reduces mean directional MSE by 0.030132 and source-task MSE by 0.005214. One directional world and three source worlds nevertheless show significant degradation. A preserved generated proposition therefore does not imply an invariant representation or forecast. For timing-only targets, whole permutation gives $-0.000195$ [$-0.000681$,0.000290], with one ungated rejection and no capability-qualified worlds. The comparator already includes event timing/counts; a numerical-only comparator was not fitted, so this experiment does not quantify complete-stream MMU. All fifteen outcomes remain in Table~\ref{tab:construct_full}.

\begin{figure}[htbp]
\centering\includegraphics[width=\textwidth]{fig7_construct}
\caption{Fixed-checkpoint controls under three predefined target mechanisms. Points and bars show mean altered-minus-aligned MSE and 95\% $t$ intervals across twenty generated worlds, conditional on shared empirical clocks. Direction flip changes the generated increase/decrease proposition while retaining source; source flip changes Cedar/Maple while retaining direction. Paraphrase retains both generated cues. The source names are linguistic content too; the experiment separates a specified directional proposition from an alternative cue rather than identifying all semantics. The complete rejection fractions and adverse paraphrase outcomes appear in the supplementary table.}
\label{fig:construct}
\end{figure}
''')

    design=r'''
\subsection{Construct-specific validation and repeated intervention maps}

Construct validation reuses the eight EPA-Air time templates with new independent numerical innovations, per-note direction/source signs, sentence assignments, arrival delays and target noise. Twenty worlds each contain 128 training, 32 validation and 128 test entities, with nonoverlapping seven-day histories. Targets add to the numerical baseline either a variance-normalized recency-pooled direction, the corresponding independently generated source cue, or training-standardized log available-note count. Noise standard deviation is 0.35. Sixteen directional/source sentences train the representation pipeline, sixteen different directional phrasings provide validation/test inputs and sixteen further phrasings provide paraphrases. The source vocabulary stays fixed across splits deliberately. Frozen MiniLM pooling, training-only PCA (at most fifteen components) and scaling precede ridge selection from $0.0001,0.01,1,100$ on mean validation entity loss. No latent direction/source readout enters the sentence predictor. The sentence predictor and numerical-plus-timing comparator each undergo validation selection, yielding 480 candidate fits and 120 selected states across the full matrix. The oracle readout directly evaluates the known generator signal without fitting.

Each aligned predictor evaluates whole permutation, direction flip, source flip, meaning-preserving paraphrase and zero at fixed event slots. Whole permutation is a nonzero cyclic rotation within available histories, retaining empty/singleton cases. Targeted edits preserve the alternative generated cue. All 1,440,000 saved-state test predictions reproduce exactly. Tests pair mean losses over 128 generated entities within a world; effect intervals use twenty independent worlds, conditional on the same empirical clocks. All predefined cells are reported, with descriptive rejection rates and no universal multiplicity or error-control claim.

The map-sensitivity batch retains the independent aligned ridge states on EPA-Air, FNSPID and Fidel-TS and draws twenty additional maps with seeds 120001--120020. Each entity/split receives an independent uniform nonzero circular shift of ordered content; the same shifts are shared across representations. Fidel shifts the common report sequence within split and shares its map across all nine plants. These draws supplement the original map, without refitting or validation reselection. They report empirical 2.5th--97.5th percentiles and sign counts over maps, separately from population or training uncertainty. Donor timestamps, changed representations, targets and noncontent features are checked for every draw. This batch does not repeat maps for the original neural checkpoints.
'''
    p=PAPER/'new_design.tex'
    s=p.read_text(encoding='utf-8')
    marker='\n\\subsection{Construct-specific validation and repeated intervention maps}'
    if marker not in s:
        p.write_text(s.rstrip()+ '\n'+design,encoding='utf-8')
    rows=[]
    names={'semantic_only':'Direction','proxy_only':'Source','timing_only':'Timing'}
    controls={'whole_permutation':'Whole permutation','meaning_flip':'Direction flip',
              'proxy_flip':'Source flip','meaning_preserving_paraphrase':'Paraphrase','zero':'Zero'}
    for x in cal:
        lo,hi=x['delta_world_mean_ci95']
        rows.append([names[x['mechanism']],controls[x['condition']],f"{x['mean_delta_MSE']:+.6f}",
                     f'[{lo:+.6f}, {hi:+.6f}]',f"{x['rejected_worlds']}/20"])
    cs='\\section{Full construct-specific control outcomes}\n\n'
    cs+=table(['Target','Control','$\\Delta$ MSE','Across-world 95\\% interval','Rejected'],rows,
        'All predefined construct/control outcomes. Rejected means an ungated one-sided paired entity test at 0.05. Intervals concern the mean over twenty generated worlds, not all natural-language constructs. Direction and source targets each qualify 20/20 worlds; timing qualifies 0/20. A 20/20 rejection fraction has Wilson 95\\% interval [83.9,100]\\%; zero rejections still have upper bound 16.1\\%.', 'tab:construct_full')
    cs+=r'''
\paragraph{Construct and intervention identities.}
Direction/source signs are independent per note. A direction flip inverts the demand proposition and retains the exact source suffix; a source flip exchanges Cedar/Maple and retains exact proposition text. The source labels are also meaningful strings, so the contrast concerns these predefined cues. Paraphrases preserve both latent cues but need not preserve embedding geometry. Capability uses the frozen validation criterion of a positive lower confidence bound plus more than 1\% benefit over the numerical/timing comparator. Source/semantic worlds pass it, yet whole permutation alone cannot isolate the directional proposition. No numerical-only model was trained for this batch. Full templates, validation candidates and all saved predictions are in the local replay package.
'''
    refresh_table_rows('construct_supplement.tex', 'tab:construct_full', cs)

    with (ROOT/'intervention_repeats/results/between_map_summary.csv').open(encoding='utf-8-sig') as f: maps=list(csv.DictReader(f))
    rows=[]
    for x in maps:
        rows.append([x['dataset'],{'gpt2':'GPT-2','minilm':'MiniLM','finbert':'FinBERT','published_weather':'Weather'}[x['representation']],
                     f"{float(x['mean_loss_difference']):+.6f}",
                     f"[{float(x['empirical_q025_loss_difference']):+.6f}, {float(x['empirical_q975_loss_difference']):+.6f}]",
                     f"{x['positive_map_count']}/20"])
    rs=r'''\subsection{Repeated maps expose intervention sensitivity separately}

All twenty additional Fidel maps raise the same aligned checkpoint's loss. Mean $\Delta$ MSE is 0.047600; the empirical 2.5th--97.5th percentile range is [0.032916,0.064487], and mean map-wise control-denominator SAG is 20.65\%. This supports a stable positive direction across the evaluated shift distribution, rather than treating the nine plants as independent map replications. The irregular ridge results are less stable. Positive map counts are 0/20, 3/20 and 3/20 on FNSPID for GPT-2, MiniLM and FinBERT; EPA-Air counts are 2/20, 6/20 and 8/20. EPA's sign therefore depends on the selected map even with unchanged states and targets. Table~\ref{tab:maps} reports all seven cells.

The percentile ranges describe realized intervention variation and are not sampling confidence intervals. Future-dated donors remain possible in this offline perturbation. Their fraction ranges from 1.10\% to 98.90\% of Fidel recipient windows, 33.96\% to 70.96\% of selected FNSPID text slots and 28.36\% to 59.70\% of selected EPA slots. These different denominators are retained explicitly. The aligned input remains timestamp-restricted; repeated maps do not establish deployment-time donor availability.
'''
    rs+=table(['Dataset','Input','Mean $\\Delta$ MSE','Empirical map range','Positive maps'],rows,
        'Same-checkpoint loss effects over twenty additional nonzero-shift maps. Ranges are empirical 2.5th--97.5th percentiles across maps and are not confidence intervals. All rows use scalar-pooled MSE. Checkpoint state and forecast units remain fixed.', 'tab:maps')
    map_table_component=('repeated_map_table_supplement.tex'
                         if (PAPER/'repeated_map_table_supplement.tex').exists()
                         else 'repeated_map_results.tex')
    refresh_table_rows(map_table_component, 'tab:maps', rs)
    initialize_prose('repeated_map_supplement.tex', r'''\section{Transformation laws and map conditioning}

The original fixed-neural batch uses control seed 9103 with entity/split random permutations, derangements and nested dose subsets, shared across architecture/training-seed conditions. Each condition has one realized map and its entity/block bootstrap conditions on that map. The original independent ridge batches also use seed 91801, with a nonzero cyclic rotation per entity/split (the shared report sequence within split for Fidel), rather than unrestricted random derangements. Construct calibration rotates within available histories and targeted edits have deterministic cue mappings. These intervention laws define different perturbation distributions and are not interchangeable.

The repeated ridge maps use the additional twenty seeds 120001--120020, with independent uniform nonzero shifts per permitted group. Source maps are shared across GPT-2/MiniLM/FinBERT for each dataset/draw. The batch retains 140 map-by-cell aggregates, 1,260 entity/plant effect rows and 94,000 donor/recipient timing rows. The minimum and maximum effects, changed-slot counts, donor timing distributions and all per-unit directions are retained, alongside baseline prediction replay, state and target checks. The twenty draws characterize this particular shift distribution; they are not exhaustive permutations or calibrated uncertainty over the original neural models.
''')

    records=read('fixed_model/results/manifest.json')['records']
    rows=[]
    for ds in ['ClusterTrace','CESNET','EPA-Air','FNSPID','RepoHealth']:
        for arch,label in [('selective_ssm','Selective'),('immtsf_recavg_gradd','RecAvg'),('immtsf_t2v_xattn','XAttn')]:
            rr=[x for x in records if x['dataset']==ds and x['architecture']==arch and x['condition']=='aligned']
            assert len(rr)==3
            populations={(x['metrics']['windows'],len(x['metrics']['entity_mse'])) for x in rr}
            assert len(populations)==1
            w,e=populations.pop()
            rows.append([ds,label,e,w,sum(x['historical_best_epoch']==0 for x in rr),'Not assessed'])
    pops=table(['Dataset','Mechanism','Entities','Windows','Initialized/3','Capability gate'],rows,
        'Retained population per aligned checkpoint and historical selection status. Three seeds share each row population, so windows are not multiplied by three. Initialized counts refer to historical best epoch zero. No formal capability gate was applied retrospectively to these checkpoints.', 'tab:fixed_population')
    refresh_table_rows('population_supplement.tex', 'tab:fixed_population', pops)
    p=PAPER/'supplement.tex';s=p.read_text(encoding='utf-8')
    if '\\input{population_supplement}' not in s:s=s.replace('\\input{generated_supplement}','\\input{population_supplement}\n\n\\input{generated_supplement}')
    p.write_text(s,encoding='utf-8')
    inputs=['construct_controls/results/summary.json','intervention_repeats/results/between_map_summary.csv','fixed_model/results/manifest.json']
    (PAPER/'ADDED_EVIDENCE_STATE.json').write_text(json.dumps({'sources':{x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in inputs},'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
    print('Construct cells:',len(cal),'map cells:',len(maps),'population rows:',len(rows))

if __name__=='__main__': main()
