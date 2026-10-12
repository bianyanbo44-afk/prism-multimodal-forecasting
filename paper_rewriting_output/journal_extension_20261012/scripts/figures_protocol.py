"""Protocol schematic and complete construct controls, in Matplotlib.

The protocol positions are explicitly illustrative, with no measured times or
performance values. The construct panels preserve all 15 saved means and
across-world intervals; this module does no fitting or inferential testing.
Shared typography, vector settings and the export/alignment gate are implemented
in figure_style.py. Outputs are fig1_estimands.pdf, fig1_estimands.svg,
fig1_estimands.png and fig7_construct.pdf, fig7_construct.svg, fig7_construct.png.
"""
from __future__ import annotations

import json

from figure_style import (
    ROOT, OUT, WIDTH, BLUE, ORANGE, TEAL, PURPLE, GRAY, INK, LIGHT,
    read_json, read_csv, source_csv, panel_label, style_axis, save_figure,
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


def _arrow(ax, start, end, color=GRAY, *, lw=1.0, rad=0.0):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, color=color,
        linewidth=lw, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def _capsule(ax, x, y, width, height, text, color, *, fontsize=9.0):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="white", edgecolor=color, linewidth=1.1, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=color,
            fontsize=fontsize, fontweight="bold", zorder=5)


def _setup_diagram(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")


def build_estimands():
    """Illustrate event invariants and the two distinct evaluation objects."""
    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(
        2, 2, left=.085, right=.98, bottom=.10, top=.91,
        height_ratios=(1.15, 1), hspace=.42, wspace=.20,
    )
    event_ax = fig.add_subplot(gs[0, :])
    fixed_ax = fig.add_subplot(gs[1, 0])
    refit_ax = fig.add_subplot(gs[1, 1])
    for ax in (event_ax, fixed_ax, refit_ax):
        _setup_diagram(ax)

    panel_label(event_ax, "a", "Retain the irregular event slots")
    panel_label(fixed_ax, "b", "Fixed-model reliance")
    panel_label(refit_ax, "c", "Value after condition-specific refitting")

    # Schematic spacing only: no experimental timestamps or numerical values.
    numerical_positions = [.19, .235, .35, .375, .49, .59, .62, .81, .85]
    text_positions = [.20, .39, .62, .83]
    tokens = ["A", "B", "C", "D"]
    token_colors = [BLUE, ORANGE, TEAL, PURPLE]
    donor_map = [1, 2, 3, 0]
    event_ax.text(.54, .98, "Illustrative event layout", ha="center",
                  va="top", fontsize=8.5, color=GRAY)
    event_ax.text(.0, .78, "Numerical\nobservations", va="center", fontsize=9)
    event_ax.text(.0, .52, "Original\ncontent", va="center", fontsize=9)
    event_ax.text(.0, .25, "Reassigned\ncontent", va="center", fontsize=9)
    event_ax.plot([.17, .90], [.78, .78], color=LIGHT, linewidth=1.4, zorder=0)
    event_ax.scatter(numerical_positions, [.78] * len(numerical_positions),
                     s=22, facecolor=INK, edgecolor="white", linewidth=.4, zorder=3)
    for x in text_positions:
        for bottom, top in ((.14, .16), (.34, .43), (.61, .85)):
            event_ax.plot([x, x], [bottom, top], color=LIGHT, lw=.85,
                          linestyle=(0, (2, 3)), zorder=0)
    for index, x in enumerate(text_positions):
        _capsule(event_ax, x, .52, .060, .145, tokens[index], token_colors[index])
        donor = donor_map[index]
        _capsule(event_ax, x, .25, .060, .145, tokens[donor], token_colors[donor])
        if index < 3:
            _arrow(event_ax, (text_positions[donor], .435), (x, .337),
                   token_colors[donor], lw=1.0, rad=.05)
    # The wrap-around donor has a deliberately distinct curved route.
    _arrow(event_ax, (text_positions[0], .435), (text_positions[3], .337),
           token_colors[0], lw=.95, rad=-.18)
    event_ax.plot([.93, .93], [.14, .86], color=GRAY, linestyle=(0, (3, 3)), lw=.8)
    event_ax.text(.93, .885, "Origin q", ha="center", va="bottom", fontsize=8.5)
    event_ax.text(.5, .035,
                  "Fixed: entities, splits, event/availability times, masks and counts",
                  ha="center", va="top", fontsize=8.5, fontweight="bold")
    event_ax.text(.5, -.077,
                  "Same numerical inputs, targets and forecast windows",
                  ha="center", va="top", fontsize=8.5)

    # The shared state is visually explicit in the fixed-function path.
    fixed_ax.text(.5, .99, "Aligned fitting + validation selection",
                  ha="center", va="top", fontsize=8.5)
    _arrow(fixed_ax, (.5, .81), (.5, .70), BLUE)
    fixed_ax.add_patch(Circle((.5, .565), .12, facecolor=LIGHT,
                             edgecolor=BLUE, linewidth=1.3, zorder=1))
    fixed_ax.text(.5, .565, "f", fontsize=16, fontstyle="italic",
                  color=BLUE, ha="center", va="center")
    fixed_ax.text(.72, .565, "One frozen\ncheckpoint", ha="left",
                  va="center", fontsize=8.5, color=BLUE)
    _arrow(fixed_ax, (.44, .465), (.24, .31), BLUE)
    _arrow(fixed_ax, (.56, .465), (.76, .31), BLUE)
    fixed_ax.text(.24, .20, "Aligned test\nloss", fontsize=9,
                  ha="center", va="center")
    fixed_ax.text(.76, .20, "Altered test\nloss", fontsize=9,
                  ha="center", va="center")
    fixed_ax.text(.5, -.09, r"$\Delta_{\mathrm{fixed}}$ = altered − aligned",
                  fontsize=12, color=BLUE, ha="center", va="top")

    # Refitting changes both the fitted state and its matching test condition.
    for x, name, symbol, color in (
        (.24, "Aligned fit", "f", BLUE),
        (.76, "Altered fit", r"$f_{\mathcal{T}}$", ORANGE),
    ):
        refit_ax.text(x, .99, name, ha="center", va="top", fontsize=9,
                      fontweight="bold", color=color)
        refit_ax.text(x, .825, "+ matching validation", ha="center",
                      va="top", fontsize=8.2, color=GRAY)
        _arrow(refit_ax, (x, .72), (x, .685), color)
        refit_ax.add_patch(Circle((x, .565), .12, facecolor=LIGHT,
                                 edgecolor=color, linewidth=1.3, zorder=1))
        refit_ax.text(x, .565, symbol, fontsize=16,
                      color=color, ha="center", va="center")
        _arrow(refit_ax, (x, .445), (x, .31), color)
        refit_ax.text(x, .20, name.replace("fit", "test") + "\nloss",
                      ha="center", va="center", fontsize=9)
    refit_ax.text(.5, -.09, r"$\Delta_{\mathrm{retrain}}$ = altered − aligned",
                  fontsize=12, color=ORANGE, ha="center", va="top")

    schematic_rows = [
        {"record_type": "illustrative_event_slot", "slot": i + 1,
         "original_token": tokens[i], "reassigned_token": tokens[donor_map[i]],
         "event_layout": "Illustrative; no empirical times or values"}
        for i in range(4)
    ]
    source_csv("fig1_estimands", schematic_rows)
    return save_figure(
        fig, "fig1_estimands",
        sources=("manuscript/main.tex", "manuscript/protocol_worked_examples.tex",
                 "scripts/figure_style.py", "scripts/figures_protocol.py"),
        note=("Illustrative event-slot schematic only; no experimental timestamps, "
              "numerical observations or loss values are fabricated. Panels b and c "
              "separate one frozen function from condition-specific refitting."),
    )


CONDITIONS = (
    "whole_permutation", "meaning_flip", "proxy_flip",
    "meaning_preserving_paraphrase", "zero",
)
CONDITION_NAMES = {
    "whole_permutation": "Whole shift",
    "meaning_flip": "Direction flip",
    "proxy_flip": "Source flip",
    "meaning_preserving_paraphrase": "Paraphrase",
    "zero": "Zero content",
}
CONDITION_COLORS = {
    "whole_permutation": GRAY, "meaning_flip": BLUE,
    "proxy_flip": ORANGE, "meaning_preserving_paraphrase": TEAL,
    "zero": GRAY,
}


def _effect_text(value):
    """Give nonzero small means legible labels without calling them zero."""
    if abs(value) < .001:
        return f"{value:+.6f}"
    return f"{value:+.3f}"


def _cue_cards(ax, config):
    """Read the actual template cues; the displayed sentences are abbreviated."""
    direction_up = "increase" if "increase" in config["test_direction_pairs"][0][0] else None
    direction_down = "decrease" if "decrease" in config["test_direction_pairs"][0][1] else None
    source_up = config["proxy_positive_suffix"].split(": ")[1].rstrip(".")
    source_down = config["proxy_negative_suffix"].split(": ")[1].rstrip(".")
    assert direction_up and direction_down, "The example must match the frozen templates"
    cards = [
        ("Aligned", direction_up, source_up),
        ("Direction flip", direction_down, source_up),
        ("Source flip", direction_up, source_down),
        ("Paraphrase", direction_up, source_up),
    ]
    for index, (name, direction, source) in enumerate(cards):
        x0 = index * .255
        ax.add_patch(FancyBboxPatch(
            (x0, .145), .232, .78,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor=LIGHT, edgecolor="none", zorder=0,
        ))
        ax.text(x0 + .116, .81, name, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x0 + .116, .565, direction, ha="center", va="center",
                fontsize=10, color=BLUE,
                fontweight="bold" if index == 1 else "normal")
        ax.text(x0 + .116, .35, source, ha="center", va="center",
                fontsize=10, color=ORANGE,
                fontweight="bold" if index == 2 else "normal")
    ax.text(.5, -.02, "Paraphrase retains both cues.", ha="center", va="top",
            fontsize=8.5, color=GRAY)
    ax.text(-.018, .565, "Direction", ha="right", va="center", fontsize=8.5, color=BLUE)
    ax.text(-.018, .35, "Source", ha="right", va="center", fontsize=8.5, color=ORANGE)


def _construct_effect_panel(ax, rows, *, timing=False, first=False):
    multiplier = 1000 if timing else 1
    for index, condition in enumerate(CONDITIONS):
        y = 4 - index
        row = rows[condition]
        mean = float(row["mean_delta_MSE"])
        low = float(row["delta_ci_low"])
        high = float(row["delta_ci_high"])
        assert low <= mean <= high, "The saved interval must contain its saved mean"
        assert int(row["worlds"]) == 20
        color = CONDITION_COLORS[condition]
        ax.barh(y, mean * multiplier, height=.29, color=color,
                alpha=.68 if condition in ("meaning_flip", "proxy_flip") else .34,
                edgecolor="none", zorder=2)
        ax.errorbar(
            mean * multiplier, y,
            xerr=np.array([[mean - low], [high - mean]]) * multiplier,
            fmt="o", markersize=4.8, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=.6, ecolor=color, elinewidth=1.4, capsize=3,
            capthick=1.1, zorder=4,
        )
        if timing:
            # Labels remain in original MSE units in the traceable source CSV.
            label = f"{mean * multiplier:+.3f}"
            label_x = mean * multiplier
            if abs(mean * multiplier) < .7:
                label_x, align = -.8, "right"
            else:
                align = "center"
        else:
            label = _effect_text(mean)
            label_x, align = (mean, "center") if abs(mean) >= .10 else (.12, "left")
        ax.annotate(
            label, (label_x, y), xytext=(0, 8), textcoords="offset points",
            ha=align, va="bottom", fontsize=8.5,
            color=color if condition in ("meaning_flip", "proxy_flip") else INK,
            fontweight="bold" if abs(mean) > 2 else "normal", zorder=5,
        )

    ax.axvline(0, color=INK, lw=.85, zorder=1)
    ax.set(ylim=(-.45, 4.70), yticks=np.arange(5))
    if first:
        ax.set_yticklabels([CONDITION_NAMES[c] for c in CONDITIONS[::-1]])
        ax.tick_params(axis="y", length=0, pad=9)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    style_axis(ax)
    if timing:
        ax.set(xlim=(-16.3, 1.45), xticks=(-15, -10, -5, 0))
        ax.set_xlabel("Altered − aligned MSE\nUnits of 0.001 (expanded)", fontsize=9)
    else:
        ax.set(xlim=(-.14, 3.5), xticks=(0, 1, 2, 3))
        ax.set_xlabel("Altered − aligned MSE", fontsize=9)


def build_construct():
    """Contrast two stipulated cues and retain the process-only boundary."""
    data_path = "construct_controls/results/summary.csv"
    config_path = "construct_controls/config.json"
    records = read_csv(data_path)
    config = read_json(config_path)
    mechanisms = ("semantic_only", "proxy_only", "timing_only")
    assert len(records) == 15, "The full construct population has 15 predefined cells"
    assert {(r["mechanism"], r["condition"]) for r in records} == {
        (m, c) for m in mechanisms for c in CONDITIONS
    }, "No predefined construct cell may be lost or duplicated"
    lookup = {(r["mechanism"], r["condition"]): r for r in records}

    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(
        2, 3, left=.185, right=.975, bottom=.16, top=.91,
        height_ratios=(1, 1.86), hspace=.43, wspace=.24,
    )
    cue_ax = fig.add_subplot(gs[0, :])
    _setup_diagram(cue_ax)
    effect_axes = [fig.add_subplot(gs[1, index]) for index in range(3)]
    panel_label(cue_ax, "a", "Change one cue while retaining the other")
    _cue_cards(cue_ax, config)

    for index, (ax, mechanism, title) in enumerate(zip(
        effect_axes, mechanisms, ("Direction target", "Source target", "Timing target")
    )):
        panel_label(ax, chr(ord("b") + index), title)
        _construct_effect_panel(
            ax, {condition: lookup[(mechanism, condition)] for condition in CONDITIONS},
            timing=mechanism == "timing_only", first=index == 0,
        )

    aligned = config["test_direction_pairs"][0][0] + config["proxy_positive_suffix"]
    examples = {
        "meaning_flip": config["test_direction_pairs"][0][1] + config["proxy_positive_suffix"],
        "proxy_flip": config["test_direction_pairs"][0][0] + config["proxy_negative_suffix"],
        "meaning_preserving_paraphrase": config["paraphrase_direction_pairs"][0][0] + config["proxy_positive_suffix"],
    }
    source_rows = []
    for record in records:
        source_rows.append({
            "mechanism": record["mechanism"], "condition": record["condition"],
            "mean_delta_MSE": record["mean_delta_MSE"],
            "delta_ci_low": record["delta_ci_low"], "delta_ci_high": record["delta_ci_high"],
            "worlds": record["worlds"], "interval": "Existing across-world 95% t interval",
            "aligned_example": aligned,
            "altered_example": examples.get(record["condition"], ""),
            "display_multiplier": 1000 if record["mechanism"] == "timing_only" else 1,
        })
    source_csv("fig7_construct", source_rows)
    return save_figure(
        fig, "fig7_construct",
        sources=(data_path, config_path, "scripts/figure_style.py", "scripts/figures_protocol.py"),
        note=("All 15 saved means and existing across-world 95% t intervals are shown. "
              "No new test or fit. Direction/source panels share an MSE scale; the "
              "timing boundary has an explicitly expanded ×1000 horizontal scale. "
              "Cue cards abbreviate actual frozen sentence templates."),
    )


def build_all():
    """Return the two provenance records after their alignment-gated export."""
    return [build_estimands(), build_construct()]


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
