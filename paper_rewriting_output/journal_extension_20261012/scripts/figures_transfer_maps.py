"""Saved validation transfer and fixed-checkpoint map variation; no fitting.

Transfer ratios use each split's own persistence MSE. Map strips retain every
draw and display the saved empirical percentiles, rather than confidence limits.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from figure_style import (
    ROOT, OUT, WIDTH, BLUE, GRAY, INK, PALE_BLUE,
    ENCODER_COLORS, DISPLAY_NAMES, panel_label, style_axis,
    read_json, read_csv, source_csv, save_figure,
)


TRANSFER_SOURCES = tuple(
    f"external_validation/results/{dataset}/results.json"
    for dataset in ("EPA-Air", "FNSPID", "Fidel")
) + ("external_validation/results/summary_table.json",)
MAP_SOURCES = (
    "intervention_repeats/results/map_aggregate.csv",
    "intervention_repeats/results/between_map_summary.csv",
)
ENCODERS = ("gpt2", "minilm", "finbert")
CONFIGURATION_STYLE = {
    "numeric_only": ("s", "--"),
    "event_structure": ("D", ":"),
    "gpt2": ("o", "-"),
    "minilm": ("^", "-"),
    "finbert": ("v", "-"),
    "published_weather": ("o", "-"),
}


def positive(value, label):
    """Require strictly positive saved MSEs and denominators for a log scale."""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Positive finite MSE required for {label}: {value}")
    return result


def same(values, label):
    first = float(values[0])
    if not all(math.isclose(first, float(v), rel_tol=0, abs_tol=1e-12)
               for v in values):
        raise ValueError(f"Inconsistent saved {label}")
    return first


def transfer_rows():
    reports = {dataset: read_json(f"external_validation/results/{dataset}/results.json")
               for dataset in ("EPA-Air", "FNSPID", "Fidel")}
    forecasting = read_json(TRANSFER_SOURCES[-1])["forecasting"]
    rows = []
    for dataset, report in reports.items():
        records = report["results"]
        if dataset == "Fidel":
            persistence = next(r for r in records
                               if r["training_condition"] == r["evaluation_condition"] == "persistence")
            baseline = {split: positive(persistence[split]["mse"], f"{dataset}/{split}/persistence")
                        for split in ("val", "test")}
            selected = [("numeric_only", next(r for r in records
                         if r["training_condition"] == r["evaluation_condition"] == "numeric_only")),
                        ("published_weather", next(r for r in records
                         if r["training_condition"] == r["evaluation_condition"] == "aligned"))]
            baseline_kind = "Seasonal persistence"
        else:
            persisted = [r for r in forecasting if r["dataset"] == dataset]
            if {r["representation"] for r in persisted} != set(ENCODERS):
                raise ValueError(f"Incomplete persistence references: {dataset}")
            baseline = {
                "val": positive(same([r["validation_persistence_mse"] for r in persisted],
                                     f"{dataset}/validation persistence"), dataset),
                "test": positive(same([r["test_persistence_mse"] for r in persisted],
                                      f"{dataset}/test persistence"), dataset),
            }
            aligned = [r for r in records
                       if r["training_condition"] == r["evaluation_condition"] == "aligned"]
            if {r["representation"] for r in aligned} != set(ENCODERS) or len(aligned) != 3:
                raise ValueError(f"Unexpected aligned encoder population: {dataset}")
            selected = [(condition, next(r for r in records
                         if r.get("representation") == "none"
                         and r["training_condition"] == r["evaluation_condition"] == condition))
                        for condition in ("numeric_only", "event_structure")]
            selected += [(rep, next(r for r in aligned if r["representation"] == rep))
                         for rep in ENCODERS]
            baseline_kind = "Last-value persistence"
        for split in ("val", "test"):
            observations = {r[split]["observations"] for _, r in selected}
            if len(observations) != 1:
                raise ValueError(f"Different scoring populations: {dataset}/{split}")
            rows.append({"dataset": dataset, "configuration": "persistence",
                         "display_name": baseline_kind, "split": split,
                         "mse": baseline[split], "persistence_mse": baseline[split],
                         "error_ratio": 1.0, "baseline_kind": baseline_kind,
                         "observed_scalar_targets": observations.pop()})
        for configuration, record in selected:
            for split in ("val", "test"):
                mse = positive(record[split]["mse"], f"{dataset}/{configuration}/{split}")
                rows.append({"dataset": dataset, "configuration": configuration,
                             "display_name": DISPLAY_NAMES[configuration], "split": split,
                             "training_condition": record["training_condition"],
                             "evaluation_condition": record["evaluation_condition"],
                             "selected_alpha": record.get("selected_alpha", ""),
                             "mse": mse, "persistence_mse": baseline[split],
                             "error_ratio": mse / baseline[split], "baseline_kind": baseline_kind,
                             "observed_scalar_targets": record[split]["observations"]})
    if len(rows) != 30:
        raise ValueError(f"Expected 24 model points and 6 baseline points, found {len(rows)}")
    return rows


def build_transfer():
    name = "fig8_transfer"
    rows = transfer_rows()
    source_csv(name, rows)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.55))
    fig.subplots_adjust(left=.085, right=.985, bottom=.31, top=.84, wspace=.40)
    limits = {"EPA-Air": (.73, 46), "FNSPID": (.77, 1.12), "Fidel": (.50, 1.12)}
    ticks = {"EPA-Air": [1, 3, 10, 30], "FNSPID": [.8, .9, 1, 1.1],
             "Fidel": [.5, .6, .8, 1]}
    for letter, ax, dataset in zip("abc", axes, ("EPA-Air", "FNSPID", "Fidel")):
        selected = [r for r in rows if r["dataset"] == dataset and r["configuration"] != "persistence"]
        configurations = list(dict.fromkeys(r["configuration"] for r in selected))
        assert np.all(np.asarray([r["error_ratio"] for r in selected]) > 0), "Log-scale positive guard"
        ax.set_yscale("log")
        ax.set_ylim(*limits[dataset])
        ax.yaxis.set_major_locator(FixedLocator(ticks[dataset]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        style_axis(ax, grid="y")
        ax.axhline(1, color=GRAY, linestyle=(0, (3, 3)), linewidth=.9, zorder=2)
        offsets = (np.arange(len(configurations)) - (len(configurations) - 1) / 2) * .055
        for configuration, offset in zip(configurations, offsets):
            values = {r["split"]: r["error_ratio"] for r in selected
                      if r["configuration"] == configuration}
            marker, linestyle = CONFIGURATION_STYLE[configuration]
            color = ENCODER_COLORS[configuration]
            is_comparator = configuration in ("numeric_only", "event_structure")
            ax.plot(np.asarray([0, 1]) + offset, [values["val"], values["test"]], color=color,
                    linestyle=linestyle, linewidth=1.25, marker=marker, markersize=4.5,
                    markerfacecolor="white" if is_comparator else color,
                    markeredgewidth=.9, zorder=3 if is_comparator else 4)
        ax.set_xlim(-.21, 1.21)
        ax.set_xticks([0, 1], ["Validation", "Test"])
        panel_label(ax, letter, "Fidel-TS" if dataset == "Fidel" else dataset)
    axes[0].set_ylabel("MSE / split persistence")
    handles = []
    for configuration in (*ENCODERS, "numeric_only", "event_structure", "published_weather"):
        marker, linestyle = CONFIGURATION_STYLE[configuration]
        color = ENCODER_COLORS[configuration]
        label = "Weather (Fidel-TS)" if configuration == "published_weather" else DISPLAY_NAMES[configuration]
        handles.append(Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                              linewidth=1.25, markersize=4.5, label=label,
                              markerfacecolor="white" if configuration in ("numeric_only", "event_structure") else color))
    handles.append(Line2D([], [], color=GRAY, linestyle=(0, (3, 3)), linewidth=.9,
                          label="Persistence = 1"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .185),
               ncol=4, handlelength=2, columnspacing=1.3, handletextpad=.55, fontsize=8)
    fig.text(.53, .027, "Log scale; panel ranges differ",
             ha="center", va="bottom", fontsize=8, color=INK)
    return save_figure(fig, name, sources=TRANSFER_SOURCES,
                       note="24 saved model/split means plus 6 persistence references; ratios use split-specific persistence. "
                            "EPA-Air: validation gains fail to transfer; FNSPID: later test gains do not retrospectively "
                            "qualify validation; Fidel: both configurations improve in both splits. "
                            "Panel-specific log-axis ranges; small horizontal offsets separate configurations within "
                            "categorical splits. Positive finite guard, no new uncertainty or gate decision.")


def map_records():
    maps = read_csv(MAP_SOURCES[0])
    summaries = read_csv(MAP_SOURCES[1])
    if len(maps) != 140 or len(summaries) != 7:
        raise ValueError("Expected every saved map draw (140) and summary cell (7)")
    grouped = defaultdict(list)
    for row in maps:
        value = float(row["loss_difference"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite saved map effect")
        if not math.isclose(value, float(row["map_mse"]) - float(row["aligned_mse"]),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved altered-minus-aligned arithmetic mismatch")
        grouped[(row["dataset"], row["representation"])].append(row)
    indexed = {(r["dataset"], r["representation"]): r for r in summaries}
    if set(grouped) != set(indexed) or len(indexed) != 7:
        raise ValueError("Map and summary populations disagree")
    for key, group in grouped.items():
        summary = indexed[key]
        if len(group) != 20 or {int(r["base_seed"]) for r in group} != set(range(120001, 120021)):
            raise ValueError(f"Incomplete saved map draws: {key}")
        values = np.asarray([float(r["loss_difference"]) for r in group])
        for field, expected in (("mean_loss_difference", values.mean()),
                                ("empirical_q025_loss_difference", np.quantile(values, .025)),
                                ("empirical_q975_loss_difference", np.quantile(values, .975))):
            if not math.isclose(float(summary[field]), float(expected), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Saved summary arithmetic mismatch: {key}/{field}")
        if int(summary["positive_map_count"]) != int(np.sum(values > 0)):
            raise ValueError(f"Saved positive-map count mismatch: {key}")
    return maps, summaries, grouped, indexed


def map_strip(ax, dataset, grouped, indexed):
    all_values = []
    labels = []
    # The small offsets only separate marks; they are not seed or temporal axes.
    offsets = np.tile(np.asarray([-.10, -.05, 0, .05, .10]), 4)
    for position, rep in enumerate(ENCODERS):
        group = sorted(grouped[(dataset, rep)], key=lambda r: int(r["base_seed"]))
        summary = indexed[(dataset, rep)]
        values = np.asarray([float(r["loss_difference"]) for r in group])
        all_values.extend(values)
        color = ENCODER_COLORS[rep]
        ax.scatter(values, position + offsets, s=13, color=color, edgecolors="white",
                   linewidths=.35, alpha=.85, zorder=3)
        center = position + .30
        low, high = (float(summary[f"empirical_q{quantile}_loss_difference"])
                     for quantile in ("025", "975"))
        ax.hlines(center, low, high, color=color, linewidth=1.6, zorder=4)
        ax.plot(float(summary["mean_loss_difference"]), center, marker="D", markersize=4.6,
                color=color, markeredgecolor="white", markeredgewidth=.45, zorder=5)
        labels.append(f"{DISPLAY_NAMES[rep]}\n{summary['positive_map_count']} / 20 positive")
    extent = max(all_values) - min(all_values)
    ax.set_xlim(min(min(all_values), 0) - .12 * extent, max(max(all_values), 0) + .12 * extent)
    ax.set_yticks(range(3), labels)
    ax.set_ylim(2.60, -.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style_axis(ax, grid="x")
    ax.axvline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=.9, zorder=2)
    if dataset == "FNSPID":
        ax.xaxis.set_major_locator(FixedLocator([-.002, -.001, 0, .001]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:g}"))
        ax.set_xlabel("ΔMSE (× 0.001)")
    else:
        ax.xaxis.set_major_locator(FixedLocator([-.15, -.10, -.05, 0, .05]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.set_xlabel("ΔMSE")


def build_maps():
    name = "fig9_map_sensitivity"
    maps, summaries, grouped, indexed = map_records()
    source_csv(name, [{"row_kind": "Map draw", **r} for r in maps]
                     + [{"row_kind": "Saved summary", **r} for r in summaries])
    fig = plt.figure(figsize=(WIDTH, 4.50))
    grid = fig.add_gridspec(2, 2, left=.16, right=.985, bottom=.16, top=.885,
                           wspace=.53, hspace=.72)
    left_top = fig.add_subplot(grid[0, 0])
    left_bottom = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[:, 1])
    for ax, letter, dataset in ((left_top, "a", "FNSPID"), (left_bottom, "b", "EPA-Air")):
        map_strip(ax, dataset, grouped, indexed)
        panel_label(ax, letter, dataset)
    summary = indexed[("Fidel", "published_weather")]
    values = np.sort([float(r["loss_difference"]) for r in grouped[("Fidel", "published_weather")]])
    if not np.all(np.diff(values) >= 0):
        raise ValueError("ECDF effects must be ordered by magnitude")
    high = max(values) * 1.14
    low_q, high_q = (float(summary[f"empirical_q{quantile}_loss_difference"])
                     for quantile in ("025", "975"))
    right.axvspan(low_q, high_q, color=PALE_BLUE, linewidth=0, zorder=0)
    right.step(np.r_[0, values, high], np.r_[0, np.arange(1, 21) / 20, 1],
               where="post", color=BLUE, linewidth=1.7, zorder=3)
    right.scatter(values, np.arange(1, 21) / 20, s=11, color=BLUE,
                  edgecolors="white", linewidths=.3, zorder=4)
    right.axvline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=.9, zorder=2)
    right.set_xlim(-.0015, high)
    right.set_ylim(-.025, 1.035)
    right.set_xticks([0, .02, .04, .06])
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    right.set_yticks([0, .25, .5, .75, 1])
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    right.set_xlabel("ΔMSE (altered − aligned)")
    right.set_ylabel("Fraction of evaluated maps")
    right.text(.06, .94, f"{summary['positive_map_count']} / 20 positive", transform=right.transAxes,
               ha="left", va="top", fontsize=8, color=INK)
    right.text(.06, .60, "Mean ΔMSE\n" + f"{float(summary['mean_loss_difference']):.6f}",
               transform=right.transAxes, ha="left", va="center", fontsize=8, color=INK)
    style_axis(right, grid="y")
    panel_label(right, "c", "Fidel-TS")
    fig.text(.54, .030, "Dots: maps; diamonds: mean; bars/shading: empirical 2.5–97.5% range.",
             ha="center", fontsize=8, color=INK)
    return save_figure(fig, name, sources=MAP_SOURCES,
                       note="All 140 map draws and all 7 saved summary cells retained. "
                            "FNSPID and EPA-Air strips use raw altered-minus-aligned MSE with separate dataset axes "
                            "(FNSPID tick units ×0.001); vertical jitter is display-only. "
                            "Fidel ECDF orders the 20 effects by magnitude, not seed/time. "
                            "Bars and shading show saved empirical 2.5–97.5 percentiles, not confidence intervals; "
                            "Fidel maps share weather context across all nine plants.")


def build_all():
    """Return the two export/provenance records for the shared figure entrypoint."""
    if not all((ROOT / source).is_file() for source in (*TRANSFER_SOURCES, *MAP_SOURCES)):
        raise FileNotFoundError("A saved transfer or map source is absent")
    OUT.mkdir(parents=True, exist_ok=True)
    return [build_transfer(), build_maps()]


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
