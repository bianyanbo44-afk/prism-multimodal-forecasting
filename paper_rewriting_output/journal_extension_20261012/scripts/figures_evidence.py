"""Evidence figures from saved calibration, checkpoint, and photovoltaic results.

Each two-panel figure answers one question: sensitivity to known signal;
reassignment versus zero-content response; or fixed versus refitted response.
All displayed intervals are saved results. Plant points describe heterogeneity
under shared weather, and have no plant-replication confidence intervals.
"""

import json

from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from figure_style import (
    BLUE, GRAY, INK, LIGHT, ORANGE, PURPLE, TEAL, WIDTH,
    panel_label, plt, read_csv, read_json, save_figure, source_csv, style_axis,
)


def _asymmetric_errors(centres, lower, upper):
    centres = np.asarray(centres, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    assert np.all(np.isfinite(np.r_[centres, lower, upper]))
    assert np.all(lower <= centres + 1e-12)
    assert np.all(upper >= centres - 1e-12)
    return np.maximum(0.0, np.stack((centres - lower, upper - centres)))


def calibration_figure():
    """Known-signal effects and joint detection, including both null cells."""
    path = "calibration/results/summary.json"
    rows = read_json(path)["conditions"]
    strengths = [0.0, 0.25, 0.5, 1.0]
    lookup = {(r["representation"], float(r["strength"])): r for r in rows}
    assert len(rows) == len(lookup) == 8
    assert set(lookup) == {(r, x) for r in ("oracle", "sentence") for x in strengths}
    assert all(r["worlds"] == 20 for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.55))
    # Preserve the previous 2.1525-inch plot height while shortening the footer.
    fig.subplots_adjust(left=.105, right=.975, bottom=.20,
                        top=.20 + 2.1525 / 3.55, wspace=.40)
    series = [("oracle", "Oracle directions", GRAY, "s"),
              ("sentence", "MiniLM text", BLUE, "o")]
    for rep_index, (rep, label, color, marker) in enumerate(series):
        selected = [lookup[rep, x] for x in strengths]
        centre = [r["mean_fixed_shuffle_delta"] for r in selected]
        interval = np.asarray([r["mean_delta_world_ci95"] for r in selected])
        offset = -.008 if rep_index == 0 else .008
        axes[0].errorbar(
            np.asarray(strengths) + offset, centre,
            yerr=_asymmetric_errors(centre, interval[:, 0], interval[:, 1]),
            color=color, marker=marker, markersize=4.8, linewidth=1.4,
            elinewidth=1.0, capsize=3, capthick=.9, label=label, zorder=3,
        )

        for screened in (False, True):
            rate_key = "gated_detection_rate" if screened else "detection_rate"
            interval_key = "gated_rate_wilson95" if screened else "detection_rate_wilson95"
            rate = [r[rate_key] for r in selected]
            interval = np.asarray([r[interval_key] for r in selected])
            # Offsets separate four measured series; they do not change signal strength.
            offset = (-.050 if rep_index == 0 else .020) + (.030 if screened else 0)
            axes[1].errorbar(
                np.asarray(strengths) + offset, rate,
                yerr=_asymmetric_errors(rate, interval[:, 0], interval[:, 1]),
                color=color, marker=marker, markerfacecolor="white" if screened else color,
                markersize=4.8, markeredgewidth=1.1, linestyle="none",
                elinewidth=.9, capsize=2.3, capthick=.8, zorder=3,
            )

    for ax in axes:
        ax.set_xlim(-.12, 1.12)
        ax.set_xticks(strengths, ["0 (null)", "0.25", "0.5", "1"])
        ax.set_xlabel("Known content-signal strength γ", labelpad=7)
        style_axis(ax, grid="y")
    axes[0].axhline(0, color=GRAY, linewidth=.7, zorder=1)
    axes[0].set_ylim(-.055, .94)
    axes[0].set_ylabel("Fixed-shuffle minus aligned MSE")
    axes[0].set_yticks([0, .2, .4, .6, .8])
    axes[1].set_ylim(-.065, 1.13)
    axes[1].set_ylabel("Worlds detected")
    axes[1].set_yticks([0, .25, .5, .75, 1])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].text(.45, .59, "Nonzero signal: 20/20\nfor both representations",
                 transform=axes[1].transAxes, fontsize=8, ha="left", va="center")
    axes[1].text(.45, .38, "Unscreened null: 3/20, 2/20\nScreened null: 0/20 for both",
                 transform=axes[1].transAxes, fontsize=8, ha="left", va="center")
    axes[1].legend(
        handles=[Line2D([], [], color=INK, marker="o", markersize=4.5,
                        linestyle="none", label="Unscreened"),
                 Line2D([], [], color=INK, marker="o", markersize=4.5,
                        markerfacecolor="white", linestyle="none", label="Screened")],
        loc="lower right", fontsize=8, borderaxespad=.8,
    )
    fig.legend(
        handles=[Line2D([], [], color=color, marker=marker, linewidth=1.4,
                        markersize=4.8, label=label) for _, label, color, marker in series],
        loc="upper center", bbox_to_anchor=(.52, .965), ncol=2, columnspacing=2.4,
        fontsize=8,
    )
    panel_label(axes[0], "a", "Loss response to known signal")
    panel_label(axes[1], "b", "Null and signal detection")
    fig.text(.52, .035,
             "Each detection fraction uses all 20 worlds, including screened detection.",
             ha="center", fontsize=8)

    source_csv("fig3_calibration", [{
        "representation": r["representation"], "strength": r["strength"],
        "worlds": r["worlds"], "optimizer_seeds_per_world": r["training_seeds_per_world"],
        "mean_fixed_shuffle_delta": r["mean_fixed_shuffle_delta"],
        "effect_ci95_low": r["mean_delta_world_ci95"][0],
        "effect_ci95_high": r["mean_delta_world_ci95"][1],
        "detected_worlds": r["detected_worlds"], "detection_rate": r["detection_rate"],
        "wilson95_low": r["detection_rate_wilson95"][0],
        "wilson95_high": r["detection_rate_wilson95"][1],
        "capability_qualified_worlds": r["capability_qualified_worlds"],
        "gated_detected_worlds": r["gated_detected_worlds"],
        "gated_detection_rate": r["gated_detection_rate"],
        "gated_wilson95_low": r["gated_rate_wilson95"][0],
        "gated_wilson95_high": r["gated_rate_wilson95"][1],
    } for r in rows])
    return save_figure(
        fig, "fig3_calibration", sources=[path],
        note="All eight cells retained. Effect intervals are saved 20-world t intervals; "
             "detection intervals are saved Wilson intervals with denominator 20. "
             "Screened detection is a joint fraction, not conditional null error among passing worlds.",
    )


def fixed_model_figure():
    """All 15 historical cells per control, with matched percentage intervals."""
    path = "fixed_model/results/paired_summary_control_denominator.csv"
    rows = [r for r in read_csv(path) if r["condition"] in ("derangement_100", "zero_content")]
    datasets = ["ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"]
    mechanisms = [("selective_ssm", "Selective", BLUE, "o"),
                  ("immtsf_recavg_gradd", "RecAvg", TEAL, "s"),
                  ("immtsf_t2v_xattn", "XAttn", PURPLE, "D")]
    controls = [("derangement_100", "Reassignment (SAG)"),
                ("zero_content", "Zero content (CBTG)")]
    lookup = {(r["dataset"], r["architecture"], r["condition"]): r for r in rows}
    assert len(rows) == len(lookup) == 30
    assert set(lookup) == {(d, a, c) for d in datasets
                           for a, _, _, _ in mechanisms for c, _ in controls}

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9), sharey=True, sharex=True)
    # Preserve the previous 2.4975-inch plot height.
    fig.subplots_adjust(left=.175, right=.975, bottom=.18,
                        top=.18 + 2.4975 / 3.9, wspace=.25)
    for ax, (control, title), letter in zip(axes, controls, ("a", "b")):
        for index, (mechanism, _, color, marker) in enumerate(mechanisms):
            selected = [lookup[d, mechanism, control] for d in datasets]
            effect = [float(r["gain_percent_control_denominator"]) for r in selected]
            low = [float(r["gain_control_ci95_block2_low"]) for r in selected]
            high = [float(r["gain_control_ci95_block2_high"]) for r in selected]
            y = np.arange(len(datasets)) + (index - 1) * .235
            ax.errorbar(
                effect, y, xerr=_asymmetric_errors(effect, low, high),
                color=color, marker=marker, markersize=4.7, linestyle="none",
                elinewidth=1.05, capsize=2.5, capthick=.9, zorder=3,
            )
        ax.axvline(0, color=GRAY, linewidth=.85, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(-16.5, 36.5)
        ax.set_xticks([-10, 0, 10, 20, 30])
        ax.set_xlabel("Control-denominator gain (%)", labelpad=7)
        style_axis(ax, grid="x")
        panel_label(ax, letter, title)
    axes[0].set_yticks(np.arange(len(datasets)), datasets)
    axes[0].set_ylim(4.55, -.55)
    axes[0].tick_params(axis="y", length=0, pad=7)
    axes[1].tick_params(axis="y", length=0)
    fig.legend(
        handles=[Line2D([], [], color=color, marker=marker, markersize=4.7,
                        linestyle="none", label=label)
                 for _, label, color, marker in mechanisms],
        loc="upper center", bbox_to_anchor=(.54, .96), ncol=3, columnspacing=2.2,
        fontsize=8,
    )
    fig.text(.52, .035,
             "Positive gain: altered input raises loss. Both panels use the same scale.",
             ha="center", fontsize=8)
    source_csv("fig4_fixed_model", [{
        **r, "gain_label": "SAG" if r["condition"] == "derangement_100" else "CBTG",
        "displayed_interval_block_length": 2, "checkpoints_averaged": 3,
        "interval_scope": "conditional on selected checkpoints, retained entities, and realized map",
    } for r in rows])
    return save_figure(
        fig, "fig4_fixed_model", sources=[path],
        note="All 30 cells retained: 15 dataset/mechanism cells for each of derangement and zero content. "
             "Points use gain_percent_control_denominator; intervals use the matched "
             "gain_control_ci95_block2_low/high fields, all already in percentage units. "
             "A common linear axis keeps the tiny reassignment effects in scale with zero-content responses.",
    )


def fidel_figure():
    """Aggregate object contrasts and all-nine-plant matched differences."""
    path = "external_validation/results/Fidel/results.json"
    data = read_json(path)
    lookup = {(r["training_condition"], r["evaluation_condition"]): r for r in data["results"]}
    aligned = lookup["aligned", "aligned"]["test"]
    fixed = lookup["aligned", "semantic_derangement"]["test"]
    retrained = lookup["semantic_derangement", "semantic_derangement"]["test"]
    numerical = lookup["numeric_only", "numeric_only"]["test"]
    plants = sorted(aligned["plants"])
    assert len(plants) == 9
    assert set(plants) == set(fixed["plants"]) == set(retrained["plants"])
    assert aligned["observations"] == fixed["observations"] == retrained["observations"] == 78840
    assert all(aligned["plants"][p]["observations"] == fixed["plants"][p]["observations"]
               == retrained["plants"][p]["observations"] == 8760 for p in plants)

    def gain(control):
        assert control["mse"] > 0
        return 100.0 * (control["mse"] - aligned["mse"]) / control["mse"]

    fixed_gain, retrained_gain, stream_utility = gain(fixed), gain(retrained), gain(numerical)
    configs = [
        ("aligned", "aligned", "Aligned", 5.2, INK, "o"),
        ("aligned", "semantic_derangement", "Fixed shift", 4.2, BLUE, "o"),
        ("semantic_derangement", "semantic_derangement", "Condition-retrained", 3.2, ORANGE, "s"),
        ("numeric_only", "numeric_only", "Numerical-only", 1.9, GRAY, "D"),
        ("persistence", "persistence", "Daily persistence", .9, GRAY, "D"),
        ("aligned", "zero", "Zero content", -.1, PURPLE, "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.0))
    # Preserve the previous 2.6085-inch plot height.
    fig.subplots_adjust(left=.195, right=.975, bottom=.17,
                        top=.17 + 2.6085 / 4.0, wspace=.48)
    for train, evaluation, _, y, color, marker in configs:
        result = lookup[train, evaluation]["test"]
        axes[0].scatter(result["mse"], y, color=color, marker=marker, s=27, zorder=3)
        if (train, evaluation) in (("aligned", "semantic_derangement"),
                                   ("semantic_derangement", "semantic_derangement")):
            axes[0].plot([aligned["mse"], result["mse"]], [y, y], color=color,
                         linewidth=1.8, zorder=2)
            axes[0].scatter(aligned["mse"], y, facecolor="white", edgecolor=INK,
                            s=20, zorder=3, linewidth=.8)
            axes[0].text(result["mse"] + .017, y,
                         f"{gain(result):.2f}% SAG", fontsize=9, color=color,
                         fontweight="bold", ha="left", va="center")
        elif train == "numeric_only":
            axes[0].text(result["mse"] + .017, y,
                         f"{stream_utility:.2f}% stream utility", fontsize=8, color=GRAY,
                         ha="left", va="center")
    axes[0].axvline(aligned["mse"], color=LIGHT, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
    axes[0].set_yticks([r[3] for r in configs], [r[2] for r in configs])
    axes[0].set_ylim(-.8, 5.95)
    axes[0].set_xlim(0, .54)
    axes[0].set_xticks([0, .1, .2, .3, .4, .5])
    axes[0].set_xlabel("Standardized test MSE", labelpad=7)
    axes[0].tick_params(axis="y", length=0, pad=6)
    style_axis(axes[0])

    for i, plant in enumerate(plants):
        base = aligned["plants"][plant]["mse"]
        effects = [fixed["plants"][plant]["mse"] - base,
                   retrained["plants"][plant]["mse"] - base]
        axes[1].plot(effects, [i, i], color=LIGHT, linewidth=2.0, zorder=1)
        axes[1].scatter(effects[0], i, color=BLUE, marker="o", s=29, zorder=3)
        axes[1].scatter(effects[1], i, color=ORANGE, marker="s", s=27, zorder=3)
    axes[1].axvline(0, color=GRAY, linewidth=.8, zorder=1)
    axes[1].set_xlim(-.003, .063)
    axes[1].set_xticks([0, .02, .04, .06])
    axes[1].set_yticks(np.arange(9), plants)
    axes[1].set_ylim(8.65, -.65)
    axes[1].set_xlabel("Shift minus aligned MSE", labelpad=7)
    axes[1].tick_params(axis="y", length=0, pad=6)
    style_axis(axes[1], grid="x")
    panel_label(axes[0], "a", "Aggregate losses")
    panel_label(axes[1], "b", "Plant-level contrasts")
    fig.legend(
        handles=[Line2D([], [], color=BLUE, marker="o", markersize=5, linestyle="none", label="Fixed shift"),
                 Line2D([], [], color=ORANGE, marker="s", markersize=5, linestyle="none", label="Condition-retrained")],
        loc="upper center", bbox_to_anchor=(.59, .965), ncol=2, columnspacing=2.0,
        fontsize=8,
    )
    fig.text(.52, .035,
             "Each SAG uses its own altered-input loss as denominator.",
             ha="center", fontsize=8)

    source_rows = []
    for train, evaluation, label, _, _, _ in configs:
        result = lookup[train, evaluation]["test"]
        source_rows.append({
            "panel": "a", "record_type": "aggregate_loss", "condition": label,
            "training_condition": train, "evaluation_condition": evaluation,
            "mse": result["mse"], "observed_scalar_targets": result["observations"],
            "aligned_mse": aligned["mse"],
            "gain_percent": (gain(result) if label in ("Fixed shift", "Condition-retrained", "Numerical-only") else ""),
            "gain_denominator_mse": (result["mse"] if label in ("Fixed shift", "Condition-retrained", "Numerical-only") else ""),
            "contrast_object": ("fixed_predictor" if label == "Fixed shift" else
                                "condition_retrained" if label == "Condition-retrained" else
                                "complete_stream_utility" if label == "Numerical-only" else "loss_context"),
        })
    for plant in plants:
        for label, result in (("Fixed shift", fixed), ("Condition-retrained", retrained)):
            base, altered = aligned["plants"][plant], result["plants"][plant]
            source_rows.append({
                "panel": "b", "record_type": "paired_plant_effect", "plant": plant,
                "condition": label, "aligned_mse": base["mse"], "mse": altered["mse"],
                "delta_mse": altered["mse"] - base["mse"],
                "observed_scalar_targets": base["observations"],
                "uncertainty": "none; descriptive plant heterogeneity under shared context",
            })
    source_csv("fig5_fidel", source_rows)
    return save_figure(
        fig, "fig5_fidel", sources=[path],
        note=f"Six saved aggregate loss conditions and all nine plants retained. Fixed SAG {fixed_gain:.8f}% "
             f"uses fixed-shift MSE {fixed['mse']:.12f}; condition-retrained SAG {retrained_gain:.8f}% "
             f"uses retrained-shift MSE {retrained['mse']:.12f}. Complete-stream utility {stream_utility:.8f}% "
             f"uses numerical-only MSE {numerical['mse']:.12f}. Plant effects are descriptive matched "
             "altered-minus-aligned MSE, with no plant-independence CI or repeated-map distribution.",
    )


def build_all():
    """Return three provenance records without changing other figure assets."""
    return [calibration_figure(), fixed_model_figure(), fidel_figure()]


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
