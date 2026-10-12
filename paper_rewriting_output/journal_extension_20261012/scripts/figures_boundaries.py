"""Paired intervention reach and score-scaling boundaries from saved results."""
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from figure_style import (WIDTH, BLUE, ORANGE, GRAY, INK, LIGHT, PALE_BLUE,
                          plt, np, read_json, panel_label, style_axis,
                          source_csv, save_figure)


def build_all():
    strength_path = "diagnostics/control_strength.json"
    scaling_path = "diagnostics/normalization_sensitivity.json"
    data = read_json(strength_path)
    scaling = read_json(scaling_path)["rows"]
    datasets = ["ClusterTrace", "CESNET", "EPA-Air", "FNSPID", "RepoHealth"]
    control = {(r["dataset"], r["condition"]): r for r in data["rows"] if r["split"] == "test"}
    numerical = {r["dataset"]: r for r in scaling}
    assert len(numerical) == len(datasets)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.7))
    fig.subplots_adjust(left=.16, right=.97, bottom=.29, top=.83, wspace=.29)
    rows = []
    for ax in axes:
        style_axis(ax)
        ax.set_ylim(4.65, -.7)
        ax.set_yticks(range(5), datasets)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    a, b = axes
    a.axhspan(.55, 1.45, color=PALE_BLUE, zorder=0)
    for i, dataset in enumerate(datasets):
        original = control[dataset, "released_shuffle"]
        deranged = control[dataset, "derangement"]
        percentages = [100 * (1 - r["identical_representation_fraction"]) for r in [original, deranged]]
        a.plot(percentages, [i, i], color="#C8D2D9", linewidth=1.4, zorder=2)
        a.scatter(percentages[0], i, color="white", edgecolor=GRAY, linewidth=1.1, s=34, zorder=3)
        a.scatter(percentages[1], i, color=BLUE, edgecolor="white", linewidth=.6, s=26, zorder=4)
        for condition, record, percentage in zip(["Historical shuffle", "Full derangement"], [original, deranged], percentages):
            rows.append({"panel": "coverage", "dataset": dataset, "condition": condition,
                         "changed_percent": percentage, "positions": record["positions"],
                         "changed_positions": record["positions"] - record["identical_representation_positions"]})
    a.annotate("9 of 16 positions", (56.25, 1), xytext=(0, -13), textcoords="offset points",
               ha="center", va="top", fontsize=8, color=BLUE)
    a.set_xlim(-3, 105)
    a.set_xticks([0, 25, 50, 75, 100])
    a.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    a.set_xlabel("Positions with changed representations", labelpad=8)
    panel_label(a, "a", "Achieved reassignment")
    a.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=GRAY, label="Historical shuffle"),
                      Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Full derangement")],
             loc="upper center", bbox_to_anchor=(.5, -.36), handlelength=1.0, borderaxespad=0, labelspacing=.5)
    for i, dataset in enumerate(datasets):
        record = numerical[dataset]
        train, full = record["train_only_normalized_mse"], record["full_entity_normalized_mse"]
        assert train > 0 and full >= 0
        b.plot([train, full], [i, i], color="#BBC6CE", linewidth=1.3, zorder=2)
        b.scatter(train, i, s=29, color=BLUE, edgecolor="white", linewidth=.5, zorder=3)
        b.scatter(full, i, s=27, color=ORANGE, marker="s", edgecolor="white", linewidth=.5, zorder=3)
        change = 100 * (full / train - 1)
        b.annotate(f"{change:+.1f}%", ((train + full) / 2, i), xytext=(0, 10),
                   textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
        rows.append({"panel": "scaling", "dataset": dataset, "train_only_mse": train,
                     "full_entity_mse": full, "displayed_mse_change_percent": change,
                     "eligible_windows": record["analyzed_windows"], "total_windows": record["total_windows"]})
    b.set_yticklabels([])
    b.set_xlim(0, 2.4)
    b.set_xticks([0, .5, 1, 1.5, 2])
    b.set_xlabel("Normalized persistence MSE", labelpad=8)
    panel_label(b, "b", "Same forecast, different score")
    b.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Train-only scaling"),
                      Line2D([], [], marker="s", linestyle="none", color=ORANGE, label="Full-entity scaling")],
             loc="upper center", bbox_to_anchor=(.5, -.36), handlelength=1.0, borderaxespad=0, labelspacing=.5)
    name = "fig2_coverage"
    source_csv(name, rows)
    return [save_figure(fig, name, sources=[strength_path, scaling_path],
                        note="All five datasets. Coverage counts retained text slots; paired scaling scores retain identical raw forecasts and eligible windows. Percentage score changes are descriptive arithmetic, not predictive improvements.")]


if __name__ == "__main__":
    build_all()
