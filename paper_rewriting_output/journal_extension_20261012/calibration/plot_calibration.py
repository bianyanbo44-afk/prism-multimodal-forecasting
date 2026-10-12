"""Render the calibration evidence and its null rejection boundary.

Figure contract: a quantitative grid; panel a shows known-signal recovery,
panel b shows the specificity limitation and frozen validation capability gate.
All eight strength/representation cells and both null rejection definitions
are retained. Figure source data are summary.json from the executed experiment.
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path("C:/Users/Lenovo/.codex/skills/nature-figure/scripts")))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False})

summary = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8"))["conditions"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
fig.subplots_adjust(left=0.095, right=0.975, bottom=0.20, top=0.88, wspace=0.42)
colors = {"oracle": "#517c9c", "sentence": "#bb805b"}
labels = {"oracle": "Oracle directions", "sentence": "MiniLM sentences"}
for r, marker in [("oracle", "o"), ("sentence", "s")]:
    rows = [z for z in summary if z["representation"] == r]
    x = np.array([z["strength"] for z in rows])
    y = np.array([z["mean_fixed_shuffle_delta"] for z in rows])
    bounds = np.array([z["mean_delta_world_ci95"] for z in rows])
    axes[0].errorbar(x, y, yerr=np.stack([y - bounds[:, 0], bounds[:, 1] - y]),
                     color=colors[r], marker=marker, markersize=4.0, linewidth=1.3,
                     capsize=2.5, label=labels[r])
axes[0].axhline(0, color="#777777", linewidth=.55)
axes[0].set(xlim=(-.04, 1.04), ylim=(-.045, .97), xticks=[0, .25, .5, 1.0],
            xlabel="Injected signal strength", ylabel="MSE increase after content permutation")
axes[0].set_title("Known-signal sensitivity", loc="left", pad=10)
axes[0].legend(loc="upper left", handlelength=1.8, borderpad=0)

for pos, r in enumerate(["oracle", "sentence"]):
    row = next(z for z in summary if z["representation"] == r and z["strength"] == 0)
    for offset, center, bound, marker, fill in [(-.11, row["detection_rate"], row["detection_rate_wilson95"], "o", "white"),
                                              (.11, row["gated_detection_rate"], row["gated_rate_wilson95"], "s", colors[r])]:
        err = np.array([[center - bound[0]], [bound[1] - center]])
        axes[1].errorbar(pos + offset, center, yerr=err, color=colors[r], marker=marker,
                         markerfacecolor=fill, markersize=4.5, linewidth=.9, capsize=3)
axes[1].axhline(.05, linestyle="--", linewidth=.8, color="#8c8c8c")
axes[1].set(xlim=(-.55, 1.55), ylim=(-.04, .52), xticks=[0, 1],
            xticklabels=["Oracle directions", "MiniLM sentences"],
            yticks=[0, .1, .2, .3, .4, .5], xlabel="Zero injected signal", ylabel="Rejected worlds / 20")
axes[1].set_title("Null rejection and capability gate", loc="left", pad=10)
axes[1].legend(handles=[Line2D([], [], color="#555555", marker="o", markerfacecolor="white", linestyle="None", label="Ungated"),
                       Line2D([], [], color="#555555", marker="s", linestyle="None", label="Capability gated")],
               loc="upper right", handlelength=1.1, borderpad=0)
for label, ax in zip(["a", "b"], axes):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-20, 12), textcoords="offset points",
                fontsize=10, fontweight="bold", va="bottom", ha="left")
base = HERE / "figures/calibration"
base.parent.mkdir(exist_ok=True)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json",
                                   overlay_svg=f"{base}.alignment.svg", require_panel_labels=True,
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(f"{base}.pdf")
fig.savefig(f"{base}.svg")
fig.savefig(f"{base}.png", dpi=300)
fig.savefig(f"{base}.tiff", dpi=600)
plt.close(fig)
