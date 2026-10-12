"""Shared Matplotlib publication style and traceable vector exports."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/figures"
WIDTH = 7.2
BLUE = "#2C7298"
ORANGE = "#C48346"
TEAL = "#338F86"
PURPLE = "#877BA9"
GRAY = "#788691"
INK = "#253541"
LIGHT = "#E8EEF2"
PALE_BLUE = "#EDF4F8"
PALE_ORANGE = "#FBF2E8"
ENCODER_COLORS = {"gpt2": BLUE, "minilm": TEAL, "finbert": PURPLE,
                  "published_weather": BLUE, "numeric_only": GRAY,
                  "event_structure": INK}
DISPLAY_NAMES = {"gpt2": "GPT-2", "minilm": "MiniLM", "finbert": "FinBERT",
                 "published_weather": "Weather", "numeric_only": "Numerical",
                 "event_structure": "Numerical/timing"}

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "axes.edgecolor": "#74808B",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#52616D", "ytick.color": "#52616D",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": .65, "ytick.major.width": .65,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

from figure_alignment import require_matplotlib_panel_alignment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    assert rows, "Empty figure source data"
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / (name + "_source.csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax, letter, title=None):
    ax.set_gid(letter)
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-18, 11),
                textcoords="offset points", fontsize=11, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)
    if title:
        ax.set_title(title, loc="left", pad=12, fontweight="bold")


def style_axis(ax, grid=None):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=.7)
    ax.tick_params(pad=4)
    return ax


def save_figure(fig, name, *, sources=(), note="", alignment_kw=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    opts = dict(alignment_kw or {})
    report = require_matplotlib_panel_alignment(
        fig, json_out=OUT / (name + ".alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=len(fig.axes) > 1, strict=True, **opts)
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".svg"))
    fig.savefig(OUT / (name + ".png"), dpi=350)
    record = {
        "name": name, "pdf": name + ".pdf", "panels": len(report["layout"]["panels"]),
        "width_in": float(fig.get_size_inches()[0]), "height_in": float(fig.get_size_inches()[1]),
        "source_files": [{"path": str(path), "sha256": digest(ROOT / path)}
                         for path in dict.fromkeys(sources)],
        "source_csv": name + "_source.csv" if (OUT / (name + "_source.csv")).exists() else None,
        "note": note, "all_observations_preserved": True,
    }
    (OUT / (name + ".provenance.json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    return record
