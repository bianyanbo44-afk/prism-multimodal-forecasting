"""Summarize numeric, raw-text, and layer-normalized-text development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "ClusterTrace": "clustertrace",
    "CESNET": "cesnet",
    "EPA-Air": "epa_air",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multimodal_development_summary.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/multimodal_development_summary.json"),
    )
    return parser.parse_args()


def model_mse(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["model"]: float(item["test_metrics"]["mse"])
        for item in payload["results"]
    }


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for dataset, slug in DATASETS.items():
        paths = {
            "numeric": args.root
            / "results"
            / f"{slug}_development_seed101_cuda"
            / "results.json",
            "raw_text": args.root
            / "results"
            / f"{slug}_multimodal_development_seed101_cuda"
            / "results.json",
            "layer_norm_text": args.root
            / "results"
            / f"{slug}_multimodal_ln_development_seed101_cuda"
            / "results.json",
        }
        result = {name: model_mse(path.resolve()) for name, path in paths.items()}
        numeric_gru = result["numeric"]["set_decay_gru_matched"]
        numeric_selective = result["numeric"][
            "selective_eventmamba_no_reliability"
        ]
        raw_gru = result["raw_text"]["set_decay_gru_matched"]
        raw_selective = result["raw_text"][
            "selective_eventmamba_no_reliability"
        ]
        ln_gru = result["layer_norm_text"]["set_decay_gru_matched"]
        ln_selective = result["layer_norm_text"][
            "selective_eventmamba_no_reliability"
        ]
        rows.append(
            {
                "dataset": dataset,
                "numeric_gru": numeric_gru,
                "numeric_selective": numeric_selective,
                "raw_text_gru": raw_gru,
                "raw_text_selective": raw_selective,
                "ln_text_gru": ln_gru,
                "ln_text_selective": ln_selective,
                "ln_selective_vs_ln_gru_percent": gain(ln_gru, ln_selective),
                "ln_selective_vs_numeric_selective_percent": gain(
                    numeric_selective, ln_selective
                ),
                "ln_selective_vs_numeric_gru_percent": gain(
                    numeric_gru, ln_selective
                ),
                "ln_win_over_gru": ln_selective < ln_gru,
            }
        )

    wins = sum(bool(row["ln_win_over_gru"]) for row in rows)
    payload = {
        "gate": "single-seed multimodal development diagnostic",
        "seed": 101,
        "status": "promising_not_confirmed",
        "layer_normalized_selective_wins_over_matched_gru": f"{wins}/{len(rows)}",
        "decision": (
            "Proceed to text-only and naive-fusion controls. Do not run final "
            "multi-seed confirmation until those controls and a low-resource "
            "failure analysis are complete."
        ),
        "rows": rows,
    }
    json_output = (args.root / args.json_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Multimodal development summary",
        "",
        "These are seed-101 development diagnostics. The test periods have now",
        "been inspected and cannot be described as untouched confirmation sets.",
        "",
        "| Dataset | Numeric GRU | Numeric selective | Raw-text GRU | Raw-text selective | LN-text GRU | LN-text selective | LN selective vs LN GRU | LN selective vs numeric selective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {numeric_gru:.6f} | {numeric_selective:.6f} | "
            "{raw_text_gru:.6f} | {raw_text_selective:.6f} | "
            "{ln_text_gru:.6f} | {ln_text_selective:.6f} | "
            "{ln_selective_vs_ln_gru_percent:+.2f}% | "
            "{ln_selective_vs_numeric_selective_percent:+.2f}% |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The layer-normalized selective fusion model beats the parameter-matched GRU on {wins}/3 datasets.",
            "ClusterTrace and EPA-Air support the hypothesis; CESNET, with only 23",
            "training windows and 27 training text-event occurrences, is a clear",
            "low-resource failure boundary. Per-note LayerNorm is parameter-free",
            "and uses no cross-sample or future statistics.",
            "",
            "Next: text-only, naive mean-fusion, semantic-shuffle, and final",
            "multi-seed controls. No top-conference claim is yet warranted.",
        ]
    )
    markdown_output = (args.root / args.output).resolve()
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
