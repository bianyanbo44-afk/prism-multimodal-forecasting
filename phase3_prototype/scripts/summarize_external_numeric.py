"""Summarize the frozen single-seed numerical external-development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RUNS = {
    "ClusterTrace": "results/clustertrace_development_seed101_cuda/results.json",
    "CESNET": "results/cesnet_development_seed101_cuda/results.json",
    "EPA-Air": "results/epa_air_development_seed101_cuda/results.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/external_numeric_development_summary.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/external_numeric_development_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for dataset, relative in DEFAULT_RUNS.items():
        path = (args.root / relative).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = {item["model"]: item for item in payload["results"]}
        last = float(completed["last_value"]["test_metrics"]["mse"])
        gru = float(
            completed["set_decay_gru_matched"]["test_metrics"]["mse"]
        )
        selective = float(
            completed["selective_eventmamba_no_reliability"]["test_metrics"][
                "mse"
            ]
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": 101,
                "last_value_test_mse": last,
                "matched_gru_test_mse": gru,
                "selective_ssm_test_mse": selective,
                "selective_vs_gru_percent": 100.0 * (gru - selective) / gru,
                "selective_vs_last_percent": 100.0 * (last - selective) / last,
                "selective_parameters": int(
                    completed["selective_eventmamba_no_reliability"]["parameters"]
                ),
                "gru_parameters": int(
                    completed["set_decay_gru_matched"]["parameters"]
                ),
                "source": str(path),
            }
        )

    verdict = (
        "FAIL: the numeric-only selective SSM does not beat the matched strong "
        "GRU on any of the three untouched Time-IMM datasets. Do not spend "
        "multi-seed compute on this frozen numeric-only configuration."
    )
    output = {
        "gate": "single-seed numerical external development",
        "status": "fail",
        "decision": verdict,
        "rows": rows,
    }
    json_path = (args.root / args.json_output).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# External numerical development gate",
        "",
        "This is a **development diagnostic**, not final multi-seed evidence.",
        "All runs use seed 101, train-only normalization, the official retained",
        "Time-IMM windows, a residual last-value decoder, and parameter-matched",
        "SetDecayGRU versus the selective continuous-time SSM.",
        "",
        "| Dataset | Last value MSE | Matched GRU MSE | Selective SSM MSE | Selective vs GRU | Selective vs last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {last_value_test_mse:.6f} | "
            "{matched_gru_test_mse:.6f} | {selective_ssm_test_mse:.6f} | "
            "{selective_vs_gru_percent:+.2f}% | "
            "{selective_vs_last_percent:+.2f}% |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            verdict,
            "",
            "The next gate must test the actual multimodal hypothesis with frozen",
            "text embeddings and numeric-only, text-only, naive-fusion, matched-GRU",
            "fusion, and selective-SSM fusion controls. A positive claim cannot be",
            "made from these numerical results.",
        ]
    )
    markdown_path = (args.root / args.output).resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
