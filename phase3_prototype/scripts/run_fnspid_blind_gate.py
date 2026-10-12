"""Execute the frozen FNSPID gate without exposing intermediate metrics.

Every child process writes stdout/stderr to a dedicated log.  This runner only
prints step names, durations, and exit codes; result JSON is intentionally not
parsed until the full batch has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fnspid_untouched_gate"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output = (PROJECT_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    manifest_path = output / "execution_manifest.json"

    config = PROJECT_ROOT / "configs" / "timeimm_multiseed_gate.json"
    preregistration = (
        PROJECT_ROOT / "results" / "untouched_fnspid_gate_preregistration.md"
    )
    data_root = PROJECT_ROOT / "external" / "Time-IMM" / "data"
    python = Path(sys.executable).resolve()
    common = [
        "--config",
        str(config),
        "--data-root",
        str(data_root),
        "--dataset",
        "FNSPID",
        "--device",
        "cuda",
    ]
    steps: list[tuple[str, list[str]]] = [
        (
            "trivial_baselines",
            [
                str(python),
                str(PROJECT_ROOT / "scripts" / "evaluate_timeimm_trivial_baselines.py"),
                "--data-root",
                str(data_root),
                "--output",
                str(output / "trivial_baselines.json"),
                "--device",
                "cuda",
            ],
        ),
        (
            "shrinkage_baseline",
            [
                str(python),
                str(PROJECT_ROOT / "scripts" / "evaluate_timeimm_shrinkage.py"),
                "--data-root",
                str(data_root),
                "--output",
                str(output / "shrinkage_baseline.json"),
                "--device",
                "cuda",
            ],
        ),
        (
            "numeric_models",
            [
                str(python),
                str(PROJECT_ROOT / "scripts" / "run_timeimm.py"),
                *common,
                "--output",
                str(output / "numeric_models"),
            ],
        ),
        (
            "multimodal_models",
            [
                str(python),
                str(PROJECT_ROOT / "scripts" / "run_timeimm.py"),
                *common,
                "--output",
                str(output / "multimodal_models"),
                "--with-text",
                "--text-cache-name",
                "text_embeddings_model=GPT2_pool=masked_mean_maxlen=320_timebase=numeric.pt",
                "--text-preprocessing",
                "layer_norm",
            ],
        ),
        (
            "multimodal_controls",
            [
                str(python),
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "run_timeimm_multimodal_controls.py"
                ),
                *common,
                "--output",
                str(output / "multimodal_controls"),
                "--text-cache-name",
                "text_embeddings_model=GPT2_pool=masked_mean_maxlen=320_timebase=numeric.pt",
            ],
        ),
        (
            "semantic_controls",
            [
                str(python),
                str(PROJECT_ROOT / "scripts" / "run_timeimm_semantic_controls.py"),
                *common,
                "--output",
                str(output / "semantic_controls"),
                "--text-cache-name",
                "text_embeddings_model=GPT2_pool=masked_mean_maxlen=320_timebase=numeric.pt",
            ],
        ),
    ]

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "purpose": "metric-blind execution of the preregistered FNSPID gate",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": str(python),
            "config_sha256": sha256(config),
            "preregistration_sha256": sha256(preregistration),
            "intermediate_metrics_exposed": False,
            "steps": [],
        }
        write_manifest(manifest_path, manifest)

    completed = {
        str(item["name"])
        for item in manifest["steps"]
        if item.get("status") == "completed"
    }
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"

    for name, command in steps:
        if name in completed:
            print(json.dumps({"step": name, "status": "skipped_completed"}), flush=True)
            continue
        log_path = logs / f"{name}.log"
        start = time.perf_counter()
        print(json.dumps({"step": name, "status": "started"}), flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed_process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record = {
            "name": name,
            "status": (
                "completed" if completed_process.returncode == 0 else "failed"
            ),
            "returncode": completed_process.returncode,
            "duration_seconds": time.perf_counter() - start,
            "command": command,
            "log": str(log_path),
        }
        manifest["steps"].append(record)
        write_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "step": name,
                    "status": record["status"],
                    "returncode": record["returncode"],
                    "duration_seconds": record["duration_seconds"],
                }
            ),
            flush=True,
        )
        if completed_process.returncode != 0:
            raise SystemExit(completed_process.returncode)

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(json.dumps({"batch": "fnspid_untouched_gate", "status": "completed"}))


if __name__ == "__main__":
    main()
