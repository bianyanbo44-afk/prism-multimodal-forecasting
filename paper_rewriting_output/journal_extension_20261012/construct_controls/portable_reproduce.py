"""Reproduce the unchanged frozen implementation with relocated data/model paths.

Only path locations are overridden; semantic/experimental/model configuration
remains unchanged. A fresh output directory is mandatory. The new local freeze
records the relocated paths and original implementation/config/helper hashes.
"""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from run_construct_controls import run

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True, help="EPA-Air processed directory containing eight entity folders")
parser.add_argument("--encoder-path", type=Path, required=True, help="Pinned MiniLM snapshot directory")
parser.add_argument("--output", type=Path, required=True, help="New, unscored output directory")
args = parser.parse_args()
cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
cfg["source_root"] = str(args.source_root.resolve())
cfg["encoder_path"] = str(args.encoder_path.resolve())
with threadpool_limits(limits=cfg["threads"]):
    run(cfg,args.output.resolve())
