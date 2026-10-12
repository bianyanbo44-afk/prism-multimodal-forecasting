"""Verify frozen SemAudit evidence and rebuild the anonymous paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DIR = PROJECT_ROOT.parent / "paper_rewriting_output" / "final_paper"

FROZEN_SUMMARIES = {
    "fidel_positive_control_summary.json":
        "a47c019566d3af7ef1e939bfc201c53e726f7545c93b8614748dfa2170d68cc7",
    "three_seed_shortcut_matrix.json":
        "e04b947640934df66dca17fd70e0d6f51be0048a903ea90284a79b70e3562d96",
    "immtsf_fusion_audit_summary.json":
        "7ddaccdc9a326c8251aa40ad936fb3ae736984d1993e77395c4c3833d3e1ac3e",
    "immtsf_xattn_audit_summary.json":
        "7f84d5652fbfabed28d6a9fd4386462c582f3d90c48df2ef88086fe8938f0110",
    "block_bootstrap_report.json":
        "524bf987643e032df27eb91610d91c4e1dd0bb0ff6c046d2c2b84566a701bf24",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_frozen_summaries() -> None:
    results_dir = PROJECT_ROOT / "results"
    print("[1/4] Verifying frozen evidence", flush=True)
    for name, expected in FROZEN_SUMMARIES.items():
        path = results_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen summary: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
            )
        json.loads(path.read_text(encoding="utf-8"))
        print(f"  OK {name}  {observed[:12]}...", flush=True)


def compile_latex(paper_dir: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        print("[4/4] LaTeX tools not found on PATH; manuscript compilation skipped.")
        return
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], paper_dir)
    run([bibtex, "main"], paper_dir)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], paper_dir)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], paper_dir)
    output = paper_dir / "main.pdf"
    if not output.is_file() or output.stat().st_size < 10_000:
        raise RuntimeError("LaTeX completed without a plausible main.pdf")
    print(f"  OK {output} ({output.stat().st_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-latex", action="store_true")
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    figure_script = paper_dir / "figures" / "make_paper_figures.py"
    if not (paper_dir / "main.tex").is_file():
        raise FileNotFoundError(f"Paper source not found: {paper_dir / 'main.tex'}")

    verify_frozen_summaries()

    if args.skip_tests:
        print("[2/4] Unit tests skipped")
    else:
        print("[2/4] Running correctness tests", flush=True)
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], PROJECT_ROOT)

    if args.skip_figures:
        print("[3/4] Figure regeneration skipped")
    else:
        if not figure_script.is_file():
            raise FileNotFoundError(f"Figure script not found: {figure_script}")
        print("[3/4] Rebuilding paper figures from frozen summaries", flush=True)
        run([sys.executable, str(figure_script), "--refresh-data"], paper_dir)

    if args.skip_latex:
        print("[4/4] LaTeX compilation skipped")
    else:
        print("[4/4] Compiling manuscript", flush=True)
        compile_latex(paper_dir)

    print("SemAudit artifact reproduction completed successfully.", flush=True)


if __name__ == "__main__":
    main()
