"""Build the eight claim-led PRISM figures from saved evidence with Matplotlib."""
import argparse
import importlib
import json
from figure_style import ROOT, OUT, digest

MODULES = ["figures_protocol", "figures_evidence", "figures_transfer_maps", "figures_boundaries"]
ORDER = ["fig1_estimands", "fig7_construct", "fig5_fidel", "fig3_calibration",
         "fig4_fixed_model", "fig8_transfer", "fig9_map_sensitivity", "fig2_coverage"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=MODULES, action="append", help="Rebuild only selected owned figure modules")
    args = parser.parse_args()
    for name in args.module or MODULES:
        importlib.import_module(name).build_all()
    records = []
    for name in ORDER:
        path = OUT / (name + ".provenance.json")
        if not path.is_file():
            raise RuntimeError(f"Missing exported main figure: {name}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    sources = {r["path"]: r["sha256"] for item in records for r in item["source_files"]}
    manifest = {
        "backend": "Python/Matplotlib", "main_figure_count": len(ORDER),
        "main_figures": ORDER, "figures": records,
        "source_files": [{"path": p, "sha256": h} for p, h in sorted(sources.items())],
        "scripts": [{"path": "scripts/" + name, "sha256": digest(ROOT / "scripts" / name)}
                    for name in ["make_journal_figures.py", "figure_style.py", "figure_alignment.py"] + [m + ".py" for m in MODULES]],
        "no_empirical_fitting_or_new_tests": True,
    }
    (OUT / "figure_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(ORDER)} main figures with editable vector text and traceable sources")


if __name__ == "__main__":
    main()
