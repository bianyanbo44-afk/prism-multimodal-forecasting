"""Verify the public artifact and replay generated evidence with NumPy only."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "paper_rewriting_output/journal_extension_20261012"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(manifest):
    rows = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    for row in rows:
        p = ROOT / row["path"]
        assert p.is_file(), f"Missing file: {row['path']}"
        assert p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], row["path"]
    return len(rows)


def obtain_synthetic():
    manifest = json.loads((ROOT / "SYNTHETIC_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    archive = ROOT / "PRISM_Synthetic_Evidence.zip"
    if not archive.is_file():
        print("Downloading generated evidence from the versioned GitHub release", flush=True)
        urllib.request.urlretrieve(manifest["archive_url"], archive)
    allowed = {row["path"]: row for row in manifest["files"]}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(allowed) | {"SYNTHETIC_EVIDENCE_MANIFEST.json"}
        for name, row in allowed.items():
            data = z.read(name)
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], name
            target = (ROOT / name).resolve()
            assert target.is_relative_to(ROOT.resolve()), name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-only", action="store_true", help="Check the committed train/select/test example without a download.")
    parser.add_argument("--download", action="store_true", help="Download and verify the versioned synthetic-evidence asset first.")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.download:
        obtain_synthetic()
    result = {"public_files_verified": verify_files(ROOT / "PUBLIC_MANIFEST.json")}
    spec = importlib.util.spec_from_file_location("saved_prism_evidence", EXT / "scripts/replay_saved_results.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    if not args.example_only:
        result["synthetic_arrays_verified"] = verify_files(ROOT / "SYNTHETIC_EVIDENCE_MANIFEST.json")
        result["calibration"] = replay.replay_calibration()
        result["construct_controls"] = replay.replay_construct()
    result["end_to_end_example"] = replay.end_to_end_example()
    result.update({"all_checks_passed": True, "seconds": time.perf_counter() - start,
                   "scope": "Generated-data replay and one train/validation-selected/test example. Natural benchmark inference requires upstream acquisition; encoder and empirical clock generation are not rerun."})
    (ROOT / "PUBLIC_REPLAY_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
