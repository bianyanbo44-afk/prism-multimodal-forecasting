"""Refresh public file identities after an intentional artifact update."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
        continue
    name = path.relative_to(ROOT).as_posix()
    if name in ("PUBLIC_MANIFEST.json", "PUBLIC_REPLAY_RESULT.json", "PRISM_Synthetic_Evidence.zip"):
        continue
    if "/results/raw/" in name and path.suffix == ".npz" and not name.endswith("/construct_controls/results/raw/world_4001.npz"):
        continue
    if path.suffix in (".py", ".md", ".tex", ".json", ".txt", ".csv", ".yml"):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for pattern in (r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{40,}",
                        r"sk-(?:proj-)?[A-Za-z0-9_-]{40,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"):
            assert not re.search(pattern, text), f"Credential-shaped literal in {name}"
    rows.append({"path": name, "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(ROOT / "PUBLIC_MANIFEST.json").write_text(json.dumps({"version": "v0.1.0", "date": "2026-10-13", "files": rows,
    "scope": "Public code, manuscript, generated states, reporting summaries and one committed training/selection example; additional generated arrays are a release asset."}, indent=2) + "\n", encoding="utf-8")
print("Public manifest:", len(rows), "files;", round(sum(x["bytes"] for x in rows) / 2**20, 2), "MiB; credential-shaped literal scan passed")
