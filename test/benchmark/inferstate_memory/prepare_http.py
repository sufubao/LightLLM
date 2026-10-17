"""Create an isolated source snapshot for the HTTP ownership experiment.

Usage: python test/benchmark/inferstate_memory/prepare_http.py REPO EMPTY_DEST
No production files or installed Python environment are modified.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if destination.exists() and any(destination.iterdir()):
    raise RuntimeError(f"Destination must be empty: {destination}")
destination.mkdir(parents=True, exist_ok=True)
tracked = [p.decode() for p in subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).split(b"\0") if p]
files = tracked + [str(p.relative_to(source)) for p in (source / "test/benchmark/inferstate_memory").glob("*.py")]
for relative in set(files):
    src = source / relative
    dest = destination / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest, follow_symlinks=False)
probe = destination / "test/benchmark/inferstate_memory/service_probe.py"
shutil.copy2(probe, destination / "_inferstate_http_probe.py")
model = destination / "lightllm/common/basemodel/basemodel.py"
model.write_text(
    model.read_text()
    + '\n\nif os.environ.get("INFERSTATE_HTTP_PROBE_DIR"):\n'
    + "    from _inferstate_http_probe import install as _install_memory_probe\n"
    + "    _install_memory_probe(TpPartBaseModel)\n"
)
metadata = {
    "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "tracked_diff_sha256": hashlib.sha256(
        subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    ).hexdigest(),
    "files_sha256": {
        relative: hashlib.sha256((destination / relative).read_bytes()).hexdigest()
        for relative in files
        if (destination / relative).is_file()
    },
}
(destination / "experiment_source.json").write_text(json.dumps(metadata, indent=2))
print(destination)
