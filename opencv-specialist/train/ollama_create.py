"""Register a GGUF with the Ollama server through its HTTP API, and record the matching Modelfile.

The server is the Windows Ollama, reached from WSL through mirrored networking. Uploading the blob
over the API avoids translating Linux paths for ollama.exe. TEMPLATE, SYSTEM and PARAMETERs come from
train.config, the same source the training run asserted its chat template against (Rule 7).

Run from opencv-specialist/:  python -m train.ollama_create <gguf> <model-name> --base <BASE>
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import httpx

from train.config import BASES, OLLAMA, OLLAMA_PARAMETERS, SYSTEM, modelfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("name")
    ap.add_argument("--base", required=True, choices=sorted(BASES))
    a = ap.parse_args()

    # Rule 6: if Unsloth wrote its own Modelfile, keep it and show whether its TEMPLATE agrees with ours.
    ours, theirs = a.gguf.parent / "Modelfile", a.gguf.parent / "Modelfile.unsloth"
    if ours.exists() and not theirs.exists():
        ours.rename(theirs)
    if theirs.exists():
        m = re.search(r'TEMPLATE """(.*?)"""', theirs.read_text(), re.S)
        same = m is not None and m.group(1) == BASES[a.base]["ollama_template"]
        print(f"Unsloth Modelfile {theirs}: TEMPLATE {'matches' if same else 'DIFFERS from'} train.config")
    else:
        print("Unsloth wrote no Modelfile")
    ours.write_text(modelfile(a.base, a.gguf.name))

    digest = sha256(a.gguf)
    with httpx.Client(base_url=OLLAMA, timeout=None) as c:
        if c.head(f"/api/blobs/{digest}").status_code != 200:
            with open(a.gguf, "rb") as f:
                c.post(f"/api/blobs/{digest}", content=f).raise_for_status()
        r = c.post("/api/create", json={
            "model": a.name, "files": {a.gguf.name: digest}, "template": BASES[a.base]["ollama_template"],
            "system": SYSTEM, "parameters": OLLAMA_PARAMETERS, "stream": False})
        r.raise_for_status()
    print(f"created Ollama model {a.name!r} from {a.gguf} ({a.gguf.stat().st_size / 2**30:.2f} GiB): {r.json()}")


if __name__ == "__main__":
    main()
