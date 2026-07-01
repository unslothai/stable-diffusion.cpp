#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.

"""Generate the release metadata for an Unsloth stable-diffusion.cpp prebuild.

Writes two files next to the bundles:
  * sd-prebuilt-sha256.json  -- {asset_name: sha256_hex} for every sd-*.zip
  * sd-prebuilt-manifest.json -- tag / source / commit / per-asset {size, sha256, label}

The Studio installer already verifies each asset against the GitHub-provided
``digest``; the sha256 index is a second, self-hosted integrity source and the
manifest is the authoritative bundle set (mirrors llama.cpp's prebuilt manifest).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

# Map an asset filename back to a host label, for the manifest (informational).
_LABEL_RE = re.compile(r"^sd-.*-bin-(?P<label>.+)\.zip$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required = True)
    ap.add_argument("--source-repo", required = True)
    ap.add_argument("--commit", required = True)
    ap.add_argument("--dist", required = True, help = "dir holding the sd-*.zip bundles")
    ap.add_argument("--out", required = True, help = "dir to write the metadata json into")
    ap.add_argument("--publish-repo", required = True)
    args = ap.parse_args()

    dist = Path(args.dist)
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    bundles = sorted(dist.glob("sd-*-bin-*.zip"))
    if not bundles:
        raise SystemExit(f"assemble_metadata: no sd-*-bin-*.zip found in {dist}")

    sha_index: dict[str, str] = {}
    assets: list[dict] = []
    for b in bundles:
        digest = _sha256(b)
        sha_index[b.name] = digest
        m = _LABEL_RE.match(b.name)
        assets.append(
            {
                "name": b.name,
                "label": m.group("label") if m else "",
                "size": b.stat().st_size,
                "sha256": digest,
            }
        )

    manifest = {
        "schema": 1,
        "tag": args.tag,
        "source_repo": args.source_repo,
        "source_commit": args.commit,
        "publish_repo": args.publish_repo,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": assets,
    }

    (out / "sd-prebuilt-sha256.json").write_text(json.dumps(sha_index, indent = 2) + "\n")
    (out / "sd-prebuilt-manifest.json").write_text(json.dumps(manifest, indent = 2) + "\n")
    print(f"assemble_metadata: {len(bundles)} bundles indexed for {args.tag}", flush = True)
    for a in assets:
        print(f"  {a['name']}  {a['size']} B  {a['sha256'][:12]}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
