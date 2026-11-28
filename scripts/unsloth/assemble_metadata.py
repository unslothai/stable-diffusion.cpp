#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.

"""Generate the release metadata for an Unsloth stable-diffusion.cpp prebuild.

Writes two files next to the bundles:
  * sd-prebuilt-sha256.json  -- {asset_name: sha256_hex} for every bundle
  * sd-prebuilt-manifest.json -- tag / source / commit / per-asset {size, sha256, label}, plus, for
    every CUDA bundle, the coverage fields of its embedded UNSLOTH_PREBUILT_INFO.json (os, arch,
    runtime_line, coverage_class, rank, supported_sms, min_sm, max_sm, sass_sms, ptx_sms), the
    fields unslothai/llama.cpp's llama-prebuilt-manifest.json carries for its CUDA bundles and Studio
    selects a per-GPU-family bundle by

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
import tarfile
import zipfile
from pathlib import Path

# Map an asset filename back to a host label, for the manifest (informational).
_LABEL_RE = re.compile(r"^sd-.*-bin-(?P<label>.+)\.zip$")
# The per-GPU-family CUDA bundles: sd-<tag>-<platform>-<arch>-<profile>.<ext>, llama.cpp's shape.
_SPLIT_RE = re.compile(
    r"^sd-.+-(?P<label>(?:linux|windows)-(?:x64|arm64)-cuda\d+-(?:older|newer|portable))\.(?:zip|tar\.gz)$"
)
_INFO_NAME = "UNSLOTH_PREBUILT_INFO.json"
# Copied into the manifest entry verbatim from the bundle's own record.
_INFO_FIELDS = (
    "backend", "os", "arch", "bundle_profile", "runtime_line", "coverage_class", "toolkit",
    "supported_sms", "min_sm", "max_sm", "sass_sms", "ptx_sms", "cuda_architectures",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _embedded_info(path: Path) -> dict | None:
    """The bundle's UNSLOTH_PREBUILT_INFO.json (top-level dir/<name>), or None when it has none."""
    try:
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as tf:
                for m in tf:
                    if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/" + _INFO_NAME):
                        f = tf.extractfile(m)
                        return json.loads(f.read().decode()) if f else None
            return None
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.count("/") == 1 and n.endswith("/" + _INFO_NAME):
                    return json.loads(zf.read(n).decode())
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"assemble_metadata: cannot read {path.name}: {exc}")
    return None


def _coverage(info: dict, name: str) -> dict:
    """Manifest coverage fields from a bundle record, validated so a malformed one fails the run
    instead of shipping a manifest Studio would misread."""
    out = {k: info[k] for k in _INFO_FIELDS if k in info}
    out["rank"] = int(info["bundle_rank"])
    sms = [str(int(v)) for v in info.get("supported_sms") or []]
    if not sms or out.get("min_sm") is None or out.get("max_sm") is None:
        raise SystemExit(f"assemble_metadata: {name} records no SM coverage")
    if not all(out["min_sm"] <= int(v) <= out["max_sm"] for v in sms):
        raise SystemExit(f"assemble_metadata: {name} lists SMs outside its min/max")
    out["supported_sms"] = sms
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required = True)
    ap.add_argument("--source-repo", required = True)
    ap.add_argument("--commit", required = True)
    # Our build is tagged <upstream tag>-u<head sha>, so the two differ whenever this tree carries
    # anything past that release, which it always does. Both are recorded: the tag identifies the
    # artifact, upstream_tag says which upstream release the tree descends from, and commit says
    # exactly what was compiled, so a consumer can tell what it has without unpacking it.
    #
    # --patches is retained and always empty. The build no longer applies a patch set at build
    # time; the fixes are commits in this repository. Keeping the field means existing manifest
    # readers do not have to change.
    ap.add_argument("--upstream-tag", default = "")
    ap.add_argument(
        "--patches",
        default = "",
        help = "retained for manifest compatibility; always empty now that fixes are commits",
    )
    ap.add_argument("--dist", required = True, help = "dir holding the sd-*.zip bundles")
    ap.add_argument("--out", required = True, help = "dir to write the metadata json into")
    ap.add_argument("--publish-repo", required = True)
    args = ap.parse_args()

    dist = Path(args.dist)
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    bundles = sorted(
        set(dist.glob("sd-*-bin-*.zip"))
        | {p for p in dist.glob("sd-*") if _SPLIT_RE.match(p.name)}
    )
    if not bundles:
        raise SystemExit(f"assemble_metadata: no sd-*-bin-*.zip found in {dist}")

    sha_index: dict[str, str] = {}
    assets: list[dict] = []
    for b in bundles:
        digest = _sha256(b)
        sha_index[b.name] = digest
        m = _LABEL_RE.match(b.name) or _SPLIT_RE.match(b.name)
        entry = {
            "name": b.name,
            "label": m.group("label") if m else "",
            "size": b.stat().st_size,
            "sha256": digest,
        }
        info = _embedded_info(b)
        if info is not None and info.get("backend") == "cuda":
            entry.update(_coverage(info, b.name))
        elif _SPLIT_RE.match(b.name):
            raise SystemExit(f"assemble_metadata: {b.name} carries no {_INFO_NAME}")
        assets.append(entry)

    manifest = {
        "schema": 1,
        "tag": args.tag,
        "upstream_tag": args.upstream_tag or args.tag,
        "patches": [p for p in args.patches.split(",") if p],
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
        cov = f"  {a['coverage_class']} sms={','.join(a['supported_sms'])}" if "supported_sms" in a else ""
        print(f"  {a['name']}  {a['size']} B  {a['sha256'][:12]}{cov}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
