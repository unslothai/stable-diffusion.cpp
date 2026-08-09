#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.

"""Package a built stable-diffusion.cpp bin directory into a release zip.

Mirrors the asset naming the Unsloth Studio installer's ``resolve_release_asset``
(studio/install_sd_cpp_prebuilt.py) already expects, so the Studio can point at
this mirror with only a repo/tag flip. Reads everything from env so the same
script serves every OS matrix leg:

    BIN_DIR   build output dir holding sd-cli / sd-server (and any sibling libs)
    OUT_DIR   where to write the .zip
    TAG       release tag, e.g. master-741-484baa4
    LABEL     asset label, e.g. Darwin-macOS-arm64 / Linux-Ubuntu-24.04-x86_64 / win-cpu-x64
    COMMIT    source commit SHA (provenance)
    SOURCE_REPO   e.g. leejet/stable-diffusion.cpp
    LICENSE_FILE  path to the LICENSE to include (optional)

The zip unpacks into a single named dir ``sd-<TAG>-bin-<LABEL>/`` containing the
binaries, their sibling runtime libs, LICENSE, and an UNSLOTH_BUILD.txt provenance
file (also the fingerprint the assemble step greps for). The Studio finder rglobs
for sd-cli / sd-server, so the internal layout is not load-bearing.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import zipfile
from pathlib import Path

# Binaries we ship, plus the runtime-library extensions to sweep in alongside them
# (static builds usually have none; Metal / a shared ggml can add a few).
_BINARIES = ("sd-cli", "sd-server", "sd-cli.exe", "sd-server.exe")
_LIB_SUFFIXES = (".dylib", ".so", ".dll", ".metal", ".metallib")

_FINGERPRINT = "Compiled by the Unsloth team"


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"package_bundle: missing required env {name}", file = sys.stderr)
        raise SystemExit(2)
    return val


def _is_runtime_lib(name: str) -> bool:
    lowered = name.lower()
    if Path(lowered).suffix in _LIB_SUFFIXES:
        return True
    # Versioned ELF sonames: libcudart.so.12, libcublas.so.12.8.4.1. Path.suffix sees
    # ".12" and would drop these, but the name has to stay exactly as DT_NEEDED spells
    # it, so match the ".so." infix instead of renaming the file.
    return ".so." in lowered


def _collect(bin_dir: Path) -> list[Path]:
    """The binaries + sibling runtime libs to ship. Recurse so a nested bin/ layout
    (some generators emit build/bin/, some build/bin/Release/) is still captured."""
    found: list[Path] = []
    for p in sorted(bin_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in _BINARIES or _is_runtime_lib(p.name):
            found.append(p)
    return found


def main() -> int:
    bin_dir = Path(_env("BIN_DIR"))
    out_dir = Path(_env("OUT_DIR"))
    tag = _env("TAG")
    label = _env("LABEL")
    commit = os.environ.get("COMMIT", "").strip() or "unknown"
    source_repo = os.environ.get("SOURCE_REPO", "").strip() or "leejet/stable-diffusion.cpp"
    license_file = os.environ.get("LICENSE_FILE", "").strip()

    if not bin_dir.is_dir():
        print(f"package_bundle: BIN_DIR {bin_dir} is not a directory", file = sys.stderr)
        return 2

    files = _collect(bin_dir)
    have_cli = any(f.name in ("sd-cli", "sd-cli.exe") for f in files)
    if not have_cli:
        print(f"package_bundle: no sd-cli under {bin_dir}; refusing to package", file = sys.stderr)
        return 1
    have_server = any(f.name in ("sd-server", "sd-server.exe") for f in files)

    out_dir.mkdir(parents = True, exist_ok = True)
    stem = f"sd-{tag}-bin-{label}"
    zip_path = out_dir / f"{stem}.zip"

    built_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = (
        f"{_FINGERPRINT}.\n"
        f"stable-diffusion.cpp prebuilt (Unsloth mirror)\n"
        f"tag: {tag}\n"
        f"label: {label}\n"
        f"source_repo: {source_repo}\n"
        f"source_commit: {commit}\n"
        f"built_at: {built_at}\n"
        f"binaries: {'sd-cli' + (' sd-server' if have_server else '')}\n"
    )

    # Deterministic-ish: sort members; drop the archive if it already exists.
    zip_path.unlink(missing_ok = True)
    with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            # Flatten under the named top-level dir; keep just the basename so the
            # binaries sit at sd-<tag>-bin-<label>/<name> regardless of build layout.
            zf.write(f, arcname = f"{stem}/{f.name}")
        zf.writestr(f"{stem}/UNSLOTH_BUILD.txt", provenance)
        if license_file and Path(license_file).is_file():
            zf.write(license_file, arcname = f"{stem}/LICENSE")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(
        f"package_bundle: wrote {zip_path} ({size_mb:.1f} MiB); "
        f"sd-cli=yes sd-server={'yes' if have_server else 'no'} files={len(files)}",
        flush = True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
