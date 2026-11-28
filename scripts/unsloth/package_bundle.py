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
    GGML_LICENSE_FILE     ggml's LICENSE, shipped as ggml.txt (optional)
    CUTLASS_LICENSE_FILE  CUTLASS's LICENSE.txt, shipped as cutlass.txt (optional, CUDA builds)
    ASSET_STEM    archive stem, default ``sd-<TAG>-bin-<LABEL>`` (optional)
    ARCHIVE_FORMAT  ``zip`` (default) or ``tar.gz`` (optional)

CUDA legs also set PROFILE / RUNTIME_LINE / COVERAGE_CLASS / RANK / ARCHS / PLATFORM_OS /
PLATFORM_ARCH (TOOLKIT and SMS optional), and the bundle then carries UNSLOTH_PREBUILT_INFO.json,
the same per-bundle record unslothai/llama.cpp's package_bundle.py embeds: which SMs the bundle has
SASS for, which it has PTX for, and the coverage class / rank Studio's selector orders by.

The archive unpacks into a single named dir (its stem, ``sd-<TAG>-bin-<LABEL>/`` by default) containing the
binaries, their sibling runtime libs, LICENSE, the ggml.txt / cutlass.txt notices upstream's
own packages carry, and an UNSLOTH_BUILD.txt provenance file (also the fingerprint the
assemble step greps for). The Studio finder rglobs
for sd-cli / sd-server, so the internal layout is not load-bearing.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import io
import json
import sys
import tarfile
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


# libggml-cuda.so.0, the name DT_NEEDED spells. The fully versioned file behind it and the bare
# libggml-cuda.so link are build-time names only.
_SONAME_RE = re.compile(r"lib.+\.so\.\d+")


def _collect(bin_dir: Path) -> list[tuple[str, Path]]:
    """(name in the bundle, file) for the binaries + sibling runtime libs. Recurse so a nested bin/
    layout (some generators emit build/bin/, some build/bin/Release/) is still captured.

    A shared-ggml build leaves each library as a real file plus two symlinks, and zipfile follows
    links, so shipping all three stores the 100+ MB CUDA library three times. Ship it once, under
    its soname."""
    wanted = [
        p for p in sorted(bin_dir.rglob("*"))
        if p.is_file() and (p.name in _BINARIES or _is_runtime_lib(p.name))
    ]
    found = {p.name: p.resolve() for p in wanted if p.is_symlink() and _SONAME_RE.fullmatch(p.name)}
    under_soname = set(found.values())
    for p in wanted:
        if p.name not in found and p.resolve() not in under_soname:
            found[p.name] = p
    return sorted(found.items())


def _optional_license(name: str) -> Path | None:
    """A license path from the optional env ``name``. Unset means skip. Set but missing is
    fatal: a notice that silently drops out of a bundle is not something to ship."""
    val = os.environ.get(name, "").strip()
    if not val:
        return None
    path = Path(val)
    if not path.is_file():
        print(f"package_bundle: {name}={val} is not a file", file = sys.stderr)
        raise SystemExit(2)
    return path


_ARCH_RE = re.compile(r"^(?P<sm>\d+)(?P<variant>[af]?)(?:-(?P<kind>real|virtual))?$")


def cuda_profile_info(archs: str, sms: str = "") -> dict:
    """SM coverage of a CMAKE_CUDA_ARCHITECTURES list. A bare arch is SASS + PTX, ``-real`` SASS only,
    ``-virtual`` PTX only. supported_sms is what Studio matches a host against: every SASS arch (or the
    explicit SMS list), plus 103 beside 100, since sm_100 SASS runs on sm_103 and no toolkit before 12.9
    has a native sm_103 target (unslothai/llama.cpp's package_bundle.py declares it the same way).
    ptx_sms is what a newer card than the list can JIT from; an arch-specific (``a``/``f``) PTX loads
    only on its own SM family, so it is not listed."""
    sass: list[int] = []
    ptx: list[int] = []
    for raw in re.split(r"[ ;,]+", archs.strip()):
        if not raw:
            continue
        m = _ARCH_RE.match(raw)
        if not m:
            raise SystemExit(f"package_bundle: cannot parse CUDA arch {raw!r} in ARCHS={archs!r}")
        sm = int(m.group("sm"))
        kind = m.group("kind")
        if kind in (None, "real"):
            sass.append(sm)
        # ggml-cuda rewrites a plain 12X to 12Xa, so its PTX would be the arch-specific kind too.
        if kind in (None, "virtual") and not m.group("variant") and not 120 <= sm <= 129:
            ptx.append(sm)
    declared = [int(s) for s in re.split(r"[ ;,]+", sms.strip()) if s] if sms.strip() else sass
    if not declared:
        raise SystemExit(f"package_bundle: ARCHS={archs!r} names no SASS arch and SMS is empty")
    supported = sorted(set(declared) | ({103} if 100 in declared else set()))
    return {
        "cuda_architectures": archs,
        "sass_sms": [str(v) for v in sorted(set(sass))],
        "ptx_sms": [str(v) for v in sorted(set(ptx))],
        "supported_sms": [str(v) for v in supported],
        "min_sm": min(declared),
        "max_sm": max(supported),
    }


def _bundle_info() -> dict | None:
    """UNSLOTH_PREBUILT_INFO.json for a CUDA leg, or None when PROFILE is unset (non-CUDA legs)."""
    profile = os.environ.get("PROFILE", "").strip()
    if not profile:
        return None
    info = {
        "schema": 1,
        "backend": "cuda",
        "bundle_profile": profile,
        "os": _env("PLATFORM_OS"),
        "arch": _env("PLATFORM_ARCH"),
        "runtime_line": _env("RUNTIME_LINE"),
        "coverage_class": _env("COVERAGE_CLASS"),
        "bundle_rank": int(_env("RANK")),
        "toolkit": os.environ.get("TOOLKIT", "").strip() or None,
    }
    info.update(cuda_profile_info(_env("ARCHS"), os.environ.get("SMS", "")))
    return info


def _add_bytes_tar(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    ti = tarfile.TarInfo(arcname)
    ti.size = len(data)
    ti.mode = 0o644
    tf.addfile(ti, io.BytesIO(data))


def main() -> int:
    bin_dir = Path(_env("BIN_DIR"))
    out_dir = Path(_env("OUT_DIR"))
    tag = _env("TAG")
    label = _env("LABEL")
    commit = os.environ.get("COMMIT", "").strip() or "unknown"
    source_repo = os.environ.get("SOURCE_REPO", "").strip() or "leejet/stable-diffusion.cpp"
    license_file = _optional_license("LICENSE_FILE")
    # ggml is compiled into every bundle and CUTLASS into the CUDA ones; upstream's build.yml
    # ships both notices under these names.
    extra_notices = (("ggml.txt", _optional_license("GGML_LICENSE_FILE")),
                     ("cutlass.txt", _optional_license("CUTLASS_LICENSE_FILE")))

    if not bin_dir.is_dir():
        print(f"package_bundle: BIN_DIR {bin_dir} is not a directory", file = sys.stderr)
        return 2

    files = _collect(bin_dir)
    have_cli = any(name in ("sd-cli", "sd-cli.exe") for name, _ in files)
    if not have_cli:
        print(f"package_bundle: no sd-cli under {bin_dir}; refusing to package", file = sys.stderr)
        return 1
    have_server = any(name in ("sd-server", "sd-server.exe") for name, _ in files)

    out_dir.mkdir(parents = True, exist_ok = True)
    stem = os.environ.get("ASSET_STEM", "").strip() or f"sd-{tag}-bin-{label}"
    archive_format = os.environ.get("ARCHIVE_FORMAT", "").strip() or "zip"
    if archive_format not in ("zip", "tar.gz"):
        print(f"package_bundle: ARCHIVE_FORMAT={archive_format} is not zip or tar.gz", file = sys.stderr)
        return 2
    zip_path = out_dir / f"{stem}.{archive_format}"
    info = _bundle_info()

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

    members: list[tuple[str, Path]] = [(f"{stem}/{name}", f) for name, f in files]
    if license_file:
        members.append((f"{stem}/LICENSE", license_file))
    for arcname, notice in extra_notices:
        if notice:
            members.append((f"{stem}/{arcname}", notice))
    generated = {f"{stem}/UNSLOTH_BUILD.txt": provenance.encode()}
    if info is not None:
        generated[f"{stem}/UNSLOTH_PREBUILT_INFO.json"] = (json.dumps(info, indent = 2) + "\n").encode()

    # Deterministic-ish: sort members; drop the archive if it already exists.
    zip_path.unlink(missing_ok = True)
    if archive_format == "zip":
        with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as zf:
            for arcname, f in members:
                # Flatten under the named top-level dir so the binaries sit at
                # <stem>/<name> regardless of build layout.
                zf.write(f, arcname = arcname)
            for arcname, data in generated.items():
                zf.writestr(arcname, data)
    else:
        # Regular files only (dereferenced), so the archive holds no links for an installer to vet.
        with tarfile.open(zip_path, "w:gz", compresslevel = 6, dereference = True) as tf:
            for arcname, f in members:
                tf.add(f, arcname = arcname, recursive = False)
            for arcname, data in generated.items():
                _add_bytes_tar(tf, arcname, data)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(
        f"package_bundle: wrote {zip_path} ({size_mb:.1f} MiB); "
        f"sd-cli=yes sd-server={'yes' if have_server else 'no'} files={len(files)}",
        flush = True,
    )
    if info is not None:
        print(
            f"package_bundle: {info['bundle_profile']} sass={','.join(info['sass_sms'])} "
            f"ptx={','.join(info['ptx_sms']) or 'none'} supported={','.join(info['supported_sms'])}",
            flush = True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
