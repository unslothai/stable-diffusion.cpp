#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.
#
# CPU bundles ship ggml's CPU backend as runtime-selected variants (GGML_BACKEND_DL +
# GGML_CPU_ALL_VARIANTS): libggml-cpu-<variant> for every x86 level from plain x86-64 up to
# AVX512 / AVX-VNNI / AMX, or every Armv8/v9 level, and ggml loads the best one the host CPU
# supports. A static -DGGML_NATIVE=OFF build is pinned to AVX2 on x86 (and crashes with SIGILL
# on anything older) and to plain Armv8.0 on Linux arm64, so most machines ran a slower kernel
# set than they support.
#
#   cpu_variants_gate.sh pin   <bin_dir>
#       Linux only: point every binary and ggml library at $ORIGIN with DT_RPATH (searched
#       before LD_LIBRARY_PATH, unlike RUNPATH), the same rule the CUDA bundles use, so the
#       bundle's own libggml-base wins over another ggml on the loader path.
#   cpu_variants_gate.sh check <bundle.zip> <floor_variant>
#       Unzip into a fresh directory and prove the bundle is self-contained and loadable:
#       the floor variant (the one every CPU of the arch can run) is present, no library or
#       binary still points at the build tree, and sd-cli started from the unzipped copy
#       loads a CPU variant from that copy.

set -euo pipefail

mode="${1:?usage: cpu_variants_gate.sh pin <bin_dir> | check <bundle.zip> <floor_variant>}"

case "$mode" in
  pin)
    bin_dir="${2:?bin_dir}"
    [ "$(uname -s)" = "Linux" ] || { echo "pin: nothing to do on $(uname -s)"; exit 0; }
    command -v patchelf > /dev/null || { echo "ERROR: patchelf not installed (the workflow installs it)" >&2; exit 1; }
    n=0
    for f in "$bin_dir"/sd-cli "$bin_dir"/sd-server "$bin_dir"/libggml*.so* "$bin_dir"/libstable-diffusion*.so*; do
      [ -f "$f" ] && [ ! -L "$f" ] || continue
      patchelf --force-rpath --set-rpath '$ORIGIN' "$f"
      n=$((n + 1))
    done
    echo "pin: set DT_RPATH=\$ORIGIN on $n files"
    ;;
  check)
    zip="${2:?bundle.zip}"
    floor="${3:?floor variant, e.g. x64 or armv8.0_1}"
    work=""
    empty=""
    # Registered before the first step that can fail, so an early exit still removes both.
    cleanup() {
      if [ -n "$work" ]; then rm -rf -- "$work"; fi
      if [ -n "$empty" ]; then rm -rf -- "$empty"; fi
    }
    trap cleanup EXIT
    work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/cpu-variants-XXXXXX")"
    unzip -q "$zip" -d "$work"
    dir="$(dirname "$(find "$work" -type f \( -name sd-cli -o -name sd-cli.exe \) | head -n1)")"
    [ -n "$dir" ] && [ -d "$dir" ] || { echo "ERROR: no sd-cli in $zip" >&2; exit 1; }
    case "$(uname -s)" in
      Darwin) ext=so ;;  # ggml names its MODULE libraries .so on macOS too
      MINGW*|MSYS*|CYGWIN*) ext=dll ;;
      *) ext=so ;;
    esac
    prefix="libggml-cpu-"
    [ "$ext" = dll ] && prefix="ggml-cpu-"
    variants="$(cd "$dir" && ls ${prefix}*.${ext} 2>/dev/null | sed -e "s/^${prefix}//" -e "s/\.${ext}\$//" | sort | tr '\n' ' ')"
    echo "variants: $variants"
    case " $variants " in
      *" $floor "*) ;;
      *) echo "ERROR: floor variant '$floor' missing; hosts below the next level would have no CPU backend" >&2; exit 1 ;;
    esac
    fail=0
    if [ "$(uname -s)" = "Linux" ]; then
      for f in "$dir"/sd-cli "$dir"/sd-server "$dir"/libggml*.so* "$dir"/libstable-diffusion*.so*; do
        [ -f "$f" ] || continue
        rp="$(readelf -d "$f" | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\1=\2/p')"
        if [ "$rp" != 'RPATH=$ORIGIN' ]; then echo "ERROR: $(basename "$f") has '${rp:-no rpath}', want RPATH=\$ORIGIN" >&2; fail=1; fi
      done
    elif [ "$(uname -s)" = "Darwin" ]; then
      for f in "$dir"/sd-cli "$dir"/sd-server "$dir"/*.dylib "$dir"/*.so; do
        [ -f "$f" ] || continue
        bad="$(otool -l "$f" | awk '/LC_RPATH/{getline; getline; print $2}' | grep -v '^@' || true)"
        if [ -n "$bad" ]; then echo "ERROR: $(basename "$f") keeps a build-tree rpath: $bad" >&2; fail=1; fi
        deps="$(otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -E 'ggml|stable-diffusion' | grep -v '^@' || true)"
        if [ -n "$deps" ]; then echo "ERROR: $(basename "$f") links ggml by absolute path: $deps" >&2; fail=1; fi
      done
    fi
    cli="$dir/sd-cli"
    [ -f "$cli.exe" ] && cli="$cli.exe"
    # A model path that cannot exist: sd-cli registers the backends, prints System Info, then
    # stops at the model. Run from an empty cwd so ggml's cwd search cannot find anything.
    empty="$(mktemp -d "${RUNNER_TEMP:-/tmp}/cpu-variants-cwd-XXXXXX")"
    out="$(cd "$empty" && env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH "$cli" -m "$empty/none.gguf" -p x -v 2>&1 || true)"
    loaded="$(printf '%s\n' "$out" | sed -n 's/.*loaded CPU backend from \(.*\)$/\1/p' | head -n1 | tr -d '\r')"
    printf '%s\n' "$out" | grep -A1 "System Info" | head -n 3 || true
    if [ -z "$loaded" ]; then
      echo "ERROR: sd-cli loaded no CPU backend variant" >&2
      printf '%s\n' "$out" | tail -n 30 >&2
      fail=1
    else
      echo "loaded: $loaded"
      loaded_dir="$(cd "$(dirname "$loaded")" && pwd -P)"
      want_dir="$(cd "$dir" && pwd -P)"
      if [ "$loaded_dir" != "$want_dir" ]; then echo "ERROR: variant came from $loaded_dir, not the bundle" >&2; fail=1; fi
    fi
    exit "$fail"
    ;;
  *)
    echo "unknown mode $mode" >&2
    exit 2
    ;;
esac
