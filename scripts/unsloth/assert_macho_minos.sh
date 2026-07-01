#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.
#
# macOS load gate for the sd-cli / sd-server Mach-O binaries. Owning the build lets
# us pin CMAKE_OSX_DEPLOYMENT_TARGET so the binary declares a load floor OLD enough
# to run on the macOS versions we support; this asserts that actually happened, that
# the arch is what we intended, and that the binary launches. Fails the job otherwise
# so we never publish a bundle that dyld refuses to load on a supported macOS.
#
# Usage: assert_macho_minos.sh <bin_dir> <expect_arch: arm64|x86_64> <deploy_target: e.g. 14.0>

set -euo pipefail

BIN_DIR="${1:?usage: assert_macho_minos.sh <bin_dir> <expect_arch> <deploy_target>}"
EXPECT_ARCH="${2:?expected arch (arm64|x86_64)}"
DEPLOY_TARGET="${3:?deploy target, e.g. 14.0}"

# a <= b using version sort.
ver_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }

fail=0
checked=0

for name in sd-cli sd-server; do
  bin="$BIN_DIR/$name"
  if [ ! -f "$bin" ]; then
    echo "note: $name not present in $BIN_DIR; skipping"
    continue
  fi
  checked=$((checked + 1))
  echo "== $name =="

  # Architecture.
  archs="$(lipo -archs "$bin" 2>/dev/null || echo unknown)"
  case " $archs " in
    *" $EXPECT_ARCH "*) echo "  arch: $archs (ok)" ;;
    *) echo "  ERROR: arch [$archs] does not contain $EXPECT_ARCH"; fail=1 ;;
  esac

  # Minimum OS: LC_BUILD_VERSION 'minos', or the legacy LC_VERSION_MIN_MACOSX 'version'.
  minos="$(otool -l "$bin" 2>/dev/null | awk '
    /LC_BUILD_VERSION/     {inbv=1}
    /LC_VERSION_MIN_MACOSX/{invm=1}
    inbv && $1=="minos"    {print $2; exit}
    invm && $1=="version"  {print $2; exit}
  ')"
  if [ -n "$minos" ]; then
    if ver_le "$minos" "$DEPLOY_TARGET"; then
      echo "  minos: $minos <= $DEPLOY_TARGET (ok)"
    else
      echo "  ERROR: minos $minos > deploy target $DEPLOY_TARGET (won't load on older macOS)"
      fail=1
    fi
  else
    echo "  WARNING: no minos load command found"
  fi

  # Launch check: the runner is the same arch we built for, so this exercises dyld.
  # sd-cli may exit non-zero on --help; treat a clean or usage exit as launched, and
  # only warn (not fail) so a help-text convention change doesn't block a release.
  if "$bin" --help >/dev/null 2>&1 || "$bin" -h >/dev/null 2>&1 || "$bin" >/dev/null 2>&1; then
    echo "  launch: ok"
  else
    rc=$?
    if [ "$rc" -ge 126 ]; then
      echo "  ERROR: $name failed to launch (exit $rc: dyld/exec error)"; fail=1
    else
      echo "  launch: exited $rc (usage/no-args; treated as launched)"
    fi
  fi
done

[ "$checked" -gt 0 ] || { echo "ERROR: no binaries found in $BIN_DIR"; exit 1; }
[ "$fail" = 0 ] || { echo "ERROR: macOS load gate failed"; exit 1; }
echo "macOS load gate passed ($checked binaries, arch=$EXPECT_ARCH, floor<=$DEPLOY_TARGET)"
