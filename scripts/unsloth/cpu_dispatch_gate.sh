#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.
#
#   cpu_dispatch_gate.sh <bundle.zip>
#
# Runs the bundle's sd-cli on emulated CPUs that are older or newer than the build runner: Intel SDE on
# Linux x86_64 (Nehalem .. Sapphire Rapids), qemu-aarch64 on Linux aarch64 (Cortex-A53 .. max). For each CPU
# it fails when
#   - the process dies or SDE reports an instruction the emulated CPU does not have. ggml dlopens EVERY CPU
#     variant library while it scores them, so a static initializer that the compiler vectorised with a
#     variant's AVX512 / SVE flags crashes every host without that ISA, before any variant is picked;
#   - no CPU variant is loaded, or the loaded variant needs a feature the emulated CPU lacks;
#   - on a 128-bit SVE vector length (CPU names ending in -sve128) a variant other than the best NEON one is loaded
#     (armv8.6_0 with i8mm, else armv8.2_2): ggml-patches/0020 skips the SVE variants there unless
#     GGML_CPU_ARM_SVE128=1 (names ending in +keep
#     set it and must then load an SVE variant);
#   - a variant that has the iqk GEMM kernels built does not enable them by default (ARM: every batch size;
#     x86 AVX2 and up: 2-12 src1 rows, ggml-patches/0018). Variants without them (pre-AVX2 x86, Armv8.0)
#     report built types 0x0.
# The runner's own CPU cannot catch either: it has the newest ISA the bundle uses.
#
# Env: SDE_DIR (an unpacked Intel SDE; downloaded and checked against SDE_SHA256 when unset),
#      QEMU_AARCH64 (default qemu-aarch64), CPU_DISPATCH_GATE_CPUS (space separated subset of the CPUs below).

set -euo pipefail

zip="${1:?usage: cpu_dispatch_gate.sh <bundle.zip>}"

SDE_URL="https://downloadmirror.intel.com/859732/sde-external-9.58.0-2025-06-16-lin.tar.xz"
SDE_SHA256="f849acecad4c9b108259c643b2688fd65c35723cd23368abe5dd64b917cc18c0"

# ggml CPU variant -> features it is built with (ggml/src/CMakeLists.txt ggml_add_cpu_backend_variant).
# AVX512 stands for F + CD + VL + DQ + BW, the set ggml's AVX512 variants require.
variant_needs() {
  case "$1" in
    x64) echo "" ;;
    sse42) echo "SSE42" ;;
    sandybridge) echo "SSE42 AVX" ;;
    ivybridge) echo "SSE42 AVX F16C" ;;
    piledriver) echo "SSE42 AVX F16C FMA" ;;
    haswell) echo "SSE42 AVX F16C FMA AVX2 BMI2" ;;
    skylakex) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512" ;;
    cannonlake) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI" ;;
    cascadelake) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI" ;;
    icelake) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI" ;;
    cooperlake) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16" ;;
    zen4) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16" ;;
    alderlake) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI" ;;
    sapphirerapids) echo "SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8" ;;
    armv8.0_1) echo "" ;;
    armv8.2_1) echo "DOTPROD" ;;
    armv8.2_2) echo "DOTPROD FP16_VA" ;;
    armv8.6_0) echo "DOTPROD FP16_VA I8MM" ;;
    armv8.2_3) echo "DOTPROD FP16_VA SVE" ;;
    armv8.6_1) echo "DOTPROD FP16_VA SVE I8MM" ;;
    armv8.6_2) echo "DOTPROD FP16_VA SVE I8MM SVE2" ;;
    armv9.2_1) echo "DOTPROD FP16_VA SVE I8MM SME" ;;
    armv9.2_2) echo "DOTPROD FP16_VA SVE I8MM SVE2 SME" ;;
    *) echo "UNKNOWN_VARIANT" ;;
  esac
}

# Emulated CPU -> features it has. x86: SDE chip flags; aarch64: qemu -cpu models.
X86_HSW="SSE42 AVX F16C FMA AVX2 BMI2"
X86_SKX="$X86_HSW AVX512"
cpu_has() {
  case "${1%-sve128}" in
    nhm|wsm) echo "SSE42" ;;
    snb) echo "SSE42 AVX" ;;
    ivb) echo "SSE42 AVX F16C" ;;
    hsw|bdw|skl) echo "$X86_HSW" ;;
    adl) echo "$X86_HSW AVX_VNNI" ;;
    skx) echo "$X86_SKX" ;;
    clx) echo "$X86_SKX AVX512_VNNI" ;;
    icl|icx) echo "$X86_SKX AVX512_VBMI AVX512_VNNI" ;;
    cpx) echo "$X86_SKX AVX512_VNNI AVX512_BF16" ;;
    spr) echo "$X86_SKX AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8 AVX_VNNI" ;;
    cortex-a53) echo "" ;;
    cortex-a76|neoverse-n1) echo "DOTPROD FP16_VA" ;;
    neoverse-v1) echo "DOTPROD FP16_VA SVE I8MM" ;;
    max) echo "DOTPROD FP16_VA SVE I8MM SVE2 SME" ;;
    *) echo "UNKNOWN_CPU" ;;
  esac
}

work=""
cleanup() { if [ -n "$work" ]; then rm -rf -- "$work"; fi; }
trap cleanup EXIT
work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/cpu-dispatch-XXXXXX")"
unzip -q "$zip" -d "$work/bundle"
cli="$(find "$work/bundle" -type f -name sd-cli | head -n1)"
[ -n "$cli" ] || { echo "ERROR: no sd-cli in $zip" >&2; exit 1; }
mkdir -p "$work/cwd"

arch="$(uname -m)"
case "$arch" in
  x86_64)
    cpus="${CPU_DISPATCH_GATE_CPUS:-nhm wsm snb ivb hsw bdw adl skx clx icl icx spr}"
    if [ -z "${SDE_DIR:-}" ]; then
      curl -fsSL --retry 3 -o "$work/sde.tar.xz" "$SDE_URL"
      echo "$SDE_SHA256  $work/sde.tar.xz" | sha256sum --check --strict
      mkdir -p "$work/sde"
      tar -xJf "$work/sde.tar.xz" -C "$work/sde" --strip-components=1
      SDE_DIR="$work/sde"
    fi
    emu() { emu_cmd=("$SDE_DIR/sde64" "-$1" --); }
    ;;
  aarch64)
    cpus="${CPU_DISPATCH_GATE_CPUS:-cortex-a53 cortex-a76 neoverse-v1 max neoverse-v1-sve128 max-sve128 max-sve128+keep}"
    qemu="${QEMU_AARCH64:-qemu-aarch64}"
    command -v "$qemu" > /dev/null || { echo "ERROR: $qemu not installed (the workflow installs qemu-user)" >&2; exit 1; }
    # -sve128: the same model with a 16-byte default SVE vector length (Neoverse N2 / V2 class hardware)
    emu() {
      case "$1" in
        *-sve128) emu_cmd=("$qemu" -cpu "${1%-sve128},sve-default-vector-length=16") ;;
        *) emu_cmd=("$qemu" -cpu "$1") ;;
      esac
    }
    ;;
  *)
    echo "cpu_dispatch_gate: nothing to emulate on $arch"
    exit 0
    ;;
esac

fail=0
printf '%-18s %-16s %-6s %s\n' cpu variant rc note
for entry in $cpus; do
  cpu="${entry%+keep}"
  keep=""
  [ "$cpu" != "$entry" ] && keep=1
  has=" $(cpu_has "$cpu") "
  # Nonexistent model: sd-cli loads (and scores) every CPU variant, initialises the chosen one, then stops.
  # Empty cwd defeats ggml's cwd search. GGML_CPU_IQK_DEBUG makes the iqk GEMM report what it was built with.
  emu "$cpu"
  set +e
  out="$(cd "$work/cwd" && env -u LD_LIBRARY_PATH GGML_CPU_IQK_DEBUG=1 ${keep:+GGML_CPU_ARM_SVE128=1} \
         "${emu_cmd[@]}" "$cli" -m "$work/cwd/none.gguf" -p x -v 2>&1)"
  rc=$?
  set -e
  loaded="$(printf '%s\n' "$out" | sed -n 's/.*loaded CPU backend from .*libggml-cpu-\(.*\)\.so$/\1/p' | head -n1 | tr -d '\r')"
  bad="$(printf '%s\n' "$out" | grep -m1 -E 'SDE-ERROR|Illegal instruction|SIGILL|core dumped' || true)"
  where="$(printf '%s\n' "$out" | sed -n 's/^Image: .*\/\([^/]*\)$/\1/p; s/^Function: \(.*\)$/in \1/p' | head -n2 | tr '\n' ' ')"
  iqk="$(printf '%s\n' "$out" | sed -n 's/^ggml-cpu iqk gemm: \(built types [^ ]*\), \(enabled [^ ]*\).*/\1 \2/p' | head -n1)"
  note=""
  # rc 1 is sd-cli's own exit for the missing model; a signal is 128+n, SDE's chip check also exits 1.
  if [ -n "$bad" ] || [ "$rc" -gt 1 ]; then
    note="CRASH: ${bad:-exit $rc} ${where}"
  elif [ -z "$loaded" ]; then
    note="no CPU variant loaded"
  else
    needs=" $(variant_needs "$loaded") "
    for f in $needs; do
      case "$has" in *" $f "*) ;; *) note="$note needs $f";; esac
    done
    case "$cpu" in
      *-sve128)
        if [ -n "$keep" ]; then
          case "$needs" in *" SVE "*) ;; *) note="$note expected an SVE variant with GGML_CPU_ARM_SVE128=1";; esac
        else
          want=armv8.2_2
          case "$has" in *" I8MM "*) want=armv8.6_0 ;; esac
          if [ "$loaded" != "$want" ]; then
            note="$note expected $want on 128-bit SVE"
          fi
        fi
        ;;
    esac
    built="$(printf '%s\n' "$iqk" | sed -n 's/^built types \([^ ,]*\).*/\1/p')"
    enabled="$(printf '%s\n' "$iqk" | sed -n 's/.*enabled \([^ ,]*\).*/\1/p')"
    if [ -n "$built" ] && [ "$built" != "0x0" ] && [ "$enabled" != "$built" ]; then
      note="$note iqk built but not enabled by default"
    fi
  fi
  printf '%-18s %-16s %-6s %s\n' "$entry" "${loaded:-none}" "$rc" "${note:-ok}${iqk:+ (iqk $iqk)}"
  if [ -n "$note" ]; then
    fail=1
    printf '%s\n' "$out" | tail -n 15 >&2
  fi
done
if [ "$fail" != 0 ]; then
  echo "ERROR: a CPU variant crashed or was picked for a CPU that lacks its features (see above)" >&2
fi
exit "$fail"
