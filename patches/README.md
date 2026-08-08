# Patches applied to the published prebuilts

`.github/workflows/unsloth-sd-prebuilt.yml` builds leejet's source **at a published release
tag**, aged for at least six hours, rather than this fork's `master`. That is deliberate: what we
publish should be traceable to a specific upstream release rather than to whatever a fork happened
to contain that day.

Every `*.patch` in this directory is applied to that tree, in filename order, before the build. A
release built with a non-empty patch set is tagged `<upstream tag>-u<id>`, where `<id>` is the
first seven hex digits of the sha256 of the concatenated patch files. So:

- an unpatched build keeps the plain upstream tag, exactly as before,
- changing, adding or removing a patch changes the tag, which is what makes the pipeline rebuild
  and republish rather than see an existing release and skip,
- the tag on the box says whether the box is stock, and which patch set it carries.

`sd-prebuilt-manifest.json` in each release records the patch filenames and the id.

## Rules

**Every patch must be open upstream.** These exist to close the gap between "fixed" and
"released", not to carry a private fork. Put the upstream pull request in the header comment of
the patch file.

**A patch that no longer applies fails the build.** This is intentional, not a bug to route
around. `git apply --check` runs on all of them before any is applied, and the run stops with the
patch name. That happens for exactly two reasons: upstream merged it, in which case delete the
file, or upstream changed the surrounding code, in which case refresh the patch against the new
tag and re-verify it.

**Delete on merge.** Once the fix ships in an upstream release the patch is dead weight, and
leaving it in place means the next release silently carries a duplicate of code upstream already
has.

## Current set

| patch | upstream PR | what it fixes |
|---|---|---|
| `0001-spare-1d-norm-weights-from-blanket-quant.patch` | leejet/stable-diffusion.cpp#1861 | a blanket `--type` quantizes 1-D norm gains whose length divides the block size, which silently destroys MiniMax-H3 output (LPIPS 0.981 against its own bf16 render) |
| `0002-h3-cfg-scale-and-audio-vae-on-cpu.patch` | leejet/stable-diffusion.cpp#1862 | H3 aborts on the default `--cfg-scale 7.0`, and aborts again on `--vae-on-cpu` because the audio VAE reaches the CPU conv1d path with F32 kernels |
| `0003-h3-img-gen-mode-guard.patch` | leejet/stable-diffusion.cpp#1863 | an H3 checkpoint run without `--mode vid_gen` core dumps on a raw ggml assert instead of saying which flag is missing |
