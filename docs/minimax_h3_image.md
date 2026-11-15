# MiniMax-H3-Image

MiniMax-H3-Image supports text-to-image generation and reference-image editing
with the same checkpoint. It reuses the MiniMax-H3 text encoder and uses a
dedicated image VAE.

## Model files

Pass the three MiniMax-H3-Image components separately:

- `--diffusion-model`: MiniMax-H3-Image diffusion transformer
- `--vae`: MiniMax-H3-Image VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint

Use the MiniMax-H3 text encoder, including its Qwen3-VL vision tower and
DeepStack mergers. An existing MiniMax-H3 text encoder can be reused. If the
vision tower is stored separately, pass it with `--llm_vision`.

Use a single-file diffusion checkpoint with the six-step PDD output heads
included, such as `minimax_h3_image_pruned_bf16.safetensors` or its INT8 ConvRot
version. Use the MiniMax-H3-Image VAE; the MiniMax-H3 video VAE and P-H3-Image
VAE are incompatible. No audio VAE or P-H3 text connector is needed.

### Download weights

- Download minimax_h3_image_pruned
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae

## Text-to-image

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A silver tabby kitten holding a sign reading 'sd.cpp', natural daylight" --cfg-scale 1.0 --steps 6 --sampling-method euler -v -W 1024 -H 1024 --offload-to-cpu --rng cpu --diffusion-fa --vae-tiling
```

Use Euler, CFG 1 and 6 steps; without `--steps`, MiniMax-H3-Image runs 6. The
default scheduler is `simple`, as in the reference workflows: at 6 steps it
lands within 9e-4 of the model's PDD sigma grid. `--scheduler discrete` or
`--sigmas minimax-h3-image` samples on the exact grid instead. Without `-W` and
`-H`, text-to-image renders 2048 × 2048, the reference template's 4 MP square.
Prompt expansion is not included.

## Reference-image editing

Add `--ref-image` (or `-r`) for reference-image editing:

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Edit <Picture 1>. Put a red scarf on the cat, preserving its face and the background." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 --steps 6 --sampling-method euler -v -W 1024 -H 1024 --offload-to-cpu --rng cpu --diffusion-fa --vae-tiling
```

`--ref-image` can be repeated. Reference images are numbered in argument order
as `<Picture 1>`, `<Picture 2>`, and so on, with up to 16 references supported
by the model. Prompts are passed through without an automatic editing prefix.

As in the reference edit template, `<Picture 1>` is first scaled to 4 MP (area
filter) and then to the nearest smaller multiple of 32 on each side (scaled to
cover it and centre-cropped). Without `-W` and `-H` the output takes that size,
for example 2720 × 1536 for a 1360 × 768 picture.

Reference images keep their aspect ratio. As in the reference workflows, each
consumer sizes a reference on its own: the VAE sees it with the short edge
scaled into 1024-2048 pixels and the text encoder with a 1024-pixel short edge,
both sides rounded to multiples of 32 in each case. A reference already at that
size is used as it is; any other is resized with Lanczos on 8-bit values. Nothing
is cropped. The reference latent is the VAE posterior sample (seed 42) in
float32, noised to t = 0.999.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Start at 1024 × 1024. Larger canvases and more references increase memory use.
- Use quantized weights, `--offload-to-cpu`, and `--vae-tiling` to reduce GPU
  memory requirements.
- The image VAE always decodes and encodes in the release tile layout (256-pixel
  tiles, overlaps of at least 64 pixels, linear blends), with or without
  `--vae-tiling`: the decoder attends over a whole tile, so an untiled decode
  gives a different image. `SD_H3_IMAGE_VAE_TILES=0` uses the generic path.
- The release generates anything under 4 MP on a larger canvas of the same
  aspect ratio and resizes the result back down. sd.cpp renders at the
  requested size; request 2048 × 2048 (or another 4 MP size) for the release
  behaviour.
- To convert the checkpoint to GGUF, keep the patch embedding in F32 as for
  MiniMax-H3 (the output heads keep their source type by name):
  `sd-cli -M convert -m minimax_h3_image_pruned_bf16.safetensors -o minimax_h3_image_pruned-Q4_K.gguf --type q4_K --tensor-type-rules "(^|\.)video_patch_proj\.weight$=f32,(^|\.)adaln_t_table$=f32"`
