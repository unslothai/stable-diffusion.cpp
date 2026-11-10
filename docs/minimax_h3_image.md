# How to Use

MiniMax-H3-Image supports text-to-image and reference-image editing. It is the MiniMax-H3 diffusion transformer without the audio stream, distilled to a fixed 6-step schedule with one output head per step, a single-frame image VAE, and the MiniMax-H3 Qwen3-VL-32B text encoder. Prompt enhancement is not included; prompts are used verbatim.

## Download weights

- MiniMax-H3-Image: the pruned single-file checkpoint `minimax_h3_image_pruned_bf16.safetensors`, or a GGUF made from it (below)
- The MiniMax-H3-Image VAE
    - safetensors: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/vae
- qwen3vl_32b_minimax_h3, the MiniMax-H3 text encoder
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main

The pruned checkpoint stores the six per-step output heads stacked in `final_layer.video_out` (head 0 whole, heads 1 to 5 as offsets from it). A file with a single output head renders wrong images and is not supported. Use the MiniMax-H3-Image VAE; the MiniMax-H3 video VAE and the P-H3-Image VAE are not compatible. No audio VAE is needed.

To make a GGUF, keep the patch embedding in F32 as for MiniMax-H3; the output heads stay F32 by name:

```
.\bin\Release\sd-cli.exe -M convert -m ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors -o ..\models\diffusion_models\minimax_h3_image_pruned-Q4_K.gguf --type q4_K --tensor-type-rules "(^|\.)video_patch_proj\.weight$=f32,(^|\.)adaln_t_table$=f32"
```

## Examples

### Text-to-image

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned-Q4_K.gguf --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image -p "a lovely cat holding a sign says 'sd.cpp'" -v --offload-to-cpu --rng cpu -H 2048 -W 2048
```

### Image editing

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned-Q4_K.gguf --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image -r source.png -r reference.png -p "Give the cat in <Picture 1> the scarf from <Picture 2>, preserving its face and the background." -v --offload-to-cpu --rng cpu -H 2048 -W 2048
```

Reference images are numbered in argument order as `<Picture 1>`, `<Picture 2>`, and so on. Each reference keeps its own size while its short edge is between 1024 and 2048 pixels and is otherwise scaled to the nearer bound; the text encoder sees it with a 1024-pixel short edge. References are never resized to the output size.

Use dimensions divisible by 32. The model samples its fixed schedule (1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0) with CFG 1 whatever `--steps` says; `--sigmas minimax-h3-image` names the same schedule. The upstream release generates anything under 4 MP (2048 × 2048) on a larger canvas of the same aspect and resizes the result back down; sd.cpp renders at the requested size, so request at least 2048 × 2048 for the release's quality. The VAE always tiles as the release does (256-pixel tiles, overlaps of at least 64 pixels); `SD_H3_IMAGE_VAE_TILES=0` uses the generic tiler instead.
