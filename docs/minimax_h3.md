# MiniMax-H3

MiniMax-H3 jointly generates video and stereo audio with a packed diffusion
transformer. The implementation supports text-to-audio-video (T2VA), optional
first-frame conditioning (I2VA), first/last-frame conditioning (FL2VA), and
image/video/audio reference conditioning (Ref2VA).

## Model files

Pass the four MiniMax-H3 components separately:

- `--diffusion-model`: MiniMax-H3 diffusion transformer
- `--vae`: MiniMax-H3 video VAE
- `--audio-vae`: MiniMax-H3 audio VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint

The text encoder must be the MiniMax-H3 variant: Qwen3-VL-32B truncated to 50
language layers and exported without the final language-model normalization.
Its Qwen3-VL vision tower, including the three DeepStack mergers, must also be
present. If the vision tower is stored separately, pass it with `--llm_vision`.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

- Download minimax_h3_fl2va/minimax_h3_ref2va
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae
- Download audio vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae

## Text-to-audio-video

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten’s energetic surfing." --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

On CUDA and ROCm, `--diffusion-fa` also enables flash attention in the video VAE decoder (a ViT),
which removes most of its attention cost without touching the text encoder. Set
`SD_H3_VAE_FLASH_ATTN=0` to decode with the previous mul_mat + softmax attention. Other backends
keep that attention by default (on Vulkan the flash attention decode was slower);
`SD_H3_VAE_FLASH_ATTN=1` turns it on there.

On NVIDIA GPUs from Ampere on, the unmasked DiT and VAE attention runs a pipelined
long-sequence variant of the ggml-cuda flash attention kernel (up to 4x faster at H3
shapes; the arithmetic per output element is unchanged). `GGML_CUDA_FA_LONGSEQ=0` restores the
stock kernel. `GGML_CUDA_FA_LONGSEQ_NCOLS=128` opts into a wider tile that is faster on A100 and L4
but not bit-identical.

Builds configured with `-DGGML_CUDA_CUDNN=ON` (the Linux CUDA prebuilt is) can run that unmasked
attention, and the `--sage-attn` attention, through cuDNN's fused attention. By default it replaces
the flash attention kernel on Ada (sm89) and B200-class (sm100) GPUs and the sage kernel on sm100 only,
where it measured faster; on an RTX 3090 (sm86) both existing kernels were faster, on an RTX 6000 Ada
the sage kernel was, and other architectures were not measured.
Only `cudnn.h` is needed at build time (`GGML_CUDA_CUDNN_INCLUDE_DIR`; the header-only
cudnn-frontend v1.26.0 is downloaded by CMake unless `GGML_CUDA_CUDNN_FRONTEND_DIR` points at a
checkout). `libcudnn.so.9` is opened when the first attention op runs and is not shipped: put a cuDNN 9
built for the same CUDA major version as the binary (cuDNN 9 for CUDA 12 for the prebuilt) on the
library path or beside the binary, or point `GGML_CUDA_CUDNN_LIB` at it (cuDNN 9.27 needs nothing
else; older 9.x releases build these kernels with NVRTC and also need that CUDA major's `libnvrtc`).
Without a loadable cuDNN, on Turing and older, for attention calls under about a million scores
(where the extra conversions cost more than cuDNN saves) and for any shape cuDNN declines, the
existing kernels run. The first call of each attention shape builds a cuDNN plan (about 0.4 to 1.3 s
on a B200, logged once). cuDNN runs F16 with F32 accumulation: for the H3 DiT at 960x544x124 it
takes about 8 ms per attention call on a B200 against 28 ms for the sage kernel and 43 ms for the
ggml flash attention kernel, and it is closer to an exact (fp64) reference than either, so frames
and audio differ slightly from the previous kernels. `GGML_CUDA_CUDNN_ATTN=0` restores them and
`GGML_CUDA_CUDNN_ATTN=1` uses cuDNN for the flash attention op on any Ampere or newer GPU;
`GGML_CUDA_CUDNN_SAGE=0` / `=1` does the same for the sage op only, and
`GGML_CUDA_CUDNN_ATTN_BF16=1` runs cuDNN in BF16 (less accurate, same speed).

The video VAE decodes one 16x16 latent tile per decoder graph by default. `SD_H3_VAE_TILE=N`
uses N x N latent tiles instead (20 decodes about 0.7 s faster at 960x544x124 on B200); the tile
seams move, so the frames differ from the default (36 dB PSNR at 20) and it is opt-in.
`SD_H3_VAE_TILE_BATCH=auto` puts several tiles into one graph, sized from free device memory (at
most 4 unless `SD_H3_VAE_TILE_BATCH_MAX` raises it), and `SD_H3_VAE_TILE_BATCH=N` forces N; the
batched projections can round differently from the per-tile decode on some GPUs, so it is
opt-in. The decoder weights stay on the device across temporal chunks
(`SD_H3_VAE_KEEP_RESIDENT=0` releases them after every chunk), and the decoder blocks use a
table-based rotary embedding and a fused SwiGLU (`SD_H3_VAE_GRAPH_OPT=0` restores the previous
graph). Each batched tile still goes through its own attention call. With one tile per graph and
flash attention, each decoder attention normalises q/k in place on the projection and one fused
RoPE op per tensor writes the head-major Q (F32) and K/V (F16) that the attention kernel reads,
replacing the table RoPE, chunk copies, permutes and casts; the frames are bit-identical
(`SD_H3_VAE_FUSED_QKV=0` restores the unfused graph). On CUDA the q/k RMS norm runs inside that
RoPE op with the same reduction as the standalone norm (`SD_H3_VAE_FUSED_QK_NORM=0` keeps it
separate).

The host side of the decode overlaps the device: each tile's blend into the frame runs on a
worker thread while the next tile computes (`SD_TILE_ASYNC_MERGE=0` blends inline; this applies
to every tiled VAE decode), each temporal chunk is trimmed, cross-faded and copied into the final
frames on a worker thread while the next chunk decodes (`SD_H3_VAE_ASYNC_ASSEMBLY=0` restores
the concatenate-at-the-end path), the rotary tables are built once per tile shape, and a tile
that allocates nothing new reuses the device free-memory reading taken earlier in the decode
instead of querying the device for every capacity check (`SD_H3_VAE_REUSE_MEMQUERY=0`). The
decoded frames are bit-identical either way.

The DiT blocks use fused ggml ops (CPU and CUDA) for the work around the matmuls and attention:
partial RoPE with the attention relayout and the K/V scale and F16 cast, per-segment adaLN
modulation and gated residuals written in place, and the MLP Linear scales folded into those ops
and the swiglu. The result is bit-identical to the unfused graph. `SD_H3_GRAPH_FAST=0` restores
the unfused graph; `SD_H3_FAST_QKV=0`, `SD_H3_FAST_MLP=0`, `SD_H3_FAST_SEGMENTS=0` and
`SD_H3_FAST_VIEWS=0` turn off one part each. With `--sage-attn` the same fused RoPE op writes the
F32 Q/K and F16 V layout the sage kernel reads (K and V carry the kv scale), so the sage path no
longer pays for the chunk / slice / rope / concat / scale / cast chain; the output is bit-identical
to the unfused sage graph. `SD_H3_FAST_SAGE_QKV=0` restores that chain.

The audio VAE's anti-aliased activations run their up/down-sampling filters as direct F32
depthwise convolutions (`CONV_2D_DW`) instead of an F16 im2col plus a matrix-vector product,
which makes the audio decode several times faster. The input is no longer rounded to F16, so
the waveform differs slightly from the previous graph (about 41 dB SNR on a 5 s clip).
`SD_H3_AUDIO_DIRECT_DW=0` restores the previous graph; backends without `CONV_2D_DW` keep it.

The long-sequence flash attention kernel, the fused cuBLAS epilogues and the fused DiT ops come from
the ggml patches in `scripts/unsloth/ggml-patches`, which the Unsloth prebuilt binaries carry. A
source build gets them by applying the patches to the `ggml` submodule before configuring
(`for p in scripts/unsloth/ggml-patches/*.patch; do git -C ggml apply "../$p"; done`); without
them the build uses the stock ggml kernels and the unfused DiT graph.

## First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "a lovely cat" -i ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

## Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Use the cat from <Picture 1> as the main character. Keep the cat’s appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat’s face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat’s head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat’s fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat’s identity unchanged." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video is a directory of image frames sorted lexicographically and is
treated as 24 fps. Repeated `--ref-video-audio` WAV files are paired by index
with repeated `--ref-video` inputs. WAV PCM (8/16/24/32-bit) and 32/64-bit
floating-point samples are accepted; audio is converted to stereo 32 kHz by the
pipeline.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images keep their aspect ratio and are only downscaled when their
pixel area exceeds the requested generation canvas.

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
`ref_audios` in `sd_vid_gen_params_t`. Each `sd_ref_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.
