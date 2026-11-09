# How to Use

Cosmos3-Nano-Sim-Transfer turns a control video (edges, or any precomputed control frames) plus a
prompt into a photoreal video. It generates one latent frame at a time with a KV cache, using a
distilled 4-step sampler with no CFG. It uses the Wan2.2 TI2V VAE and needs no separate text
encoder: the prompt goes through the transformer's own text path.

## Download weights

- Transformer: https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main/transformer
  - merge the shards into one file (drops `lm_head`, which video generation never reads):

    ```sh
    python scripts/convert_cosmos3.py <checkpoint>/transformer cosmos3_nano_sim_transfer_bf16.safetensors
    ```

  - optional GGUF: `sd-cli -M convert -m cosmos3_nano_sim_transfer_bf16.safetensors -o cosmos3_nano_sim_transfer_q8_0.gguf --type q8_0`
- VAE: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/vae/wan2.2_vae.safetensors

The weights are under the NVIDIA OpenMDW License 1.1; the ported code carries the same notice
(see `NOTICE` and `licenses/OpenMDW-1.1.txt`).

## Control input

- `--control-video <dir>`: control frames (for example Canny edge maps, which the model was trained
  on), one image per frame in lexicographic order.
- `--ref-video <dir>`: plain RGB frames. Edge control is derived from them with a fast gradient
  magnitude filter (grayscale, light blur, Sobel, percentile normalization) instead of Canny.

## Notes

- Use `-M vid_gen`, 832x480 and `--fps 30`, which is what the model was trained for.
- Frame counts follow Wan2.2 temporal compression (`4n + 1`, for example 121).
- Prompts can be plain text or the structured JSON captions from the model card. Duration,
  frame rate and resolution are added to the prompt the same way the reference does.
- The KV history is kept in full by default. `--model-args cosmos3_window=<latent frames>,cosmos3_sink=<frames>`
  bounds it for long videos (window 2 or more, sink below the window).
- Keep flash attention on (`--flash-attn auto`, the default, picks it on CUDA); without it each frame is about 2.4x slower.

## Example

```sh
./bin/sd-cli -M vid_gen --diffusion-model cosmos3_nano_sim_transfer_bf16.safetensors --vae wan2.2_vae.safetensors \
    --control-video canny_frames -W 832 -H 480 --video-frames 121 --fps 30 -s 2025 \
    -p "A glossy deep-red rally car slides through a gravel corner in a pine forest at golden hour." -o rally.avi
```
