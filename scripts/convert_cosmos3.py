#!/usr/bin/env python3
"""Merge the sharded Cosmos3 transformer into one safetensors file for sd.cpp.

    python scripts/convert_cosmos3.py <checkpoint>/transformer cosmos3_nano_sim_transfer_bf16.safetensors

Drops the tensors the video path never reads (lm_head, the understanding-path
final norm). Weights are streamed, so memory use stays at one copy buffer.
For GGUF: sd-cli -M convert -m <out>.safetensors -o <out>.gguf --type q8_0
"""

import argparse
import json
import struct
from pathlib import Path

UNUSED = {"lm_head.weight", "norm.weight"}
COPY_BUFFER_SIZE = 8 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transformer_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    index = json.loads((args.transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))

    entries = []
    for shard in shards:
        path = args.transformer_dir / shard
        header, data_start = read_header(path)
        for name, info in header.items():
            if name not in UNUSED:
                entries.append((name, info, path, data_start))
    entries.sort(key=lambda e: e[0])

    out_header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, info, _, _ in entries:
        begin, end = info["data_offsets"]
        out_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)

    with open(args.output, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name, info, path, data_start in entries:
            begin, end = info["data_offsets"]
            with open(path, "rb") as src:
                src.seek(data_start + begin)
                remaining = end - begin
                while remaining > 0:
                    chunk = src.read(min(COPY_BUFFER_SIZE, remaining))
                    if not chunk:
                        raise IOError(f"{path}: truncated tensor {name}")
                    out.write(chunk)
                    remaining -= len(chunk)
    print(f"wrote {len(entries)} tensors, {offset / 2**30:.1f} GiB, to {args.output}")


if __name__ == "__main__":
    main()
