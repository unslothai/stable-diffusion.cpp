#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    int d;
    int heads;
    int tokens;
    int rot_half;
    ulong x_nb[4];
    ulong pe_nb[4];
    int interleaved;
};

// Column pair j of a row: (2j, 2j + 1) when interleaved or past the rotated part, else (j, j + rot_half).
static inline int2 rope_pair(constant RopeParams& p, int j) {
    const bool split = !p.interleaved && j < p.rot_half;
    return split ? int2(j, j + p.rot_half) : int2(2 * j, 2 * j + 1);
}

// pe [2, 2, rot_half, tokens] holds [[cos, -sin], [sin, cos]] per pair; pairs from rot_half on pass through.
static inline float2 rope_rotate(constant RopeParams& p, device const char* pos, int j, float a, float b) {
    if (j >= p.rot_half) {
        return float2(a, b);
    }
    device const char* m = pos + j * p.pe_nb[2];

    const float c = *(device const float*)(m);
    const float s = *(device const float*)(m + p.pe_nb[1]);

    if (p.interleaved) {
        const float neg_s = *(device const float*)(m + p.pe_nb[0]);
        const float c1    = *(device const float*)(m + p.pe_nb[0] + p.pe_nb[1]);
        return float2(a * c + b * neg_s, a * s + b * c1);
    }
    return float2(a * c - b * s, a * s + b * c);
}

// x [d, heads, tokens, batch] with any strides; dst contiguous [d, tokens, heads * batch].
kernel void sd_rope_apply_f32(
    constant RopeParams& p [[buffer(0)]],
    device const char* x [[buffer(1)]],
    device const char* pe [[buffer(2)]],
    device float* dst [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg [[threads_per_threadgroup]]) {
    const ulong token = tgpig.x;
    const ulong head  = tgpig.y;
    const ulong batch = tgpig.z;

    device const char* row = x + head * p.x_nb[1] + token * p.x_nb[2] + batch * p.x_nb[3];
    device const char* pos = pe + token * p.pe_nb[3];
    device float* out      = dst + (token + p.tokens * (head + p.heads * batch)) * p.d;

    for (int j = tpitg.x; j < p.d / 2; j += ntg.x) {
        const int2 i   = rope_pair(p, j);
        const float a  = *(device const float*)(row + i.x * p.x_nb[0]);
        const float b  = *(device const float*)(row + i.y * p.x_nb[0]);
        const float2 o = rope_rotate(p, pos, j, a, b);
        out[i.x]       = o.x;
        out[i.y]       = o.y;
    }
}
