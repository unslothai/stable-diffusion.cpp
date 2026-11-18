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

struct RmsRopeParams {
    RopeParams rope;
    float eps;
    float output_scale;
};

// rms_norm(x) * weight, then RoPE as sd_rope_apply_f32, then optional scale and cast to T.
// The reduction follows ggml's kernel_rms_norm_mul_f32_4 so the result matches the unfused graph.
template <typename T>
kernel void sd_rms_norm_mul_rope(
    constant RmsRopeParams& p [[buffer(0)]],
    device const char* x [[buffer(1)]],
    device const char* pe [[buffer(2)]],
    device const float4* weight [[buffer(3)]],
    device char* dst [[buffer(4)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort3 ntg [[threads_per_threadgroup]]) {
    constant RopeParams& r = p.rope;
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    const ulong head  = tgpig.x;
    const ulong token = tgpig.y;
    const ulong batch = tgpig.z;
    const int n4      = r.d / 4;

    device const float4* row = (device const float4*)(x + head * r.x_nb[1] + token * r.x_nb[2] + batch * r.x_nb[3]);

    float sumf = 0.0f;
    for (int i = tpitg.x; i < n4; i += ntg.x) {
        sumf += dot(row[i], row[i]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / r.d;
    const float scale = 1.0f / sqrt(mean + p.eps);

    threadgroup float4* y = (threadgroup float4*)(shmem + 32);
    for (int i = tpitg.x; i < n4; i += ntg.x) {
        y[i] = (row[i] * scale) * weight[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const float* v = shmem + 32;
    device const char* pos     = pe + token * r.pe_nb[3];
    device T* out              = (device T*)dst + (token + r.tokens * (head + r.heads * batch)) * r.d;

    for (int j = tpitg.x; j < r.d / 2; j += ntg.x) {
        const int2 i   = rope_pair(r, j);
        const float2 o = rope_rotate(r, pos, j, v[i.x], v[i.y]);
        out[i.x]       = T(o.x * p.output_scale);
        out[i.y]       = T(o.y * p.output_scale);
    }
}

typedef decltype(sd_rms_norm_mul_rope<float>) sd_rms_norm_mul_rope_t;

template [[host_name("sd_rms_norm_mul_rope_f32")]] kernel sd_rms_norm_mul_rope_t sd_rms_norm_mul_rope<float>;
template [[host_name("sd_rms_norm_mul_rope_f16")]] kernel sd_rms_norm_mul_rope_t sd_rms_norm_mul_rope<half>;
