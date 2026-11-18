#include "core/ggml_extensions.h"
#include "ggml-ext-metal.h"
#include "rope_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace sd_metal_rope {

    struct RopeParams {
        int32_t d, heads, tokens, rot_half;
        uint64_t x_nb[4], pe_nb[4];
        int32_t interleaved;
    };
    struct RmsRopeParams {
        RopeParams rope;
        float eps;
        float output_scale;
    };
    static_assert(sizeof(RopeParams) == 88 && sizeof(RmsRopeParams) == 96, "RoPE argument layout mismatch");

    struct Kernels {
        id<MTLComputePipelineState> apply    = nil;
        id<MTLComputePipelineState> fused    = nil;
        id<MTLComputePipelineState> fused_16 = nil;
        ggml_ext_metal_buffer_fn buffer      = nullptr;
        ggml_ext_metal_submit_fn submit      = nullptr;

        ~Kernels() {
            [apply release];
            [fused release];
            [fused_16 release];
        }
    };

    struct Execution {
        const ggml_ext_execution_context* context;
        ggml_tensor* const* nodes;
        int count;
        Kernels* kernels;
    };

    static float param(const ggml_tensor* node, int i) {
        float value;
        std::memcpy(&value, node->op_params + i, sizeof(value));
        return value;
    }

    static bool is_rope(const ggml_tensor* node) {
        return node && ggml_ext_op_get(node) == sd_ggml_rope_op();
    }

    static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        if (!sd_ggml_rope_describe(node))
            return false;
        const auto* x  = node->src[0];
        const auto* pe = node->src[1];
        for (int i = 0; i < 4; ++i) {
            if (x->nb[i] % sizeof(float) || pe->nb[i] % sizeof(float))
                return false;
        }
        return x->ne[0] <= INT_MAX && x->ne[1] <= INT_MAX && x->ne[2] <= INT_MAX && x->ne[3] <= UINT32_MAX;
    }

    static bool fusion_disabled() {
        static const bool disabled = std::getenv("GGML_METAL_FUSION_DISABLE") != nullptr;
        return disabled;
    }

    static bool fused_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope) {
        if (!norm || !mul || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || mul->src[0] != norm ||
            !is_rope(rope) || rope->src[0] != mul || !supports(nullptr, rope, nullptr))
            return false;
        const auto* x      = norm->src[0];
        const auto* weight = mul->src[1];
        // The row is staged in threadgroup memory and reduced as float4, like ggml's rms_norm_f32_4.
        return x && weight && weight != norm && x->type == GGML_TYPE_F32 && norm->type == GGML_TYPE_F32 &&
               mul->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
               x->ne[0] % 4 == 0 && x->ne[0] <= 1024 && ggml_are_same_shape(x, norm) && ggml_are_same_shape(norm, mul) &&
               weight->ne[0] == x->ne[0] && ggml_nelements(weight) == x->ne[0] && ggml_is_contiguous(weight);
    }

    static bool cast_supports(const ggml_tensor* rope, const ggml_tensor* scale, const ggml_tensor* copy) {
        return (!scale || (scale->op == GGML_OP_SCALE && scale->src[0] == rope && scale->type == GGML_TYPE_F32 && param(scale, 1) == 0.f)) &&
               copy && copy->op == GGML_OP_CPY && copy->src[0] == (scale ? scale : rope) && copy->type == GGML_TYPE_F16 &&
               ggml_is_contiguous(copy) && ggml_are_same_shape(rope, copy);
    }

    static int match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
        auto* g     = const_cast<ggml_cgraph*>(graph);
        const int n = ggml_graph_n_nodes(g);
        if (fusion_disabled() || first + 2 >= n || !fused_supports(ggml_graph_node(g, first), ggml_graph_node(g, first + 1), ggml_graph_node(g, first + 2)))
            return 0;
        if (first + 4 < n && cast_supports(ggml_graph_node(g, first + 2), ggml_graph_node(g, first + 3), ggml_graph_node(g, first + 4)))
            return 5;
        return first + 3 < n && cast_supports(ggml_graph_node(g, first + 2), nullptr, ggml_graph_node(g, first + 3)) ? 4 : 3;
    }

    // Moves the norm / mul producers (and a scale before the cast) next to the RoPE so match sees one chain.
    static void optimize(ggml_backend_dev_t, ggml_cgraph* graph, void*) {
        if (fusion_disabled())
            return;
        for (int last = 2; last < ggml_graph_n_nodes(graph); ++last) {
            const auto* output = ggml_graph_node(graph, last);
            const auto* source = output->op == GGML_OP_CPY ? output->src[0] : nullptr;
            const auto* scale  = source && source->op == GGML_OP_SCALE ? source : nullptr;
            const auto* rope   = source ? (scale ? scale->src[0] : source) : output;
            if (!is_rope(rope))
                continue;
            const auto* mul  = rope->src[0];
            const auto* norm = mul->src[0];
            if (!norm || !fused_supports(norm, mul, rope))
                continue;
            const int count = output == rope ? 3 : cast_supports(rope, scale, output) ? (scale ? 5 : 4)
                                                                                      : 0;
            if (!count)
                continue;
            const ggml_tensor* nodes[] = {norm, mul, rope, scale ? scale : output, output};
            int indices[5], next = count - 1;
            for (int i = last; i >= std::max(0, last - 64) && next >= 0; --i)
                if (ggml_graph_node(graph, i) == nodes[next])
                    indices[next--] = i;
            if (next >= 0)
                continue;
            int first = indices[0];
            for (int i = 1; i < count; ++i) {
                if (!ggml_ext_graph_delay(graph, first, i, indices[i]))
                    break;
                first = indices[i] - i;
            }
        }
    }

    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* bytes, void*) {
        const auto* output = nodes[count - 1];
        // RoPE outputs and owning casts cannot reuse allocator input storage; existing storage may overlap an input.
        *bytes = output->data || output->view_src ? ggml_nbytes(output) : 0;
        return GGML_STATUS_SUCCESS;
    }

    static bool bind_tensor(Kernels* kernels, const ggml_tensor* tensor, ggml_ext_metal_buffer& result) {
        auto buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !tensor->data)
            return false;
        const size_t offset = uintptr_t(tensor->data) - uintptr_t(ggml_backend_buffer_get_base(buffer));
        return kernels->buffer(buffer, offset, ggml_nbytes(tensor), &result);
    }

    // Host-visible storage is compared by address: mapped buffers expose overlapping MTLBuffer windows, and
    // separate host-pointer buffers can wrap the same memory. Private storage has no host address.
    static bool overlaps(const ggml_ext_metal_buffer& a, size_t a_size, const ggml_ext_metal_buffer& b, size_t b_size) {
        auto* a_host = static_cast<char*>([(id<MTLBuffer>)a.buffer contents]);
        auto* b_host = static_cast<char*>([(id<MTLBuffer>)b.buffer contents]);
        if (a_host && b_host) {
            const auto x = uintptr_t(a_host) + a.offset, y = uintptr_t(b_host) + b.offset;
            return x < y + b_size && y < x + a_size;
        }
        return a.buffer == b.buffer && a.offset < b.offset + b_size && b.offset < a.offset + a_size;
    }

    static void set_buffer(id<MTLComputeCommandEncoder> encoder, const ggml_ext_metal_buffer& buffer, int index) {
        [encoder setBuffer:(id<MTLBuffer>)buffer.buffer offset:buffer.offset atIndex:index];
    }

    static ggml_status encode(void* command_buffer, void* user_data) {
        auto& e            = *static_cast<Execution*>(user_data);
        auto* kernels      = e.kernels;
        const bool fused   = e.count > 1;
        const auto* rope   = e.nodes[fused ? 2 : 0];
        const auto* x      = fused ? e.nodes[0]->src[0] : rope->src[0];
        const auto* pe     = rope->src[1];
        const auto* weight = fused ? e.nodes[1]->src[1] : nullptr;
        const auto* output = e.nodes[e.count - 1];
        ggml_ext_metal_buffer bx, bpe, bw, dst;
        if (!bind_tensor(kernels, x, bx) || !bind_tensor(kernels, pe, bpe) || !bind_tensor(kernels, output, dst) ||
            (weight && !bind_tensor(kernels, weight, bw)))
            return GGML_STATUS_FAILED;
        const size_t bytes = ggml_nbytes(output);
        const bool alias   = overlaps(dst, bytes, bx, ggml_nbytes(x)) || overlaps(dst, bytes, bpe, ggml_nbytes(pe)) ||
                             (weight && overlaps(dst, bytes, bw, ggml_nbytes(weight)));
        auto target        = dst;
        if (alias && (!e.context->workspace || !kernels->buffer(e.context->workspace, 0, bytes, &target)))
            return GGML_STATUS_FAILED;

        RmsRopeParams p = {};
        p.rope.d        = int32_t(x->ne[0]);
        p.rope.heads    = int32_t(x->ne[1]);
        p.rope.tokens   = int32_t(x->ne[2]);
        p.rope.rot_half = int32_t(pe->ne[2]);
        for (int i = 0; i < 4; ++i) {
            p.rope.x_nb[i]  = x->nb[i];
            p.rope.pe_nb[i] = pe->nb[i];
        }
        std::memcpy(&p.rope.interleaved, ggml_ext_op_attributes(rope), sizeof(p.rope.interleaved));

        id<MTLCommandBuffer> command         = (id<MTLCommandBuffer>)command_buffer;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        set_buffer(encoder, bx, 1);
        set_buffer(encoder, bpe, 2);
        if (!fused) {
            [encoder setComputePipelineState:kernels->apply];
            [encoder setBytes:&p.rope length:sizeof(p.rope) atIndex:0];
            set_buffer(encoder, target, 3);
            const NSUInteger threads = std::max<NSUInteger>(1, std::min<NSUInteger>(p.rope.d / 2, kernels->apply.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreadgroups:MTLSizeMake(x->ne[2], x->ne[1], x->ne[3]) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        } else {
            id<MTLComputePipelineState> pipeline = output->type == GGML_TYPE_F16 ? kernels->fused_16 : kernels->fused;
            p.eps                                = param(e.nodes[0], 0);
            p.output_scale                       = e.count == 5 ? param(e.nodes[3], 0) : 1.f;
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            set_buffer(encoder, bw, 3);
            set_buffer(encoder, target, 4);
            // Same threadgroup width as ggml's rms_norm dispatch, so the reduction order matches.
            const int n4  = p.rope.d / 4;
            const int max = int(pipeline.maxTotalThreadsPerThreadgroup);
            int threads   = 32;
            while (threads < n4 && threads < max)
                threads *= 2;
            threads = std::min(std::min(threads, max), (n4 + 31) / 32 * 32);
            [encoder setThreadgroupMemoryLength:(32 + p.rope.d) * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(x->ne[1], x->ne[2], x->ne[3]) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        [encoder endEncoding];
        if (alias) {
            id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
            [copy copyFromBuffer:(id<MTLBuffer>)target.buffer
                     sourceOffset:target.offset
                         toBuffer:(id<MTLBuffer>)dst.buffer
                destinationOffset:dst.offset
                             size:bytes];
            [copy endEncoding];
        }
        return GGML_STATUS_SUCCESS;
    }

    static ggml_status execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int count, void*, void* user_data) {
        Execution e = {context, nodes, count, static_cast<Kernels*>(user_data)};
        return e.kernels->submit(context->backend, encode, &e);
    }

}

void sd_ggml_register_metal_rope(ggml_backend_dev_t device) {
    using namespace sd_metal_rope;
    auto reg        = ggml_backend_dev_backend_reg(device);
    auto get_device = reinterpret_cast<ggml_ext_metal_device_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_device_v1"));
    auto get_buffer = reinterpret_cast<ggml_ext_metal_buffer_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_buffer_v1"));
    auto submit     = reinterpret_cast<ggml_ext_metal_submit_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_submit_v1"));
    if (!get_device || !get_buffer || !submit)
        return;
    @autoreleasepool {
        id<MTLDevice> metal = (id<MTLDevice>)get_device(device);
        if (!metal)
            return;
        auto kernels    = std::make_unique<Kernels>();
        kernels->buffer = get_buffer;
        kernels->submit = submit;
        NSError* error  = nil;
        // ggml's default compile options, so results match the native Metal kernels around this op.
        MTLCompileOptions* options = [MTLCompileOptions new];
        id<MTLLibrary> library     = [metal newLibraryWithSource:[NSString stringWithUTF8String:sd_metal_rope_source] options:options error:&error];
        [options release];
        if (!library) {
            NSLog(@"sd.cpp Metal RoPE compilation failed: %@", error);
            return;
        }
        auto pipeline = [&](NSString* name) {
            id<MTLFunction> function           = [library newFunctionWithName:name];
            id<MTLComputePipelineState> result = function ? [metal newComputePipelineStateWithFunction:function error:&error] : nil;
            [function release];
            return result;
        };
        kernels->apply    = pipeline(@"sd_rope_apply_f32");
        kernels->fused    = pipeline(@"sd_rms_norm_mul_rope_f32");
        kernels->fused_16 = pipeline(@"sd_rms_norm_mul_rope_f16");
        [library release];
        if (!kernels->apply) {
            NSLog(@"sd.cpp Metal RoPE pipeline unavailable: %@", error);
            return;
        }
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.rope_apply.metal.v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_rope_op();
        desc.priority             = 100;
        desc.supports             = supports;
        desc.prepare              = prepare;
        desc.destroy              = [](void*, void*) {};
        desc.execute              = execute;
        desc.user_data            = kernels.get();
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
        // The reduction keeps one partial sum per 32-wide SIMD group.
        auto fusable = [](id<MTLComputePipelineState> p) {
            return p && p.threadExecutionWidth == 32;
        };
        if (fusable(kernels->fused) && fusable(kernels->fused_16)) {
            desc.name      = "sd.rms_norm_mul_rope.metal.v1";
            desc.op        = GGML_OP_RMS_NORM;
            desc.custom_op = nullptr;
            desc.priority  = 200;
            desc.supports  = nullptr;
            desc.match     = match;
            desc.optimize  = optimize;
            GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
        }
        static std::vector<std::unique_ptr<Kernels>> registrations;
        registrations.push_back(std::move(kernels));
    }
}
