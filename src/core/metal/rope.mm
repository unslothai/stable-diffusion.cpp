#include "core/ggml_extensions.h"
#include "ggml-ext-metal.h"
#include "rope_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <memory>
#include <vector>

namespace sd_metal_rope {

    struct RopeParams {
        int32_t d, heads, tokens, rot_half;
        uint64_t x_nb[4], pe_nb[4];
        int32_t interleaved;
    };
    static_assert(sizeof(RopeParams) == 88, "RoPE argument layout mismatch");

    struct Kernels {
        id<MTLComputePipelineState> apply = nil;
        ggml_ext_metal_buffer_fn buffer   = nullptr;
        ggml_ext_metal_submit_fn submit   = nullptr;

        ~Kernels() { [apply release]; }
    };

    struct Execution {
        const ggml_ext_execution_context* context;
        ggml_tensor* const* nodes;
        int count;
        Kernels* kernels;
    };

    static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        if (!sd_ggml_rope_describe(node))
            return false;
        const auto* x  = node->src[0];
        const auto* pe = node->src[1];
        for (int i = 0; i < 4; ++i) {
            if (x->nb[i] % sizeof(float) || pe->nb[i] % sizeof(float))
                return false;
        }
        return x->ne[0] <= INT_MAX && x->ne[1] <= INT_MAX && x->ne[2] <= INT_MAX &&
               x->ne[3] <= UINT32_MAX;
    }

    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* bytes, void*) {
        const auto* output = nodes[count - 1];
        // RoPE outputs cannot reuse allocator input storage; existing storage may
        // overlap an input.
        *bytes = output->data || output->view_src ? ggml_nbytes(output) : 0;
        return GGML_STATUS_SUCCESS;
    }

    static bool bind_tensor(Kernels* kernels, const ggml_tensor* tensor, ggml_ext_metal_buffer& result) {
        auto buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !tensor->data)
            return false;
        const size_t offset =
            uintptr_t(tensor->data) - uintptr_t(ggml_backend_buffer_get_base(buffer));
        return kernels->buffer(buffer, offset, ggml_nbytes(tensor), &result);
    }

    // Host-visible storage is compared by address: mapped buffers expose
    // overlapping MTLBuffer windows, and separate host-pointer buffers can wrap the
    // same memory. Private storage has no host address.
    static bool overlaps(const ggml_ext_metal_buffer& a, size_t a_size, const ggml_ext_metal_buffer& b, size_t b_size) {
        auto* a_host = static_cast<char*>([(id<MTLBuffer>)a.buffer contents]);
        auto* b_host = static_cast<char*>([(id<MTLBuffer>)b.buffer contents]);
        if (a_host && b_host) {
            const auto x = uintptr_t(a_host) + a.offset,
                       y = uintptr_t(b_host) + b.offset;
            return x < y + b_size && y < x + a_size;
        }
        return a.buffer == b.buffer && a.offset < b.offset + b_size &&
               b.offset < a.offset + a_size;
    }

    static void set_buffer(id<MTLComputeCommandEncoder> encoder,
                           const ggml_ext_metal_buffer& buffer,
                           int index) {
        [encoder setBuffer:(id<MTLBuffer>)buffer.buffer
                    offset:buffer.offset
                   atIndex:index];
    }

    static ggml_status encode(void* command_buffer, void* user_data) {
        auto& e            = *static_cast<Execution*>(user_data);
        auto* kernels      = e.kernels;
        const auto* rope   = e.nodes[0];
        const auto* x      = rope->src[0];
        const auto* pe     = rope->src[1];
        const auto* output = e.nodes[e.count - 1];
        ggml_ext_metal_buffer bx, bpe, dst;
        if (!bind_tensor(kernels, x, bx) || !bind_tensor(kernels, pe, bpe) ||
            !bind_tensor(kernels, output, dst))
            return GGML_STATUS_FAILED;
        const size_t bytes = ggml_nbytes(output);
        const bool alias   = overlaps(dst, bytes, bx, ggml_nbytes(x)) ||
                             overlaps(dst, bytes, bpe, ggml_nbytes(pe));
        auto target        = dst;
        if (alias && (!e.context->workspace ||
                      !kernels->buffer(e.context->workspace, 0, bytes, &target)))
            return GGML_STATUS_FAILED;

        RopeParams p = {};
        p.d          = int32_t(x->ne[0]);
        p.heads      = int32_t(x->ne[1]);
        p.tokens     = int32_t(x->ne[2]);
        p.rot_half   = int32_t(pe->ne[2]);
        for (int i = 0; i < 4; ++i) {
            p.x_nb[i]  = x->nb[i];
            p.pe_nb[i] = pe->nb[i];
        }
        std::memcpy(&p.interleaved, ggml_ext_op_attributes(rope),
                    sizeof(p.interleaved));

        id<MTLCommandBuffer> command         = (id<MTLCommandBuffer>)command_buffer;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:kernels->apply];
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        set_buffer(encoder, bx, 1);
        set_buffer(encoder, bpe, 2);
        set_buffer(encoder, target, 3);
        const NSUInteger threads = std::max<NSUInteger>(
            1, std::min<NSUInteger>(p.d / 2,
                                    kernels->apply.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreadgroups:MTLSizeMake(x->ne[2], x->ne[1], x->ne[3])
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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

    static ggml_status execute(const ggml_ext_execution_context* context,
                               ggml_tensor* const* nodes,
                               int count,
                               void*,
                               void* user_data) {
        Execution e = {context, nodes, count, static_cast<Kernels*>(user_data)};
        return e.kernels->submit(context->backend, encode, &e);
    }

}  // namespace sd_metal_rope

void sd_ggml_register_metal_rope(ggml_backend_dev_t device) {
    using namespace sd_metal_rope;
    auto reg        = ggml_backend_dev_backend_reg(device);
    auto get_device = reinterpret_cast<ggml_ext_metal_device_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_device_v1"));
    auto get_buffer = reinterpret_cast<ggml_ext_metal_buffer_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_buffer_v1"));
    auto submit = reinterpret_cast<ggml_ext_metal_submit_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_submit_v1"));
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
        // ggml's default compile options, so results match the native Metal kernels
        // around this op.
        MTLCompileOptions* options = [MTLCompileOptions new];
        id<MTLLibrary> library     = [metal
            newLibraryWithSource:[NSString
                                     stringWithUTF8String:sd_metal_rope_source]
                         options:options
                           error:&error];
        [options release];
        if (!library) {
            NSLog(@"sd.cpp Metal RoPE compilation failed: %@", error);
            return;
        }
        id<MTLFunction> function =
            [library newFunctionWithName:@"sd_rope_apply_f32"];
        kernels->apply = function
                             ? [metal newComputePipelineStateWithFunction:function
                                                                    error:&error]
                             : nil;
        [function release];
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
        static std::vector<std::unique_ptr<Kernels>> registrations;
        registrations.push_back(std::move(kernels));
    }
}
