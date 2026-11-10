// Standalone check of the auto-fit placement planner:
// g++ -std=c++17 -I src -I ggml/include scripts/backend_fit_plan_test.cpp && ./a.out
#include <cstdio>
#include <cstdlib>

#include "core/backend_fit_plan.h"

using namespace sd::backend_fit;

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

static std::vector<Component> components(int64_t dit_mib, int64_t te_mib, int64_t vae_mib) {
    return {
        {ComponentKind::DIT, "DiT", dit_mib * MiB, 2048 * MiB, 512 * MiB},
        {ComponentKind::CONDITIONER, "Conditioner", te_mib * MiB, 2048 * MiB, 512 * MiB},
        {ComponentKind::VAE, "VAE", vae_mib * MiB, 1024 * MiB, 64 * MiB},
    };
}

static Device metal(int64_t free_mib, bool shared) {
    Device d;
    d.name         = "MTL0";
    d.description  = "Apple M-series";
    d.free_bytes   = free_mib * MiB;
    d.budget_bytes = (free_mib - 512) * MiB;  // auto-fit default margin
    d.type         = GGML_BACKEND_DEVICE_TYPE_GPU;
#ifdef SD_BACKEND_FIT_PLAN_SHARED_MEMORY
    d.shares_host_memory = shared;
#else
    (void)shared;
#endif
    return d;
}

// Bytes that land in the one physical pool: RAM-resident plus anything on a unified-memory device.
static int64_t host_pool_bytes(const Plan& plan, const std::vector<Component>& comps) {
    int64_t total = 0;
    for (size_t ci = 0; ci < comps.size(); ++ci) {
        if (plan.decisions[ci].params_location == ParamsLocation::CPU ||
            plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU ||
            plan.decisions[ci].params_location == ParamsLocation::OTHER_GPU) {
            total += comps[ci].params_bytes;
        }
    }
    return total;
}

static const char* loc(ParamsLocation l) {
    switch (l) {
        case ParamsLocation::MAIN_GPU:
            return "device";
        case ParamsLocation::CPU:
            return "ram";
        case ParamsLocation::OTHER_GPU:
            return "other";
        case ParamsLocation::DISK:
            return "disk";
    }
    return "?";
}

static void print(const char* tag, const Plan& plan, const std::vector<Component>& comps, int64_t ram_budget) {
    std::printf("%s: dit=%s te=%s vae=%s host_pool=%lld MiB ram_budget=%lld MiB\n", tag,
                loc(plan.decisions[0].params_location), loc(plan.decisions[1].params_location),
                loc(plan.decisions[2].params_location),
                (long long)(host_pool_bytes(plan, comps) / MiB), (long long)(ram_budget / MiB));
}

static int64_t ram_budget_from_free(int64_t free_mib) {
    const int64_t free_ram = free_mib * MiB;
    return std::max<int64_t>(free_ram - std::max<int64_t>(2048 * MiB, free_ram / 10), 0);  // derive_backend_specs
}

int main() {
    // 24 GB Apple silicon Mac, Qwen-Image-class model: Metal working set 18 GiB, 16 GiB of RAM free.
    {
        const auto comps = components(12288, 8192, 256);
        const int64_t rb = ram_budget_from_free(16384);
        const auto plan  = compute_plan(comps, {metal(18432, true)}, rb);
        print("24GB mac, unified", plan, comps, rb);
        CHECK(plan.valid);
        CHECK(plan.decisions[0].params_location == ParamsLocation::MAIN_GPU);
        // Weights in the shared pool must fit what is actually free, not free RAM plus the Metal working set again.
        CHECK(host_pool_bytes(plan, comps) <= rb);
        CHECK(plan.decisions[1].params_location == ParamsLocation::DISK);
    }
    // Same numbers on a discrete card (or SD_AUTOFIT_UMA=0): plan unchanged, encoder stays in RAM.
    {
        const auto comps = components(12288, 8192, 256);
        const int64_t rb = ram_budget_from_free(16384);
        const auto plan  = compute_plan(comps, {metal(18432, false)}, rb);
        print("discrete / kill switch", plan, comps, rb);
        CHECK(plan.decisions[0].params_location == ParamsLocation::MAIN_GPU);
        CHECK(plan.decisions[1].params_location == ParamsLocation::CPU);
        CHECK(plan.decisions[2].params_location == ParamsLocation::CPU);
    }
    // 64 GB Mac where everything fits on the device: unified accounting changes nothing.
    {
        const auto comps = components(12288, 8192, 256);
        const int64_t rb = ram_budget_from_free(50000);
        const auto a     = compute_plan(comps, {metal(49152, true)}, rb);
        const auto b     = compute_plan(comps, {metal(49152, false)}, rb);
        print("64GB mac, unified", a, comps, rb);
        for (size_t ci = 0; ci < comps.size(); ++ci) {
            CHECK(a.decisions[ci].params_location == ParamsLocation::MAIN_GPU);
            CHECK(a.decisions[ci].params_location == b.decisions[ci].params_location);
        }
    }
    // Denoiser too large for the working set goes to RAM; the encoder then must not also claim that RAM via Metal.
    {
        const auto comps = components(14336, 6144, 256);
        const int64_t rb = ram_budget_from_free(20480);
        const auto plan  = compute_plan(comps, {metal(12288, true)}, rb);
        print("denoiser in RAM, unified", plan, comps, rb);
        CHECK(plan.decisions[0].params_location == ParamsLocation::CPU);
        CHECK(host_pool_bytes(plan, comps) <= rb);
    }
    if (failures == 0) {
        std::printf("backend_fit_plan_test: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
