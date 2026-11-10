#ifndef __SD_BACKEND_FIT_PLAN_H__
#define __SD_BACKEND_FIT_PLAN_H__

// Pure placement planner for --auto-fit; no model or backend state, so it can be checked standalone.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"

#define SD_BACKEND_FIT_PLAN_SHARED_MEMORY 1

namespace sd::backend_fit {

    static constexpr int64_t MiB = 1024ll * 1024;

    enum class ComponentKind {
        DIT,
        CONDITIONER,
        VAE,
    };

    struct Component {
        ComponentKind kind;
        const char* name;
        int64_t params_bytes  = 0;
        int64_t reserve_bytes = 0;
        int64_t staging_bytes = 0;
    };

    struct Device {
        std::string name;
        std::string description;
        int64_t free_bytes              = 0;
        int64_t budget_bytes            = 0;
        enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_GPU;
        // Unified memory (Apple silicon Metal): its weights also come out of the RAM budget, and RAM weights out of it.
        bool shares_host_memory = false;
    };

    enum class ParamsLocation {
        MAIN_GPU,
        CPU,
        OTHER_GPU,
        DISK,
    };

    struct Decision {
        ParamsLocation params_location = ParamsLocation::DISK;
        size_t params_device           = SIZE_MAX;
    };

    struct Runtime {
        std::string name;
        std::vector<size_t> devices;
    };

    struct Plan {
        bool valid         = false;
        size_t main_device = SIZE_MAX;
        std::vector<Runtime> runtimes;
        std::vector<Decision> decisions;
    };

    inline size_t select_main_device(const std::vector<Device>& devices) {
        // Shared system memory must not make an integrated GPU outrank a discrete GPU.
        for (bool discrete : {true, false}) {
            size_t main_device = SIZE_MAX;
            for (size_t di = 0; di < devices.size(); ++di) {
                if ((devices[di].type == GGML_BACKEND_DEVICE_TYPE_GPU) != discrete) {
                    continue;
                }
                if (devices[di].budget_bytes > 0 &&
                    (main_device == SIZE_MAX || devices[di].budget_bytes > devices[main_device].budget_bytes)) {
                    main_device = di;
                }
            }
            if (main_device != SIZE_MAX) {
                return main_device;
            }
        }
        return SIZE_MAX;
    }

    inline Plan compute_plan(const std::vector<Component>& components,
                             const std::vector<Device>& devices,
                             int64_t ram_budget_bytes,
                             const std::vector<Runtime>& runtimes = {}) {
        Plan plan;
        plan.main_device = select_main_device(devices);
        plan.runtimes    = runtimes;
        if (plan.runtimes.empty()) {
            if (plan.main_device == SIZE_MAX) {
                return plan;
            }
            plan.runtimes.resize(components.size(), {devices[plan.main_device].name, {plan.main_device}});
        }

        std::vector<size_t> order(components.size());
        for (size_t ci = 0; ci < components.size(); ++ci) {
            order[ci] = ci;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return components[a].kind < components[b].kind;
        });

        std::vector<int64_t> remaining;
        for (const Device& device : devices) {
            remaining.push_back(std::max<int64_t>(device.budget_bytes, 0));
        }
        ram_budget_bytes = std::max<int64_t>(ram_budget_bytes, 0);
        plan.decisions.resize(components.size());

        // One physical pool: a placement on either side shrinks the other.
        auto charge_shared_ram = [&](const std::vector<size_t>& placed, int64_t bytes) {
            if (std::any_of(placed.begin(), placed.end(), [&](size_t di) { return devices[di].shares_host_memory; })) {
                ram_budget_bytes = std::max<int64_t>(ram_budget_bytes - bytes, 0);
            }
        };
        auto charge_shared_devices = [&](int64_t bytes) {
            for (size_t di = 0; di < devices.size(); ++di) {
                if (devices[di].shares_host_memory) {
                    remaining[di] = std::max<int64_t>(remaining[di] - bytes, 0);
                }
            }
        };

        auto uses_device = [&](size_t ci, size_t di) {
            const auto& runtime_devices = plan.runtimes[ci].devices;
            return std::find(runtime_devices.begin(), runtime_devices.end(), di) != runtime_devices.end();
        };
        auto headroom_for = [&](size_t ci, size_t di) {
            // Higher-priority offloaded weights need cache space on their compute devices.
            int64_t headroom = 0;
            for (size_t other = 0; other < components.size(); ++other) {
                if (components[other].params_bytes == 0 || !uses_device(other, di)) {
                    continue;
                }
                const bool resident          = other == ci || plan.decisions[other].params_location == ParamsLocation::MAIN_GPU;
                const int64_t cached_weights = components[other].kind < components[ci].kind
                                                   ? components[other].params_bytes
                                                   : components[other].staging_bytes;
                headroom                     = std::max(headroom, components[other].reserve_bytes +
                                                                      (resident ? 0 : cached_weights));
            }
            return headroom;
        };

        for (size_t ci : order) {
            const Component& comp = components[ci];
            Decision& decision    = plan.decisions[ci];
            if (comp.params_bytes == 0) {
                continue;
            }

            const auto& runtime_devices = plan.runtimes[ci].devices;
            const bool fits_runtime     = !runtime_devices.empty() &&
                                          std::all_of(runtime_devices.begin(), runtime_devices.end(), [&](size_t di) {
                                          const int64_t headroom = headroom_for(ci, di);
                                          return headroom <= remaining[di] && comp.params_bytes <= remaining[di] - headroom;
                                          });
            if (fits_runtime) {
                decision.params_location = ParamsLocation::MAIN_GPU;
                decision.params_device   = runtime_devices.front();
                // Exact split allocations are unavailable until the runners build their plans.
                for (size_t di : runtime_devices) {
                    remaining[di] -= comp.params_bytes;
                }
                charge_shared_ram(runtime_devices, comp.params_bytes);
                continue;
            }
            if (comp.params_bytes <= ram_budget_bytes) {
                decision.params_location = ParamsLocation::CPU;
                ram_budget_bytes -= comp.params_bytes;
                charge_shared_devices(comp.params_bytes);
                continue;
            }
            if (runtime_devices.empty()) {
                continue;
            }

            size_t best = SIZE_MAX;
            for (size_t di = 0; di < devices.size(); ++di) {
                const int64_t headroom = headroom_for(ci, di);
                if (!uses_device(ci, di) && headroom <= remaining[di] && comp.params_bytes <= remaining[di] - headroom &&
                    (best == SIZE_MAX || remaining[di] > remaining[best])) {
                    best = di;
                }
            }
            if (best != SIZE_MAX) {
                decision.params_location = ParamsLocation::OTHER_GPU;
                decision.params_device   = best;
                remaining[best] -= comp.params_bytes;
                charge_shared_ram({best}, comp.params_bytes);
            }
        }
        plan.valid = true;
        return plan;
    }

}  // namespace sd::backend_fit

#endif  // __SD_BACKEND_FIT_PLAN_H__
