#include "backend_fit.h"
#include "backend_fit_plan.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-backend.h"

namespace sd::backend_fit {

    static bool classify_tensor(const std::string& name, ComponentKind& out) {
        auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };

        if (contains("model.diffusion_model.") || contains("unet.")) {
            out = ComponentKind::DIT;
            return true;
        }
        if (contains("first_stage_model.") ||
            name.rfind("vae.", 0) == 0 ||
            name.rfind("tae.", 0) == 0) {
            out = ComponentKind::VAE;
            return true;
        }
        if (contains("text_encoders") ||
            contains("cond_stage_model") ||
            contains("te.text_model.") ||
            contains("conditioner") ||
            name.rfind("text_encoder.", 0) == 0 ||
            name.rfind("text_embedding_projection.", 0) == 0 ||
            contains(".aggregate_embed.")) {
            out = ComponentKind::CONDITIONER;
            return true;
        }
        return false;
    }

    static std::vector<Component> estimate_components(ModelLoader& loader, ggml_type override_wtype) {
        int64_t bytes[3]          = {0, 0, 0};
        int64_t largest_tensor[3] = {0, 0, 0};
        for (const auto& [name, stored_tensor] : loader.get_tensor_storage_map()) {
            TensorStorage ts = stored_tensor;
            ComponentKind kind;
            if (!classify_tensor(ts.name, kind)) {
                continue;
            }
            if (ts.expected_type != GGML_TYPE_COUNT) {
                ts.type = ts.expected_type;
            } else if (override_wtype != GGML_TYPE_COUNT && loader.tensor_should_be_converted(ts, override_wtype)) {
                ts.type = override_wtype;
            }
            const int64_t tensor_bytes = (int64_t)ts.nbytes() + 64;
            bytes[int(kind)] += tensor_bytes;
            largest_tensor[int(kind)] = std::max(largest_tensor[int(kind)], tensor_bytes);
        }

        return {
            {ComponentKind::DIT, "DiT", bytes[int(ComponentKind::DIT)], 2048 * MiB, largest_tensor[int(ComponentKind::DIT)]},
            {ComponentKind::CONDITIONER, "Conditioner", bytes[int(ComponentKind::CONDITIONER)], 2048 * MiB, largest_tensor[int(ComponentKind::CONDITIONER)]},
            {ComponentKind::VAE, "VAE", bytes[int(ComponentKind::VAE)], 1024 * MiB, largest_tensor[int(ComponentKind::VAE)]},
        };
    }

    static std::string budget_key(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return name;
    }

    // ggml has no unified-memory property; every Metal device on Apple silicon draws from host RAM.
    static bool device_shares_host_memory(ggml_backend_dev_t dev) {
        const char* env = std::getenv("SD_AUTOFIT_UMA");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            return false;
        }
#if defined(__APPLE__) && defined(__aarch64__)
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : "";
        return std::strcmp(reg_name, "MTL") == 0 || std::strcmp(reg_name, "Metal") == 0;
#else
        (void)dev;
        return false;
#endif
    }

    static std::vector<Device> enumerate_gpu_devices(const sd::ggml_graph_cut::MaxVramAssignment& budgets,
                                                     bool include_other_devices) {
        std::vector<Device> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type        = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                (!include_other_devices || type == GGML_BACKEND_DEVICE_TYPE_CPU)) {
                continue;
            }
            Device device;
            device.name        = ggml_backend_dev_name(dev);
            device.description = ggml_backend_dev_description(dev);
            device.type        = type;
            device.shares_host_memory = device_shares_host_memory(dev);
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            device.free_bytes = (int64_t)free_bytes;

            float gib = budgets.default_gib;
            auto it   = budgets.backend_gib.find(budget_key(device.name));
            if (it != budgets.backend_gib.end()) {
                gib = it->second;
            }
            if (gib > 0.f) {
                device.budget_bytes = (int64_t)std::min(gib * 1024.0 * MiB, (double)device.free_bytes);
            } else if (gib < 0.f) {
                device.budget_bytes = (int64_t)std::max<double>(device.free_bytes + gib * 1024.0 * MiB, 0);
            } else {
                device.budget_bytes = std::max<int64_t>(device.free_bytes - 512 * MiB, 0);
            }
            out.push_back(std::move(device));
        }
        return out;
    }

    static int64_t available_ram_bytes() {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            return (int64_t)status.ullAvailPhys;
        }
#elif defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        int64_t kib = 0;
        while (meminfo >> key >> kib >> unit) {
            if (key == "MemAvailable:" && unit == "kB" && kib >= 0) {
                return kib * 1024;
            }
        }
#elif defined(__APPLE__)
        const mach_port_t host = mach_host_self();
        vm_size_t page_size    = 0;
        vm_statistics64_data_t stats{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        const bool ok                = host_page_size(host, &page_size) == KERN_SUCCESS &&
                                       host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count) == KERN_SUCCESS;
        mach_port_deallocate(mach_task_self(), host);
        if (ok) {
            return ((int64_t)stats.free_count + stats.inactive_count) * page_size;
        }
#endif
        return -1;
    }

    static std::string params_backend_name(const Decision& decision, const std::vector<Device>& devices) {
        switch (decision.params_location) {
            case ParamsLocation::MAIN_GPU:
            case ParamsLocation::OTHER_GPU:
                return devices[decision.params_device].name;
            case ParamsLocation::CPU:
                return "cpu";
            case ParamsLocation::DISK:
                return "disk";
        }
        return "disk";
    }

    static void print_plan(const Plan& plan,
                           const std::vector<Component>& components,
                           const std::vector<Device>& devices,
                           int64_t free_ram,
                           int64_t ram_budget) {
        LOG_INFO("auto-fit plan:");
        LOG_INFO("  devices:");
        for (const Device& device : devices) {
            LOG_INFO("    %-12s %-32s free %6lld MiB, budget %6lld MiB%s",
                     device.name.c_str(), device.description.c_str(),
                     (long long)(device.free_bytes / MiB), (long long)(device.budget_bytes / MiB),
                     device.shares_host_memory ? " (shares RAM)" : "");
        }
        if (free_ram < 0) {
            LOG_WARN("auto-fit: available RAM is unknown; skipping CPU parameter residency");
        } else {
            LOG_INFO("    RAM          free %6lld MiB, params budget %6lld MiB",
                     (long long)(free_ram / MiB), (long long)(ram_budget / MiB));
        }
        LOG_INFO("  compute-device weight cache priority: diffusion > te > vae");
        LOG_INFO("  components (params: compute device -> RAM -> other GPU -> disk):");
        for (size_t ci = 0; ci < components.size(); ++ci) {
            const Component& comp = components[ci];
            if (comp.params_bytes == 0) {
                continue;
            }
            const std::string params = plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU
                                           ? plan.runtimes[ci].name
                                           : params_backend_name(plan.decisions[ci], devices);
            LOG_INFO("    %-12s params %6lld MiB, compute reserve %5lld MiB -> compute %s, params %s",
                     comp.name, (long long)(comp.params_bytes / MiB), (long long)(comp.reserve_bytes / MiB),
                     plan.runtimes[ci].name.c_str(), params.c_str());
        }
    }

    static void append_assignment(std::string& spec, const char* key, const std::string& value) {
        if (!spec.empty()) {
            spec += ",";
        }
        spec += key;
        spec += "=";
        spec += value;
    }

    static const char* module_key(ComponentKind kind) {
        switch (kind) {
            case ComponentKind::DIT:
                return "diffusion";
            case ComponentKind::CONDITIONER:
                return "te";
            case ComponentKind::VAE:
                return "vae";
        }
        return "";
    }

    static bool resolve_runtimes(const std::vector<Component>& components,
                                 const std::vector<Device>& devices,
                                 std::string& runtime_spec,
                                 std::vector<Runtime>& runtimes,
                                 std::string& error) {
        SDBackendAssignment assignment;
        if (!sd_parse_backend_assignment(runtime_spec, &assignment, &error)) {
            return false;
        }
        const size_t main_device        = select_main_device(devices);
        const SDBackendModule modules[] = {SDBackendModule::DIFFUSION, SDBackendModule::TE, SDBackendModule::VAE};
        for (const Component& comp : components) {
            std::string name = assignment.get(modules[int(comp.kind)]);
            if (name.empty()) {
                name = main_device == SIZE_MAX ? "cpu" : devices[main_device].name;
                if (comp.params_bytes > 0) {
                    append_assignment(runtime_spec, module_key(comp.kind), name);
                }
            }
            Runtime runtime;
            for (const std::string& part : split_string(name, '&')) {
                if (trim(part).empty()) {
                    continue;
                }
                const std::string resolved = sd_backend_resolve_name(part);
                if (resolved.empty()) {
                    error = "backend '" + part + "' was not found";
                    return false;
                }
                if (!runtime.name.empty()) {
                    runtime.name += "&";
                }
                runtime.name += resolved;
                for (size_t di = 0; di < devices.size(); ++di) {
                    if (devices[di].name == resolved &&
                        std::find(runtime.devices.begin(), runtime.devices.end(), di) == runtime.devices.end()) {
                        runtime.devices.push_back(di);
                    }
                }
            }
            runtimes.push_back(std::move(runtime));
        }
        return true;
    }

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec) {
        std::string error;
        if (!budgets.canonicalize_backend_keys(&error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }

        // Resolve once to ensure dynamic backends are loaded before enumerating devices.
        sd_backend_resolve_name("");
        const auto components = estimate_components(loader, override_wtype);
        const auto devices    = enumerate_gpu_devices(budgets, !runtime_spec.empty());
        std::vector<Runtime> runtimes;
        if (!runtime_spec.empty() && !resolve_runtimes(components, devices, runtime_spec, runtimes, error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }
        const int64_t free_ram   = available_ram_bytes();
        const int64_t ram_budget = std::max<int64_t>(free_ram - std::max<int64_t>(2048 * MiB, free_ram / 10), 0);
        const auto plan          = compute_plan(components, devices, ram_budget, runtimes);
        params_spec.clear();
        if (!plan.valid) {
            if (devices.empty()) {
                LOG_WARN("auto-fit: no GPU devices; using the default backend");
            } else {
                LOG_WARN("auto-fit: no GPU memory budget available; using CPU");
                runtime_spec = "cpu";
            }
            return true;
        }

        print_plan(plan, components, devices, free_ram, ram_budget);
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (components[ci].params_bytes == 0) {
                continue;
            }
            const char* key = module_key(components[ci].kind);
            if (runtimes.empty()) {
                append_assignment(runtime_spec, key, plan.runtimes[ci].name);
            }
            if (plan.decisions[ci].params_location != ParamsLocation::MAIN_GPU) {
                append_assignment(params_spec, key, params_backend_name(plan.decisions[ci], devices));
            }
        }

        // Keep the planner's safety margin when the runner resolves its device limits.
        for (const Device& device : devices) {
            if (device.budget_bytes > 0) {
                budgets.backend_gib[budget_key(device.name)] = (float)(device.budget_bytes / (1024.0 * MiB));
            }
        }
        budgets.resolved_backend_bytes.clear();

        LOG_INFO("auto-fit: --backend \"%s\"%s%s%s",
                 runtime_spec.empty() ? "(default)" : runtime_spec.c_str(),
                 params_spec.empty() ? "" : " --params-backend \"",
                 params_spec.c_str(), params_spec.empty() ? "" : "\"");
        return true;
    }

    bool prepare_vae_decode_retry_tiling(sd_tiling_params_t& tiling_params,
                                         bool prefer_temporal_tiling,
                                         ggml_status status,
                                         int latent_tile_size_w,
                                         int latent_tile_size_h,
                                         int scale_factor) {
        return prepare_vae_retry_tiling(tiling_params, prefer_temporal_tiling, status,
                                        latent_tile_size_w, latent_tile_size_h, scale_factor, false);
    }

    bool prepare_vae_retry_tiling(sd_tiling_params_t& tiling_params,
                                  bool prefer_temporal_tiling,
                                  ggml_status status,
                                  int latent_tile_size_w,
                                  int latent_tile_size_h,
                                  int scale_factor,
                                  bool encode) {
        // Execution failures can leave the device unusable; tiling only helps with allocation failures.
        if (status != GGML_STATUS_ALLOC_FAILED) {
            return false;
        }
        const char* stage      = encode ? "encode" : "decode";
        const char* retry_mode = nullptr;
        if (prefer_temporal_tiling && !tiling_params.temporal_tiling) {
            tiling_params.temporal_tiling = true;
            retry_mode                    = tiling_params.enabled ? "spatial+temporal" : "temporal";
        } else {
            if (latent_tile_size_w <= 0 || latent_tile_size_h <= 0 || scale_factor <= 0) {
                return false;
            }
            auto smaller_tile = [&](int size) {
                return std::min(size, std::max(4, size / 2));
            };
            const int tile_size_w = smaller_tile(latent_tile_size_w);
            const int tile_size_h = smaller_tile(latent_tile_size_h);
            if (tile_size_w == latent_tile_size_w && tile_size_h == latent_tile_size_h) {
                return false;
            }
            tiling_params.enabled     = true;
            tiling_params.rel_size_w  = 0.0f;
            tiling_params.rel_size_h  = 0.0f;
            tiling_params.tile_size_w = tile_size_w * scale_factor;
            tiling_params.tile_size_h = tile_size_h * scale_factor;
            retry_mode                = tiling_params.temporal_tiling ? "spatial+temporal" : "spatial";
            LOG_WARN("Reducing VAE %s tiles from %dx%d to %dx%d image pixels",
                     stage,
                     latent_tile_size_w * scale_factor, latent_tile_size_h * scale_factor,
                     tiling_params.tile_size_w, tiling_params.tile_size_h);
        }

        LOG_WARN("VAE %s ran out of memory; retrying with %s tiling",
                 stage,
                 retry_mode);
        return true;
    }

}  // namespace sd::backend_fit
