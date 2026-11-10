#include "core/host_memory.h"

#include <algorithm>
#include <fstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#if defined(__linux__)
static bool read_cgroup_value(const std::string& path, int64_t& value) {
    std::ifstream in(path);
    std::string text;
    if (!(in >> text) || text == "max") {
        return false;
    }
    try {
        value = std::stoll(text);
    } catch (...) {
        return false;
    }
    return true;
}

// Smallest memory.max - memory.current from this process's cgroup up to the root.
static int64_t cgroup_headroom_bytes() {
    std::ifstream cgroup("/proc/self/cgroup");
    std::string line, rel;
    while (std::getline(cgroup, line)) {
        if (line.rfind("0::", 0) == 0) {
            rel = line.substr(3);
            break;
        }
    }
    int64_t headroom = -1;
    while (!rel.empty()) {
        const std::string dir = "/sys/fs/cgroup" + rel;
        int64_t limit = 0, current = 0;
        if (read_cgroup_value(dir + "/memory.max", limit) && read_cgroup_value(dir + "/memory.current", current)) {
            // Inactive page cache is charged to the cgroup but reclaimed before an OOM (the kubelet working set).
            std::ifstream stat(dir + "/memory.stat");
            std::string key;
            int64_t value = 0;
            while (stat >> key >> value) {
                if (key == "inactive_file") {
                    current = std::max<int64_t>(current - value, 0);
                    break;
                }
            }
            const int64_t room = std::max<int64_t>(limit - current, 0);
            headroom           = headroom < 0 ? room : std::min(headroom, room);
        }
        if (rel == "/") {
            break;
        }
        const size_t slash = rel.find_last_of('/');
        rel                = slash == 0 || slash == std::string::npos ? "/" : rel.substr(0, slash);
    }
    return headroom;
}
#endif

int64_t sd_available_host_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return (int64_t)status.ullAvailPhys;
    }
#elif defined(__linux__)
    int64_t available = -1;
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    int64_t kib = 0;
    while (meminfo >> key >> kib >> unit) {
        if (key == "MemAvailable:" && unit == "kB" && kib >= 0) {
            available = kib * 1024;
            break;
        }
    }
    const int64_t cgroup = cgroup_headroom_bytes();
    if (cgroup >= 0) {
        available = available < 0 ? cgroup : std::min(available, cgroup);
    }
    return available;
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
