#ifndef __SD_CORE_HOST_MEMORY_H__
#define __SD_CORE_HOST_MEMORY_H__

#include <cstdint>

// Host RAM bytes usable without swapping (Linux: min of MemAvailable and cgroup v2 memory.max headroom), or -1 if unknown.
int64_t sd_available_host_memory_bytes();

#endif  // __SD_CORE_HOST_MEMORY_H__
