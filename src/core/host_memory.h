#ifndef __SD_CORE_HOST_MEMORY_H__
#define __SD_CORE_HOST_MEMORY_H__

#include <cstdint>

// Bytes this process can still take from host RAM without swapping, or -1 when unknown.
// On Linux this is the smaller of MemAvailable and the headroom under any cgroup v2 memory.max.
int64_t sd_available_host_memory_bytes();

#endif  // __SD_CORE_HOST_MEMORY_H__
