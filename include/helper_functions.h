#pragma once

#include <cstring>

#include <cuda_runtime.h>

inline bool checkCmdLineFlag(const int argc, const char** argv, const char* flag_name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag_name) == 0) {
            return true;
        }

        if (argv[i][0] == '-' && std::strcmp(argv[i] + 1, flag_name) == 0) {
            return true;
        }

        if (argv[i][0] == '-' && argv[i][1] == '-' &&
            std::strcmp(argv[i] + 2, flag_name) == 0) {
            return true;
        }
    }
    return false;
}

inline int findCudaDevice(int /*argc*/, const char** /*argv*/) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaSetDevice(0);
        return 0;
    }
    return -1;
}
