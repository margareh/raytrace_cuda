#include <algorithm>

// absolute max is 1024
// best performance is usually between 128 and 512
constexpr int CUDA_NUM_THREADS = 256;

// NVIDIA GeForce RTX 3060 has 3840 cores
// should maybe do fewer threads per block and queue up more blocks?
// could prevent needing to wait for some threads to finish processing
// constexpr int MAXIMUM_NUM_BLOCKS = 65535;
constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GET_BLOCKS(const int N) {
  return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
                           MAXIMUM_NUM_BLOCKS),
                  // Use at least 1 block, since CUDA does not allow empty block
                  1);
}
