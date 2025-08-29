#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "HorizonCUDAKernel.cuh"

void HorizonCUDA(at::Tensor hmap, at::Tensor azim, at::Tensor elev,
                  int W, int H, int WB, int HB, float max_range, float res, float min_elev, float elev_delta) {
    // call to CUDA kernel
    HorizonCUDAKernel(hmap.data_ptr<float>(), azim.data_ptr<float>(), elev.data_ptr<float>(),
                      W, H, WB, HB, max_range, res, min_elev, elev_delta, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("HorizonCUDA", &HorizonCUDA, "Find horizon using CUDA");
}
