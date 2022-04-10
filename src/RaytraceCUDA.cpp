#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "RaytraceCUDAKernel.cuh"

void RaytraceCUDA(at::Tensor hmap, at::Tensor poses_inds,
                  at::Tensor max_pts_inds, at::Tensor mask,
                  int W, int H, int P) {
    // number of rays
    int N = (int) (max_pts_inds.numel() / (3 * P));

    // call to CUDA kernel
    RaytraceCUDAKernel(hmap.data_ptr<float>(), poses_inds.data_ptr<float>(), max_pts_inds.data_ptr<float>(),
                       mask.data_ptr<bool>(), N, W, H, P, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("RaytraceCUDA", &RaytraceCUDA, "Raytrace using CUDA");
}
