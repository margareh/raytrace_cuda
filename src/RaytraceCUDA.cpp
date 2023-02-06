#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "RaytraceCUDAKernel.cuh"

void RaytraceCUDA(at::Tensor hmap, at::Tensor poses_inds,
                  at::Tensor max_pts_inds, at::Tensor scan,
                  int W, int H, int P, float res) {
    // number of rays
    int N = (int) (max_pts_inds.numel() / (3 * P));

    // call to CUDA kernel
    RaytraceCUDAKernel(hmap.data_ptr<float>(), poses_inds.data_ptr<float>(), max_pts_inds.data_ptr<float>(),
                       scan.data_ptr<float>(), N, W, H, P, res, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("RaytraceCUDA", &RaytraceCUDA, "Raytrace using CUDA");
}
