void RaytraceCUDAKernel(float *hmap, float *poses_inds, float *max_pts_inds,
			 bool *mask, int N, int W, int H, cudaStream_t stream);
