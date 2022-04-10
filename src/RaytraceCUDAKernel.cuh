void RaytraceCUDAKernel(float *hmap, float *poses_inds, float *max_pts_inds,
			 bool *mask, int N, int W, int H, int P, cudaStream_t stream);
