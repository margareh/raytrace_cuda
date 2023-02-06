void RaytraceCUDAKernel(float *hmap, float *poses_inds, float *max_pts_inds,
			 float *scan, int N, int W, int H, int P, float res, cudaStream_t stream);
