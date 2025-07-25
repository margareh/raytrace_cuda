void HorizonCUDAKernel(float *hmap, float *azim, float *elev,
	int W, int H, int A, int WB, int HB, float max_range, float res, 
	float min_elev, float elev_delta, cudaStream_t stream);
