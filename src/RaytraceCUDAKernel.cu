#include "gpu.cuh"
#include <iostream>

__global__ void raytrace_k(float *hmap, float *poses_inds, float *max_pts_inds,
                  		   bool *mask, int N, int W, int H, int P) {

	// Adapted from http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html

	// Get indices
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Ray index (one thread per ray)
	int j = int(floor(i / N)); // Pose index
	if (i > (N * P)) return;

	/***** raytrace through ray (this is sloppy repetitive code but I'm lazy) *****/

	// Pull ending values for specific ray
	int x_max = (int)max_pts_inds[3*i];
	int y_max = (int)max_pts_inds[3*i+1];
	float z_max = max_pts_inds[3*i+2];

	// Pull pose values
	int pose_x = poses_inds[3*j];
	int pose_y = poses_inds[3*j+1];
	float pose_z = poses_inds[3*j+2];
	int z_start = int(floor(pose_z));
	
	// Setup
	int dx = abs(x_max - pose_x);
	int dy = abs(y_max - pose_y);
	float dz = fabs(z_max - pose_z);

	double dt_dx = 1.0 / dx;
	double dt_dy = 1.0 / dy;
	double dt_dz = 1.0 / dz;

	int n = 1;
	int x_inc, y_inc, z_inc;
	double t_next_x, t_next_y, t_next_z;

	// define initial variables based on cases
	// x
	if (dx == 0){
		x_inc = 0;
		t_next_x = 1000.0;
	} else if (x_max > pose_x) {
		x_inc = 1;
		n += x_max - pose_x;
		t_next_x = dt_dx;
	} else {
		x_inc = -1;
		n += pose_x - x_max;
		t_next_x = -dt_dx;
	}

	// y
	if (dy == 0){
		y_inc = 0;
		t_next_y = 1000.0;
	} else if (y_max > pose_y) {
		y_inc = 1;
		n += y_max - pose_y;
		t_next_y = dt_dy;
	} else {
		y_inc = -1;
		n += pose_y - y_max;
		t_next_y = -dt_dy;
	}

	// z
	if (dz == 0){
		z_inc = 0;
		t_next_z = 1000.0;
	} else if (z_max > pose_z) {
		z_inc = 1;
		n += int(floor(z_max)) - pose_z;
		t_next_z = (z_start + 1 - pose_z) * dt_dz;
	} else {
		z_inc = -1;
		n += pose_z - int(floor(z_max));
		t_next_z = (pose_z - z_start) * dt_dz;
	}

	// loop through ray and update mask as necessary
	float z_curr = z_start;
	int x_grid = pose_x;
	int y_grid = pose_y;
	int z_grid = z_start;
	float hmap_z;
	double t = 0;

	for (; n > 0; --n){

		// check if current grid index is valid (return if not)
		if (x_grid >= W || x_grid < 0 || y_grid >= H || y_grid < 0) return;

		// Get current x, y, and z given t
		z_curr = z_start + t * z_inc * dz;
		
		// check if current position is above ground (update mask and return if not)
		hmap_z = hmap[y_grid + x_grid * H];
		if (hmap_z >= z_curr) {
			mask[y_grid + x_grid * H] = false;
			return;
		}

		// take a step along the ray
		if (t_next_x < t_next_y && t_next_x < t_next_z) {

			// x is min
			x_grid += x_inc;
			t = t_next_x;
			t_next_x += dt_dx;

		} else if(t_next_y < t_next_x && t_next_y < t_next_z) {

			// y is min
			y_grid += y_inc;
			t = t_next_y;
			t_next_y += dt_dy;

		} else {

			// z is min
			z_grid += z_inc;
			t = t_next_z;
			t_next_z += dt_dz;

		}
	}
}

void RaytraceCUDAKernel(float *hmap, float *poses_inds, float *max_pts_inds,
						bool *mask, int N, int W, int H, int P, cudaStream_t stream) {
						
	// create shared arrays for heightmap and mask
	float *d_hmap, *d_pose, *d_max_pts;
	bool *d_mask;
  	cudaMalloc(&d_pose, 3 * P * sizeof(float)); 
  	cudaMalloc(&d_max_pts, N * 3 * P * sizeof(float));
	cudaMalloc(&d_hmap, H * W * sizeof(float));
	cudaMalloc(&d_mask, H * W * sizeof(bool));

	cudaMemcpy(d_pose, poses_inds, 3 * P * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_max_pts, max_pts_inds, N * 3 * P * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hmap, hmap, H * W * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, H * W * sizeof(bool), cudaMemcpyHostToDevice);

	int T = N * P;
	raytrace_k<<<GET_BLOCKS(T), CUDA_NUM_THREADS, 0, stream>>>(d_hmap, d_pose, d_max_pts, d_mask, N, W, H, P);

	// Read mask results
	cudaMemcpy(mask, d_mask, H * W * sizeof(bool), cudaMemcpyDeviceToHost);
	
	// error handling
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		std::cout << "CUDA kernel failed with error: " << cudaGetErrorString(err);
	}
}
