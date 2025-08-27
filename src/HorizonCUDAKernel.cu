#include "gpu.cuh"
#include <iostream>
#include <math.h>


__device__ float raytrace(float *hmap, float *start_point, float *end_point, float res, int H, int W) {

	// Adapted from http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html

	/***** raytrace through ray (this is sloppy repetitive code but I'm lazy) *****/

	// Pull ending values for specific ray
	int x_max = (int)end_point[0];
	int y_max = (int)end_point[1];
	float z_max = end_point[2];

	// Pull pose values
	int pose_x = start_point[0];
	int pose_y = start_point[1];
	float pose_z = start_point[2];
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
		t_next_x = dt_dx;
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
		t_next_y = dt_dy;
	}

	// z
	if (dz == 0){
		z_inc = 0;
		t_next_z = 1000.0;
	} else if (z_max > z_start) {
		z_inc = 1;
		n += int(floor(z_max)) - z_start;
		t_next_z = (z_start + 1 - pose_z) * dt_dz;
	} else {
		z_inc = -1;
		n += z_start - int(floor(z_max));
		t_next_z = (pose_z - z_start) * dt_dz;
	}

	// loop through ray and update mask as necessary
	float z_curr = pose_z;
	int x_grid = pose_x;
	int y_grid = pose_y;
	int z_grid = z_start;
	float hmap_z, range;
	float x_out, y_out;
	double t = 0;
	float pose_x_m = res * pose_x;
	float pose_y_m = res * pose_y;

	for (; n > 0; --n){

		// check if current grid index is valid (return if not)
		if (x_grid >= W || x_grid < 0 || y_grid >= H || y_grid < 0) return -1.0;

		// Get current x, y, and z given t
		z_curr = pose_z + t * z_inc * dz;
		
		// check if current position is above ground (update scan and return if not)
		hmap_z = hmap[y_grid * W + x_grid];
		if (hmap_z >= z_curr && abs(t) > 0) {
			x_out = res * x_grid;
			y_out = res * y_grid;
			range = sqrt((x_out - pose_x_m) * (x_out - pose_x_m) + (y_out - pose_y_m) * (y_out - pose_y_m) + (z_curr - pose_z) * (z_curr - pose_z));
			return range;
		}

		// take a step along the ray
		if (t_next_x <= t_next_y && t_next_x <= t_next_z) {

			// x is min
			x_grid += x_inc;
			t = t_next_x;
			t_next_x += dt_dx;

		} else if(t_next_y <= t_next_x && t_next_y <= t_next_z) {

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
	return -1.0;

}


__global__ void horizon_k(float *hmap, float *azim, float *elev,
                  		   int W, int H, int A, float max_range, 
						   float res, float min_elev, float elev_delta) {

	// Get indices
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Ray index (one thread per ray) out of total rays
	int j = int(floor(i / (H * W))); // Azimuth index
	int k = int(floor(i / A)); // Grid point index for output grid (within boundary zone)
	int y_ind = int(floor(i / (A * W))); // Y index
	int x_ind = k - W*y_ind; // X index
	if (i >= (H * W * A)) return;

	// Get grid point indices for heightmap grid cell
	// These are offset by the boundary area
	int b = int(floor(max_range * 1000 / res)); // this will be max range in pixels (grid indices)
	int kb = k + W*b + 2*b*y_ind + 2*b*b + b;
	int xb_ind = x_ind+b; // Grid point index, x axis
	int yb_ind = y_ind+b; // Grid point index, y axis

	// Define start point and azimuthal angle
	float curr_height = hmap[kb]; // k*A+j but need to adjust k to account for boundary points
	float start_point[3] = {xb_ind, yb_ind, curr_height + 0.01};
	float curr_azim = azim[j] * (M_PI / 180); // converted to rad

	// Max range in meters
	float max_range_m = max_range * 1000;

	// Start with min elevation and loop until we find no terrain
	float range = 0;
	float curr_elev = min_elev * (M_PI / 180); // converted to rad
	elev_delta *= (M_PI / 180); // converted to rad
	int iter=0;
	while (max_range_m - range > 0.00001) {

		// Increment elevation
		// we're technically skipping the first but that's fine
		// min_elev used later to mark points without intersections
		curr_elev += elev_delta;
	
		// Define end point based on current grid cell, azimuth, elevation
		float cos_elev = cos(curr_elev);
		float end_point[3] = { cos(curr_azim) * cos_elev / res, sin(curr_azim) * cos_elev / res, sin(curr_elev) };
		for (int c=0; c<3; c++){
			end_point[c] *= max_range_m;
			end_point[c] += start_point[c];
		}

		// Call raytrace
		range = raytrace(hmap, start_point, end_point, res, H+2*b, W+2*b);
		if (range < 0) {
			// error in raytracing (out of bounds or didn't find intersection)
			// elevation for this point will be set to minimum value
			// also setting range to max range to break out of loop
			range = max_range_m;
		}
		iter++;

	}

	// Store the results
	// need to subtract off change in elevation for last one that intersects with the terrain
	elev[k*A + j] = curr_elev - elev_delta; // dimension order: y, x, azim
	// elev[k*A + j] = end_point[0];

}

void HorizonCUDAKernel(float *hmap, float *azim, float *elev, 
					   int W, int H, int A, int WB, int HB, float max_range, float res, 
					   float min_elev, float elev_delta, cudaStream_t stream) {
						
	// create shared arrays for heightmap and mask
	float *d_hmap, *d_azim, *d_elev;
	cudaMalloc(&d_hmap, WB * HB * sizeof(float));
  	cudaMalloc(&d_azim, A * sizeof(float));
	cudaMalloc(&d_elev, H * W * A * sizeof(float));

	cudaMemcpy(d_hmap, hmap, WB * HB * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_azim, azim, A * sizeof(float), cudaMemcpyHostToDevice);

	horizon_k<<<GET_BLOCKS(H * W * A), CUDA_NUM_THREADS, 0, stream>>>(d_hmap, d_azim, d_elev, W, H, A, max_range, res, min_elev, elev_delta);

	// Read mask results
	cudaMemcpy(elev, d_elev, H * W * A * sizeof(float), cudaMemcpyDeviceToHost);
	
	// error handling
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		std::cout << "CUDA kernel failed with error: " << cudaGetErrorString(err) << std::endl;
	}

	// Clear memory
	cudaFree(d_hmap);
	cudaFree(d_azim);
	cudaFree(d_elev);

	d_hmap=NULL;
	d_azim=NULL;
	d_elev=NULL;
}
