import torch
import numpy as np
from RaytraceCUDA import RaytraceCUDA
from HorizonCUDA import HorizonCUDA

def raytrace(hmap, poses_inds, max_pts_inds, max_range, res):
	"""
	Python wrapper for CUDA raytracing
	Inputs: heightmap, pose in grid, endpoints of rays
	Outputs: bool mask, 1 = free and 0 = hit
	"""

	# Create scan object to store results in
	# Initialize to max range, will adjust as ray intersections are found
	scan = torch.ones(max_pts_inds.shape[0:3], device=torch.device('cuda')) * max_range

	# Flatten inputs so rays are all along one index
	P = poses_inds.shape[0]
	poses_flat = poses_inds.flatten()
	max_pts_flat = max_pts_inds.transpose(2,0).flatten()

	# Flatten heightmap and mask
	H = hmap.shape[0]
	W = hmap.shape[1]
	hmap = hmap.flatten()

	Hs = scan.shape[0]
	Ws = scan.shape[1]
	scan = scan.reshape((Hs*Ws*P))

	# Call to CUDA kernel wrapper
	RaytraceCUDA(hmap, poses_flat, max_pts_flat, scan.float(), W, H, P, res)

	# Reshape the scan and return
	scan = scan.cpu().reshape((Hs,Ws,P))

	return scan


def raytrace_horizon(hmap, azim, res=1, max_range=4, min_elev=-89, elev_delta=0.25):
	"""
	Python wrapper for CUDA raytracing at each point in heightmap
	Goal is to return horizon for set angular increments
	Inputs: heightmap, azimuthal angles
	Params: 
		max_range	: maximum search range for determining horizon [km]
		res			: resolution of DEM [m]
		min_elev	: starting elevation for determining horizon [degrees]
		elev_delta	: change in elevation for determining horizon [degrees]
	Outputs: elevation angle for each heightmap point and azimuth
	"""

	# Limit to inner portion of heightmap that we have all boundaries for
	h, w = hmap.shape
	hmap_limit = hmap[max_range:(h-max_range-1), max_range:(w-max_range-1)]

	# Get dimensions of heightmap and azimuths that we're going to use
	H, W = hmap_limit.shape
	A = len(azim)

	# Create scan object to store results in
	elev = torch.empty((H,W,A), dtype=np.float32)

	# Convert inputs to tensors if not already
	if not isinstance(hmap_limit, torch.Tensor):
		hmap_limit = torch.Tensor(hmap_limit)

	if not isinstance(azim, torch.Tensor):
		azim = torch.Tensor(azim)

	# Call to CUDA kernel wrapper for horizon calculation
	HorizonCUDA(hmap_limit, azim, elev, max_range, res, min_elev, elev_delta)
	elev = elev.cpu().reshape((H, W, A))

	return elev
