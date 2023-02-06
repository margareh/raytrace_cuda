import torch
from RaytraceCUDA import RaytraceCUDA

def raytrace(hmap, poses_inds, max_pts_inds, max_range, res):
	"""
	Python wrapper for CUDA raytracing
	Inputs: heightmap, pose in grid, endpoints of rays
	Outputs: bool mask, 1 = free and 0 = hit
	"""

	# Create scan object to store results in
	# Initialize to max range, will adjust as ray intersections are found
	scan = torch.ones(max_pts_inds.shape[0:3], device=torch.device('cuda')).float() * max_range

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
	RaytraceCUDA(hmap, poses_flat, max_pts_flat, scan, W, H, P, res)

	# Reshape the scan and return
	scan = scan.cpu().reshape((Hs,Ws,P))

	return scan
