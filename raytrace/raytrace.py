import torch
from RaytraceCUDA import RaytraceCUDA

def raytrace(hmap, poses_inds, max_pts_inds):
	"""
	Python wrapper for CUDA raytracing
	Inputs: heightmap, pose in grid, endpoints of rays
	Outputs: bool mask, 1 = free and 0 = hit
	"""

	# Create mask object to store results in
	mask = torch.ones_like(hmap, device=torch.device('cuda')).bool()

	# Flatten inputs so rays are all along one index
	P = poses_inds.shape[0]
	poses_flat = poses_inds.flatten()
	max_pts_flat = max_pts_inds.transpose(2,0).flatten()

	# Flatten heightmap and mask
	H = hmap.shape[0]
	W = hmap.shape[1]
	hmap = hmap.flatten()
	mask = mask.flatten()

	# Call to CUDA kernel wrapper
	RaytraceCUDA(hmap, poses_flat, max_pts_flat, mask, W, H, P)
	return torch.reshape(mask, (H, W))
