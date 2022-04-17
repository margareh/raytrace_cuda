# Parallel Raytracing over Rays and Poses
Performs parallelized ray tracing using CUDA to compute ray crossings relative to a heightmap for a set of rays and poses at which the rays should be evaluated.

## Inputs
[TODO]

## Outputs
[TODO]

## Sources
Underlying algorithm based on http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html, which relies on the Amanatides and Woo ray tracing algorithm. This algorithm searches for the next grid cell crossing and 
