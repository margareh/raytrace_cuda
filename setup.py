from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='raytrace_cuda',
    version='0.0',
    packages=find_packages(),
    license='MIT License',
    ext_modules=[
        CUDAExtension(
            name='RaytraceCUDA',
            sources=[
                'src/RaytraceCUDA.cpp',
                'src/RaytraceCUDAKernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
