from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "FL", "version.py")) as f:
    exec(f.read())

setup(
    name="FL",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        #"torch>=1.9",
        "tqdm",
        "xraylib",
        "scikit-image",
        "scipy",
        "matplotlib",
    ],
    ext_modules=[
        CUDAExtension(
            name="FL.cuda_lib.atten_cuda",
            sources=[
                "FL/cuda_lib/atten_cuda.cpp",
                "FL/cuda_lib/atten_cuda_kernel.cu",
            ],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
