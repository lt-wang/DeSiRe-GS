#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import torch

os.path.dirname(os.path.abspath(__file__))

# 获取CUDA架构，针对RTX 4090优化
def get_cuda_arch_flags():
    try:
        # 尝试自动检测
        if torch.cuda.is_available():
            arch_list = []
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                arch_list.append(f"{capability[0]}.{capability[1]}")
            if arch_list:
                return [f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}" for arch in set(arch_list)]
    except:
        pass
    
    # 如果自动检测失败，为RTX 4090 (compute capability 8.9) 和其他常见架构
    common_archs = ["7.5", "8.0", "8.6", "8.9"]  # 包含RTX 4090的8.9
    return [f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}" for arch in common_archs]

setup(
    name="diff_gauss",
    packages=["diff_gauss"],
    ext_modules=[
        CUDAExtension(
            name="diff_gauss._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-Xcompiler", "-fno-gnu-unique",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ] + get_cuda_arch_flags()
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    
)
