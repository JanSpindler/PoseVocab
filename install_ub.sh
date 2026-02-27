#!/bin/bash

conda install nvidia::cuda-toolkit==12.8.1 -y
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install glfw igl numpy open3d opencv-python PyOpenGL PyOpenGL-accelerate PyYAML scikit-image trimesh
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

cd ./utils/posevocab_custom_ops
python setup.py install
cd ../..
