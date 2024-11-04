#!/bin/bash

python_executable=python$1

# Install torch
$python_executable -m pip install numpy pyyaml scipy ipython mkl mkl-include ninja cython typing pandas typing-extensions dataclasses setuptools && conda clean -ya
$python_executable -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Print version information
$python_executable --version
$python_executable -c "import torch; print('PyTorch:', torch.__version__)"
