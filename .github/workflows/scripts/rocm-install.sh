#!/bin/bash

# Replace '.' with '-' ex: 11.8 -> 11-8
rocm_version=$(echo $1 | tr "." "-")
# Removes '-' and '.' ex: ubuntu-20.04 -> ubuntu2004
OS=$(echo $2 | tr -d ".\-")

# Installs ROCm
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
wget https://repo.radeon.com/amdgpu-install/${rocm_version}/ubuntu/focal/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm

# Test nvcc
PATH=/opt/rocm-6.2.2/bin:${PATH}
rocminfo
clinfo

# Log gcc, g++, c++ versions
gcc --version
g++ --version
c++ --version
