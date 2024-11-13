cmake_minimum_required(VERSION 3.26)
project(gradlib_extensions LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

find_package(Torch REQUIRED)

set(VLLM_GPU_LANG "HIP")

# Importing torch recognizes and sets up some HIP/ROCm configuration but does
# not let cmake recognize .hip files. In order to get cmake to understand the
# .hip extension automatically, HIP must be enabled explicitly.
enable_language(HIP)

override_gpu_arches(VLLM_GPU_ARCHES
${VLLM_GPU_LANG}
"${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})
set(GRADLIB_SRC "csrc/hipbsolgemm.cu"
  "csrc/rocsolgemm.cu"
  "csrc/torch_bindings.cpp")

define_gpu_extension_target(
    _gradlib_C
    DESTINATION gradlib
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${GRADLIB_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES "gfx942"
    USE_SABI 3
    WITH_SOABI
)