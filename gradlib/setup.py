import os
import sys
from shutil import which
from pathlib import Path
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
#from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#from torch.utils.hipify import hipify_python
from typing import Dict, List

ext_modules = []
ROOT_DIR = os.path.dirname(__file__)


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


def is_ninja_available() -> bool:
    return which("ninja") is not None


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        try:
            # os.sched_getaffinity() isn't universally available, so fall
            #  back to os.cpu_count() if we get an error here.
            num_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            num_jobs = os.cpu_count()

        nvcc_threads = None

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = os.getenv('CMAKE_BUILD_TYPE', default_cfg)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DVLLM_TARGET_DEVICE=cuda',
        ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DVLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("gradlib.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for i in range(ext.name.count('.')):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)



#ext_modules.append(
#    CUDAExtension(
#        name='gradlib',
#        sources=['./csrc/rocsolgemm.cu', './csrc/hipbsolgemm.cu', './csrc/torch_bindings.cpp'],
#        include_dirs=include_dirs,
#        # add additional libraries argument for hipblaslt
#        libraries=['hipblaslt', 'rocblas'],
#        extra_compile_args={
#            'cxx': [
#                '-O3',
#                '-DLEGACY_HIPBLAS_DIRECT=ON',
#            ],
#            'nvcc': [
#                '-O3',
#                '-U__CUDA_NO_HALF_OPERATORS__',
#                '-U__CUDA_NO_HALF_CONVERSIONS__',
#                "-ftemplate-depth=1024",
#                '-DLEGACY_HIPBLAS_DIRECT=ON',
#            ] + extra_args
#        }))

ext_modules.append(CMakeExtension(name="gradlib._gradlib_C"))

setup(
    name='gradlib',
    packages=['gradlib'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if len(ext_modules) > 0 else {},
)

# python setup.py build && cp build/lib*/gradlib* ../
