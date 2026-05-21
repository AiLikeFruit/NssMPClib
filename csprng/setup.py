import glob
import os
import re
import shutil
import subprocess
import sys


def _nvcc_release(cuda_home):
    if not cuda_home:
        return None
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else None


def _auto_set_cuda_home(torch_cuda):
    """Align CUDA_HOME to torch.version.cuda.

    Duplicated from NssMPClib/setup.py because pip installs this package in a
    separate subprocess, so env vars set there don't reach us.
    """
    if not torch_cuda:
        return
    current = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if _nvcc_release(current) == torch_cuda:
        return
    candidates = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    for cand in candidates:
        if _nvcc_release(cand) == torch_cuda:
            os.environ["CUDA_HOME"] = cand
            print(f"Notice: auto-set CUDA_HOME={cand} (matches torch.version.cuda={torch_cuda})")
            return
    major = torch_cuda.split(".")[0]
    for cand in candidates:
        rel = _nvcc_release(cand)
        if rel and rel.split(".")[0] == major:
            os.environ["CUDA_HOME"] = cand
            print(
                f"Warning: no exact CUDA {torch_cuda} toolkit found; using {cand} (release {rel})."
            )
            return
    print(
        f"Warning: torch was built against CUDA {torch_cuda} but no matching "
        f"/usr/local/cuda-* was found (CUDA_HOME='{os.environ.get('CUDA_HOME')}').",
        file=sys.stderr,
    )


import torch  # noqa: E402

_auto_set_cuda_home(torch.version.cuda)

from setuptools import Command, find_packages, setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension  # noqa: E402

version = open("version.txt", "r").read().strip()
sha = "Unknown"
package_name = "torchcsprng"

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]
print(f"Building wheel {package_name}-{version}")


def write_version_file():
    version_path = os.path.join(cwd, "torchcsprng", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        # f.write("from torchcsprng.extension import _check_cuda_version\n")
        # f.write("if _check_cuda_version() > 0:\n")
        # f.write("    cuda = _check_cuda_version()\n")


write_version_file()

with open("README.md", "r") as fh:
    long_description = fh.read()


def append_flags(flags, flags_to_append):
    for flag in flags_to_append:
        if not flag in flags:
            flags.append(flag)
    return flags


def get_extensions():
    build_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"

    module_name = "torchcsprng"

    extensions_dir = os.path.join(cwd, module_name, "csrc")

    openmp = "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    cxx_flags = os.getenv("CXX_FLAGS", "")
    if cxx_flags == "":
        cxx_flags = []
    else:
        cxx_flags = cxx_flags.split(" ")
    if openmp:
        if sys.platform == "linux":
            cxx_flags = append_flags(cxx_flags, ["-fopenmp"])
        elif sys.platform == "win32":
            cxx_flags = append_flags(cxx_flags, ["/openmp"])
        # elif sys.platform == 'darwin':
        #     cxx_flags = append_flags(cxx_flags, ['-Xpreprocessor', '-fopenmp'])

    if build_cuda:
        extension = CUDAExtension
        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]

        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        nvcc_flags = append_flags(nvcc_flags, ["--expt-extended-lambda", "-Xcompiler"])
        extra_compile_args = {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        }
    else:
        extra_compile_args = {
            "cxx": cxx_flags,
        }

    ext_modules = [
        extension(
            module_name + "._C",
            sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class fast_install(Command):
    description = "Custom install command that cleans project and installs wheel"
    user_options = []  # Required variable

    def initialize_options(self):
        pass  # Required method

    def finalize_options(self):
        pass  # Required method

    def run(self):
        os.system("python setup.py clean")
        os.system("python setup.py bdist_wheel")
        os.system(f"pip install {glob.glob('./dist/*.whl')[0]} --force-reinstall --no-deps")


class clean(Command):
    description = "Custom clean command that cleans project based on .gitignore rules"
    user_options = []  # Required variable

    def initialize_options(self):
        pass  # Required method

    def finalize_options(self):
        pass  # Required method

    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()
        start_deleting = False
        for wildcard in filter(None, ignores.split("\n")):
            if wildcard == "# do not change or delete this comment - `python setup.py clean` deletes everything after this line":
                start_deleting = True
            if not start_deleting:
                continue
            for filename in glob.glob(wildcard, recursive=True):
                try:
                    os.remove(filename)
                    print(f"Removed file: {filename}")
                except OSError as e:
                    shutil.rmtree(filename, ignore_errors=True)
                    print(f"Removed directory: {filename}")


setup(
    # Metadata
    name=package_name,
    version=version,
    author="Pavel Belevich",
    author_email="pbelevich@fb.com",
    url="https://github.com/pytorch/csprng",
    description="Cryptographically secure pseudorandom number generators for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3",
    # Package info
    packages=find_packages(exclude=("test",)),
    package_data={"": ["*.pyi"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires="torch>=2.3.0",
    ext_modules=get_extensions(),
    test_suite="test",
    cmdclass={
        "fast_install": fast_install,
        "build_ext": BuildExtension,
        "clean": clean,
    },
)
