import glob
import os
import re
import subprocess
import sys

from setuptools import find_packages, setup


def _ensure_submodules():
    base = os.path.dirname(os.path.abspath(__file__))
    required = {
        "cutlass headers": os.path.join(base, "cutlass", "include", "cutlass", "cutlass.h"),
        "bundled torchcsprng": os.path.join(base, "csprng", "setup.py"),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not os.path.exists(path)]
    if missing:
        sys.exit(
            "Error: required source trees are missing:\n  "
            + "\n  ".join(missing)
            + "\nIf you cloned without --recursive, run:\n"
            "    git submodule update --init --recursive"
        )


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
    """Pick a /usr/local/cuda-* whose nvcc release matches torch.version.cuda.

    PyTorch's cpp_extension compares torch.version.cuda against the system nvcc
    and refuses to build on mismatch. On hosts with multiple CUDA toolkits
    side-by-side (common on Ubuntu), the /usr/local/cuda symlink often points
    at the wrong one. We try to repair CUDA_HOME automatically before the
    BuildExtension runs and emit an actionable message if we can't.
    """
    if not torch_cuda:
        return  # CPU-only torch; nothing to align.

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
                f"Warning: no exact CUDA {torch_cuda} toolkit found; using {cand} (release {rel}). "
                "PyTorch may emit a minor-version mismatch warning."
            )
            return

    _print_toolkit_install_guide(torch_cuda)


def _detect_ubuntu_codename():
    try:
        with open("/etc/os-release") as f:
            data = dict(
                line.strip().split("=", 1)
                for line in f
                if "=" in line and not line.startswith("#")
            )
    except OSError:
        return None
    if data.get("ID", "").strip('"').lower() != "ubuntu":
        return None
    ver = data.get("VERSION_ID", "").strip('"').replace(".", "")
    return f"ubuntu{ver}" if ver else None


def _print_toolkit_install_guide(torch_cuda):
    """Print actionable install commands for the missing CUDA toolkit."""
    cu_short = f"cu{torch_cuda.replace('.', '')}"
    major_minor = torch_cuda  # e.g. "12.8"
    apt_pkg = f"cuda-toolkit-{torch_cuda.replace('.', '-')}"  # cuda-toolkit-12-8

    lines = [
        f"Warning: torch was built against CUDA {torch_cuda} but no matching "
        f"/usr/local/cuda-* was found (CUDA_HOME='{os.environ.get('CUDA_HOME')}').",
        "",
        f"Pick one of the following to proceed:",
        "",
    ]

    codename = _detect_ubuntu_codename()
    if codename:
        lines += [
            f"  [A] Install CUDA Toolkit {major_minor} via apt on {codename}:",
            f"        wget https://developer.download.nvidia.com/compute/cuda/repos/"
            f"{codename}/x86_64/cuda-keyring_1.1-1_all.deb",
            f"        sudo dpkg -i cuda-keyring_1.1-1_all.deb",
            f"        sudo apt-get update && sudo apt-get install -y {apt_pkg}",
            f"        export CUDA_HOME=/usr/local/cuda-{major_minor}",
            "",
        ]
    else:
        lines += [
            f"  [A] Install CUDA Toolkit {major_minor} via your OS package manager — see "
            f"https://developer.nvidia.com/cuda-{major_minor}-0-download-archive",
            "",
        ]

    lines += [
        f"  [B] Reinstall torch to match the toolkit already on this host, e.g.:",
        f"        pip install torch --index-url https://download.pytorch.org/whl/{cu_short}",
        "",
        f"  [C] Use the standard install without local CUDA extension compilation:",
        f"        NSSMPC_SKIP_CUTLASS=1 NSSMPC_SKIP_CSPRNG_CUDA=1 pip install -e . --no-build-isolation",
    ]
    print("\n".join(lines), file=sys.stderr)


def _set_torch_cuda_arch_list(torch):
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch and env_arch != "Common":
        print(f"Using existing TORCH_CUDA_ARCH_LIST: {env_arch}")
        return

    fallback = "8.0 8.6"
    try:
        if not torch.cuda.is_available():
            os.environ["TORCH_CUDA_ARCH_LIST"] = fallback
            print(f"Notice: no GPU detected at build time; defaulting TORCH_CUDA_ARCH_LIST={fallback}")
            return
        arch_list = []
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            arch = f"{cap[0]}.{cap[1]}"
            if arch not in arch_list:
                arch_list.append(arch)
        detected = " ".join(arch_list) or fallback
        os.environ["TORCH_CUDA_ARCH_LIST"] = detected
        print(f"Auto-detected TORCH_CUDA_ARCH_LIST={detected}")
    except Exception as e:
        os.environ["TORCH_CUDA_ARCH_LIST"] = fallback
        print(f"Warning: CUDA arch auto-detection failed ({e}); defaulted to {fallback}")


_ensure_submodules()

try:
    import torch
except ImportError:
    sys.exit(
        "Error: torch must be installed before building NssMPClib.\n"
        "Install a torch wheel that matches your CUDA toolkit, e.g.:\n"
        "    pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128\n"
        "Then re-run `pip install -e . --no-build-isolation`."
    )

_auto_set_cuda_home(torch.version.cuda)
_set_torch_cuda_arch_list(torch)

from torch.utils import cpp_extension  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

if os.environ.get("CUDA_HOME"):
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

base_path = os.path.dirname(os.path.abspath(__file__))

cutlass_include_dir = os.path.join(base_path, "cutlass", "include")
cutlass_util_include_dir = os.path.join(base_path, "cutlass", "tools", "util", "include")
kernel_source_file = os.path.join("nssmpc", "infra", "utils", "nss_cutlass_kernels.cu")

SKIP_CUTLASS = os.environ.get("NSSMPC_SKIP_CUTLASS", "").lower() in ("1", "true", "yes")

ext_modules = []
cmdclass = {}

if SKIP_CUTLASS:
    print(
        "Notice: NSSMPC_SKIP_CUTLASS is set; skipping CUTLASS extension. "
        "The runtime will fall back to the pure-PyTorch matmul path "
        "(see nssmpc/infra/utils/cuda.py)."
    )
elif not torch.version.cuda:
    print(
        "Notice: torch is CPU-only; skipping CUTLASS extension build. "
        "Install a CUDA-enabled torch and reinstall to enable the fast matmul path."
    )
elif not os.path.exists(kernel_source_file):
    print(f"Warning: kernel source {kernel_source_file} not found; skipping CUTLASS extension.")
else:
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
    ]
    cxx_flags = ["-O3", "-std=c++17"]
    ext_modules = [
        CUDAExtension(
            name="nssmpc.infra.utils.cutlass_kernels",
            sources=[kernel_source_file],
            include_dirs=[
                cutlass_include_dir,
                cutlass_util_include_dir,
                os.path.dirname(kernel_source_file),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="NssMPClib",
    version="1.0.0b1",
    author="XDU NSS Lab",
    author_email="nss@xidian.edu.cn",
    description="A General-Purpose Secure Multi-Party Computation Library Based on PyTorch",
    url="https://gitcode.com/openHiTLS/NssMPClib",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        f"torchcsprng @ file://{base_path}/csprng",
        "torch>=2.5.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security :: Cryptography",
    ],
)
