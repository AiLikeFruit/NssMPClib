#!/usr/bin/env python3
"""Recommend an installation path for the current machine.

This script is intentionally read-only: it diagnoses Python, PyTorch, CUDA,
nvcc, GPU architecture, and submodule state, then prints the installation
commands that are most likely to work.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


YES_VALUES = {"1", "true", "yes", "on"}

# CUDA versions for which PyTorch publishes a cu* wheel index.
# Keep sorted ascending; recommendations always pick the highest entry that is
# <= the driver-supported CUDA or local nvcc release.
PYTORCH_CUDA_INDEXES = ("11.8", "12.1", "12.4", "12.6", "12.8")

# Minimum CUDA compute capability supported by modern PyTorch wheels (sm_50).
# Cards below this (Kepler sm_3.x and earlier) cannot run cu* wheels.
MIN_PYTORCH_COMPUTE_CAP = (5, 0)


@dataclass(frozen=True)
class NvccInfo:
    path: str
    release: str | None


@dataclass(frozen=True)
class TorchInfo:
    installed: bool
    version: str | None = None
    cuda_version: str | None = None
    cuda_available: bool = False
    devices: tuple[tuple[int, str, str], ...] = ()
    error: str | None = None


def run_cmd(args: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in YES_VALUES


def detect_torch() -> TorchInfo:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return TorchInfo(installed=False, error=str(exc))

    devices: list[tuple[int, str, str]] = []
    cuda_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                devices.append((idx, torch.cuda.get_device_name(idx), f"{major}.{minor}"))
    except Exception as exc:
        return TorchInfo(
            installed=True,
            version=getattr(torch, "__version__", None),
            cuda_version=getattr(torch.version, "cuda", None),
            cuda_available=False,
            devices=tuple(devices),
            error=f"torch CUDA probe failed: {exc}",
        )

    return TorchInfo(
        installed=True,
        version=getattr(torch, "__version__", None),
        cuda_version=getattr(torch.version, "cuda", None),
        cuda_available=cuda_available,
        devices=tuple(devices),
    )


def nvcc_release(nvcc_path: str) -> str | None:
    code, out = run_cmd([nvcc_path, "--version"])
    if code != 0:
        return None
    match = re.search(r"release\s+(\d+\.\d+)", out)
    return match.group(1) if match else None


def unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def detect_nvccs() -> list[NvccInfo]:
    candidates: list[str] = []

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "bin" / "nvcc"))

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(path_nvcc)

    candidates.append("/usr/local/cuda/bin/nvcc")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True))

    infos = []
    for path in unique_paths(candidates):
        if Path(path).is_file():
            infos.append(NvccInfo(path=path, release=nvcc_release(path)))
    return infos


def detect_nvidia_smi() -> tuple[bool, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, None
    code, out = run_cmd([smi, "-L"])
    if code != 0 or not out:
        return True, None
    return True, out


def _parse_version(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    m = re.match(r"^\s*(\d+)\.(\d+)", value)
    return (int(m.group(1)), int(m.group(2))) if m else None


def nvidia_smi_driver_cuda() -> str | None:
    """Max CUDA version the NVIDIA driver supports, parsed from `nvidia-smi` header."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    code, out = run_cmd([smi])
    if code != 0:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def nvidia_smi_compute_caps() -> list[str]:
    """Per-GPU compute capability strings from `nvidia-smi --query-gpu=compute_cap`."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    code, out = run_cmd([smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def pytorch_cu_index_for(cuda_version: str | None) -> str | None:
    """Highest entry in PYTORCH_CUDA_INDEXES that is <= cuda_version."""
    target = _parse_version(cuda_version)
    if not target:
        return None
    best: tuple[int, int] | None = None
    best_str: str | None = None
    for v in PYTORCH_CUDA_INDEXES:
        parsed = _parse_version(v)
        if parsed and parsed <= target and (best is None or parsed > best):
            best = parsed
            best_str = v
    return best_str


def recommended_cuda_torch_index(
    nvccs: list[NvccInfo], driver_cuda: str | None
) -> tuple[str, str] | None:
    """Pick the best (cuda_version, wheel_index_url) for installing PyTorch.

    Prefers a wheel matching a local nvcc release (so the user can also build
    NssMPClib's CUDA extensions with that nvcc), then falls back to the
    driver-supported max CUDA. Returns None when nothing in
    PYTORCH_CUDA_INDEXES fits.
    """
    preferred = preferred_nvcc(nvccs)
    if preferred and preferred.release:
        picked = pytorch_cu_index_for(preferred.release)
        if picked:
            return picked, torch_index(picked)
    picked = pytorch_cu_index_for(driver_cuda)
    if picked:
        return picked, torch_index(picked)
    return None


def all_compute_caps_too_old(caps: Iterable[str]) -> bool:
    """True iff at least one cap was reported and the max is below MIN_PYTORCH_COMPUTE_CAP."""
    parsed = [p for p in (_parse_version(c) for c in caps) if p]
    if not parsed:
        return False
    return max(parsed) < MIN_PYTORCH_COMPUTE_CAP


def ubuntu_codename() -> str | None:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return None
    data: dict[str, str] = {}
    for line in os_release.read_text(errors="ignore").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    if data.get("ID", "").lower() != "ubuntu":
        return None
    version = data.get("VERSION_ID", "").replace(".", "")
    return f"ubuntu{version}" if version else None


def cuda_apt_package(cuda_version: str) -> str:
    return f"cuda-toolkit-{cuda_version.replace('.', '-')}"


def torch_index(cuda_version: str) -> str:
    return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"


def matching_nvcc(torch_cuda: str | None, nvccs: list[NvccInfo]) -> NvccInfo | None:
    if not torch_cuda:
        return None
    for info in nvccs:
        if info.release == torch_cuda:
            return info
    return None


def major_matching_nvcc(torch_cuda: str | None, nvccs: list[NvccInfo]) -> NvccInfo | None:
    if not torch_cuda:
        return None
    major = torch_cuda.split(".", 1)[0]
    for info in nvccs:
        if info.release and info.release.split(".", 1)[0] == major:
            return info
    return None


def preferred_nvcc(nvccs: list[NvccInfo]) -> NvccInfo | None:
    versioned = [info for info in nvccs if info.release]
    if not versioned:
        return nvccs[0] if nvccs else None

    def key(info: NvccInfo) -> tuple[int, int]:
        major, _, minor = (info.release or "0.0").partition(".")
        return int(major or 0), int(minor or 0)

    return max(versioned, key=key)


def submodule_status(root: Path) -> list[str]:
    missing = []
    if not (root / "cutlass" / "include" / "cutlass" / "cutlass.h").exists():
        missing.append("cutlass")
    if not (root / "csprng" / "setup.py").exists():
        missing.append("csprng")
    return missing


def detect_missing_build_deps() -> list[str]:
    """Return missing names from ('setuptools', 'wheel') that --no-build-isolation needs."""
    missing = []
    for name in ("setuptools", "wheel"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def detect_cpp_compiler() -> tuple[bool, str | None]:
    """Best-effort detection of a C/C++ compiler that PyTorch can use for extension builds.

    Returns (has_compiler, advice_message). torchcsprng always builds a native
    extension, so this is required regardless of CUDA path.
    """
    if platform.system() == "Windows":
        if shutil.which("cl"):
            return True, None
        # Look for an actual cl.exe inside common VS / Build Tools layouts —
        # directory existence alone isn't enough (installer leaves empty stubs).
        cl_patterns = [
            r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\BuildTools\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
        ]
        if any(glob.glob(p) for p in cl_patterns):
            return False, (
                "Microsoft Visual C++ appears to be installed but 'cl' is not on PATH. "
                "Open the 'x64 Native Tools Command Prompt for VS' (or run vcvarsall.bat) "
                "before retrying, so that the compiler is reachable."
            )
        return False, (
            "Building torchcsprng's C++ extension requires Microsoft Visual C++ 14.0 or newer. "
            "Install 'Build Tools for Visual Studio' (free) from "
            "https://visualstudio.microsoft.com/visual-cpp-build-tools/ "
            "with the 'Desktop development with C++' workload, then open the "
            "'x64 Native Tools Command Prompt for VS' (so 'cl' is on PATH) before retrying."
        )
    # Unix-like
    for name in ("c++", "g++", "clang++", "cc", "gcc", "clang"):
        if shutil.which(name):
            return True, None
    return False, (
        "Building C/C++ extensions requires a system C++ compiler (gcc/g++ or clang). "
        "On Ubuntu/Debian: sudo apt-get install build-essential."
    )


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_command(command: str, intro: str = "Run this command:") -> None:
    print(intro)
    print()
    print(command)


def recommend(root: Path, torch_info: TorchInfo, nvccs: list[NvccInfo]) -> None:
    missing_submodules = submodule_status(root)
    missing_build_deps = detect_missing_build_deps()
    skip_cutlass = env_enabled("NSSMPC_SKIP_CUTLASS")
    skip_csprng_cuda = env_enabled("NSSMPC_SKIP_CSPRNG_CUDA")

    print_section("Recommendation")

    if missing_submodules:
        print("Submodules are missing, so installation should start here:")
        print_command("git submodule update --init --recursive")
        print("Then rerun: python3 scripts/installation_advice.py")
        return

    if missing_build_deps:
        print(
            "Build dependencies for --no-build-isolation are missing: "
            + ", ".join(missing_build_deps)
            + ". pip needs them to run the editable build and produce a wheel."
        )
        print_command(
            "pip install --upgrade " + " ".join(missing_build_deps),
            "Run this command to install them:",
        )
        print("Then rerun: python3 scripts/installation_advice.py")
        return

    has_compiler, compiler_advice = detect_cpp_compiler()
    if not has_compiler:
        print(
            "No C/C++ compiler was detected, but torchcsprng's native extension must "
            "be compiled from source."
        )
        if compiler_advice:
            print(compiler_advice)
        print("Then rerun: python3 scripts/installation_advice.py")
        return

    if skip_cutlass or skip_csprng_cuda:
        print("Skip flags are already set in the environment.")
        if skip_cutlass:
            print("  NSSMPC_SKIP_CUTLASS is enabled.")
        if skip_csprng_cuda:
            print("  NSSMPC_SKIP_CSPRNG_CUDA is enabled.")
        print("Recommended standard install:")
        print_command("NSSMPC_SKIP_CUTLASS=1 NSSMPC_SKIP_CSPRNG_CUDA=1 pip install -e . --no-build-isolation")
        return

    if not torch_info.installed:
        print("PyTorch is not installed, so CUDA capability cannot be evaluated yet.")
        has_smi, _ = detect_nvidia_smi()
        driver_cuda = nvidia_smi_driver_cuda()
        compute_caps = nvidia_smi_compute_caps()

        if not has_smi and not nvccs:
            print("No CUDA toolchain or NVIDIA driver detected; this looks like a CPU-only host.")
            print_command(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                "Run this command to install CPU PyTorch:",
            )
            print("Then rerun: python3 scripts/installation_advice.py")
            return

        if all_compute_caps_too_old(compute_caps):
            cap_str = ", ".join(compute_caps) if compute_caps else "<unknown>"
            min_cap = f"sm_{MIN_PYTORCH_COMPUTE_CAP[0]}{MIN_PYTORCH_COMPUTE_CAP[1]}"
            print(
                f"NVIDIA driver detected, but GPU compute capability ({cap_str}) is below "
                f"{min_cap}, which recent PyTorch CUDA wheels do not support."
            )
            print_command(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                "Run this command to install CPU PyTorch instead:",
            )
            print("Then rerun: python3 scripts/installation_advice.py")
            return

        picked = recommended_cuda_torch_index(nvccs, driver_cuda)
        if picked:
            cu_version, cu_url = picked
            label = f"cu{cu_version.replace('.', '')}"
            context_parts = []
            preferred = preferred_nvcc(nvccs)
            if preferred and preferred.release:
                context_parts.append(f"local nvcc {preferred.release}")
            if driver_cuda:
                context_parts.append(f"driver max CUDA {driver_cuda}")
            context = "; ".join(context_parts) or "GPU present"
            print(f"Choosing PyTorch {label} ({context}).")
            print_command(
                f"pip install torch torchvision torchaudio --index-url {cu_url}",
                f"Run this command to install a CUDA PyTorch build ({label}):",
            )
            print("Then rerun: python3 scripts/installation_advice.py")
            return

        if driver_cuda:
            print(
                f"NVIDIA driver detected but max supported CUDA ({driver_cuda}) is below the "
                f"lowest PyTorch CUDA wheel (cu{PYTORCH_CUDA_INDEXES[0].replace('.', '')}). "
                "Update the NVIDIA driver, or install CPU PyTorch:"
            )
        else:
            print(
                "NVIDIA driver detected but its CUDA support could not be determined. "
                "Falling back to CPU PyTorch:"
            )
        print_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Run this command to install CPU PyTorch:",
        )
        print("Then rerun: python3 scripts/installation_advice.py")
        return

    if torch_info.cuda_version and torch_info.cuda_available:
        exact = matching_nvcc(torch_info.cuda_version, nvccs)
        if exact:
            print("CUDA PyTorch and a matching nvcc were detected.")
            print("setup.py will auto-detect CUDA_HOME and TORCH_CUDA_ARCH_LIST.")
            print_command(
                "pip install -e . --no-build-isolation",
                "Run this command to install NssMPClib with CUDA extensions:",
            )
            return

        major_match = major_matching_nvcc(torch_info.cuda_version, nvccs)
        if major_match:
            print(
                f"PyTorch was built for CUDA {torch_info.cuda_version}, but the closest nvcc "
                f"is {major_match.release} at {major_match.path}."
            )
        else:
            print(f"PyTorch was built for CUDA {torch_info.cuda_version}, but no matching nvcc was found.")

        print(
            "Install the CUDA Toolkit / nvcc version that matches torch.version.cuda, "
            "then rerun the installation advice script."
        )

        codename = ubuntu_codename()
        if codename:
            print("Ubuntu apt example:")
            print_command(
                "wget https://developer.download.nvidia.com/compute/cuda/repos/"
                f"{codename}/x86_64/cuda-keyring_1.1-1_all.deb\n"
                "sudo dpkg -i cuda-keyring_1.1-1_all.deb\n"
                "sudo apt-get update\n"
                f"sudo apt-get install -y {cuda_apt_package(torch_info.cuda_version)}\n"
                f"export CUDA_HOME=/usr/local/cuda-{torch_info.cuda_version}",
                "Run these commands to install the matching CUDA Toolkit on Ubuntu:",
            )
        else:
            print("CUDA Toolkit download page:")
            print_command(
                f"https://developer.nvidia.com/cuda-{torch_info.cuda_version}-0-download-archive",
                "Open this page and install the matching CUDA Toolkit:",
            )

        print("If you prefer matching PyTorch to an existing toolkit instead, reinstall torch with:")
        print_command(
            f"pip install torch torchvision torchaudio --index-url {torch_index(torch_info.cuda_version)}",
            "Alternative command:",
        )
        print("After fixing the toolkit or PyTorch version, rerun: python3 scripts/installation_advice.py")
        return

    if torch_info.cuda_version and not torch_info.cuda_available:
        print(
            f"PyTorch was built with CUDA {torch_info.cuda_version}, but torch.cuda.is_available() is false."
        )
        print("Check the NVIDIA driver, container GPU passthrough, or CUDA_VISIBLE_DEVICES first.")
        print("For a non-GPU install, use:")
        print_command("NSSMPC_SKIP_CUTLASS=1 NSSMPC_SKIP_CSPRNG_CUDA=1 pip install -e . --no-build-isolation")
        return

    has_nvidia_smi, smi_output = detect_nvidia_smi()
    if has_nvidia_smi:
        print("PyTorch is installed as a CPU-only build, though an NVIDIA driver is present.")
        if smi_output:
            print("Detected GPUs:")
            for line in smi_output.splitlines():
                print(f"  {line}")
        print(
            "setup.py and csprng/setup.py both auto-skip their CUDA extensions when "
            "torch is CPU-only, so the standard install below works as-is for a CPU run."
        )
        print_command(
            "pip install -e . --no-build-isolation",
            "Run this command to install NssMPClib (CPU path):",
        )
        compute_caps = nvidia_smi_compute_caps()
        if not all_compute_caps_too_old(compute_caps):
            driver_cuda = nvidia_smi_driver_cuda()
            picked = recommended_cuda_torch_index(nvccs, driver_cuda)
            if picked:
                cu_version, cu_url = picked
                label = f"cu{cu_version.replace('.', '')}"
                print()
                print(
                    f"If you would rather use the GPU, install CUDA torch ({label}) first, "
                    "then rerun this script:"
                )
                print_command(
                    f"pip install torch torchvision torchaudio --index-url {cu_url}",
                    f"Optional CUDA torch install ({label}):",
                )
        return

    print("CPU-only environment detected; setup.py will auto-skip the CUDA extensions.")
    print_command(
        "pip install -e . --no-build-isolation",
        "Run this command to install NssMPClib (CPU path):",
    )


def main() -> int:
    root = repo_root()
    torch_info = detect_torch()
    nvccs = detect_nvccs()
    has_nvidia_smi, smi_output = detect_nvidia_smi()

    print("NssMPClib installation advice")
    print(f"Repository: {root}")

    print_section("Environment")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME') or '<unset>'}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST') or '<unset>'}")

    print_section("PyTorch")
    if torch_info.installed:
        print(f"torch: {torch_info.version}")
        print(f"torch.version.cuda: {torch_info.cuda_version or '<cpu-only>'}")
        print(f"torch.cuda.is_available(): {torch_info.cuda_available}")
        if torch_info.devices:
            for idx, name, cap in torch_info.devices:
                print(f"GPU {idx}: {name} (sm_{cap.replace('.', '')})")
        if torch_info.error:
            print(f"warning: {torch_info.error}")
    else:
        print("torch: not installed")
        if torch_info.error:
            print(f"import error: {torch_info.error}")

    print_section("CUDA Toolchain")
    if nvccs:
        for info in nvccs:
            print(f"nvcc: {info.path} (release {info.release or 'unknown'})")
    else:
        print("nvcc: not found")
    if has_nvidia_smi:
        print("nvidia-smi: found")
        if smi_output:
            for line in smi_output.splitlines():
                print(f"  {line}")
        driver_cuda = nvidia_smi_driver_cuda()
        if driver_cuda:
            print(f"driver-supported CUDA (max): {driver_cuda}")
        caps = nvidia_smi_compute_caps()
        if caps:
            print("compute capability: " + ", ".join(caps))
    else:
        print("nvidia-smi: not found")

    print_section("Submodules")
    missing = submodule_status(root)
    if missing:
        print("missing: " + ", ".join(missing))
    else:
        print("cutlass: ok")
        print("csprng: ok")

    print_section("Build dependencies")
    for name in ("setuptools", "wheel"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{name}: {version}")
        except ImportError:
            print(f"{name}: missing")
    has_compiler, _ = detect_cpp_compiler()
    print(f"C/C++ compiler: {'found' if has_compiler else 'not found'}")

    recommend(root, torch_info, nvccs)

    print("\nNote: this script only recommends commands; it does not install anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
