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


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_command(command: str, intro: str = "Run this command:") -> None:
    print(intro)
    print()
    print(command)


def recommend(root: Path, torch_info: TorchInfo, nvccs: list[NvccInfo]) -> None:
    missing_submodules = submodule_status(root)
    skip_cutlass = env_enabled("NSSMPC_SKIP_CUTLASS")
    skip_csprng_cuda = env_enabled("NSSMPC_SKIP_CSPRNG_CUDA")

    print_section("Recommendation")

    if missing_submodules:
        print("Submodules are missing, so installation should start here:")
        print_command("git submodule update --init --recursive")
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
        print("Install a PyTorch build that matches your GPU/driver first.")
        preferred = preferred_nvcc(nvccs)
        if preferred and preferred.release:
            print(f"The newest detected local nvcc is CUDA {preferred.release} at {preferred.path}.")
            print_command(
                "pip install torch torchvision torchaudio "
                f"--index-url {torch_index(preferred.release)}",
                "Run this command to install a matching PyTorch build:",
            )
        else:
            print_command(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
                "Run this command to install a CUDA PyTorch build:",
            )
        print("Then rerun: python3 scripts/installation_advice.py")
        print("If your machine is CPU-only, use the standard install after installing CPU PyTorch.")
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
        print("An NVIDIA driver appears to be present, but this PyTorch build is CPU-only.")
        if smi_output:
            print("Detected GPUs:")
            for line in smi_output.splitlines():
                print(f"  {line}")
        print("Install a CUDA-enabled PyTorch wheel first.")
        preferred = preferred_nvcc(nvccs)
        if preferred and preferred.release:
            print(f"The newest detected local nvcc is CUDA {preferred.release} at {preferred.path}.")
            print_command(
                "pip install torch torchvision torchaudio "
                f"--index-url {torch_index(preferred.release)}",
                "Run this command to install a matching PyTorch build:",
            )
        else:
            print_command(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
                "Run this command to install a CUDA PyTorch build:",
            )
        print("Then rerun: python3 scripts/installation_advice.py")
        return

    print("No usable NVIDIA CUDA path was detected.")
    print("Recommended standard install:")
    print_command("NSSMPC_SKIP_CUTLASS=1 NSSMPC_SKIP_CSPRNG_CUDA=1 pip install -e . --no-build-isolation")


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
    else:
        print("nvidia-smi: not found")

    print_section("Submodules")
    missing = submodule_status(root)
    if missing:
        print("missing: " + ", ".join(missing))
    else:
        print("cutlass: ok")
        print("csprng: ok")

    recommend(root, torch_info, nvccs)

    print("\nNote: this script only recommends commands; it does not install anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
