# NssMPClib - A General-Purpose Secure Multi-Party Computation Library Based on PyTorch
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/XidianNSS/NssMPClib)

## Introduction

NssMPClib is a secure multi-party computation (MPC) library designed specifically for machine learning, offering
familiar PyTorch-style APIs that make privacy-preserving machine learning development as straightforward as regular
PyTorch programming.

It implements diverse privacy-preserving computation protocols based on both Arithmetic Secret Sharing and Function
Secret Sharing.
## Key Features

- **PyTorch Integration**: Leverages PyTorch tensor operations for ease of use
- **Torch-like APIs**: Familiar APIs for seamless transition from standard PyTorch to secure computation
- **Multiple Security Models**: Supports both Semi-Honest and Honest-Majority security assumptions
- **Flexible Party Configurations**: 2-party and 3-party computation setups
- **Multiple Secret Sharing Schemes**:
  - Additive Secret Sharing (2-party)
  - Replicated Secret Sharing (3-party)
- **Function Secret Sharing (FSS)** implementations with multiple variants:
- **Privacy-Preserving Neural Network Inference**: Support for secure model evaluation
- **Ring-based Computation**: All operations performed on finite rings for cryptographic security

## System Requirements

- **OS**: Linux (required for proper compilation)
- **Python**: 3.10 or higher (recommended: 3.12)
- **PyTorch**: >=2.3.0 (recommended: 2.7.1)
- **Additional**: C++ compiler (gcc/g++), CUDA toolkit (for GPU support)

## Installation

### Step 0: Match torch's CUDA version to your toolkit

Because NssMPClib compiles a CUDA extension at install time, PyTorch's
`cpp_extension` will refuse to build if `torch.version.cuda` does not match the
`nvcc` reachable from `CUDA_HOME` (or `/usr/local/cuda`). On hosts that ship
multiple CUDA toolkits side by side, `setup.py` tries to auto-pick the right
`/usr/local/cuda-X.Y` for you — but the right toolkit still has to exist on
the machine.

```bash
# What CUDA does your torch want?
python -c "import torch; print(torch.version.cuda)"

# What does your system actually have?
ls -d /usr/local/cuda-* 2>/dev/null
```

Suppose `torch.version.cuda` prints `12.8`. Pick one of:

**A. Install the matching toolkit via conda (no sudo, recommended)**
```bash
conda install -c nvidia/label/cuda-12.8.0 \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev
```

**B. Install the matching toolkit via apt (Ubuntu)**
```bash
# Replace ubuntu2404 with your release (e.g. ubuntu2204):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8
```

**C. Reinstall torch to match a toolkit you already have**
```bash
# Check your local toolkit version first:
# nvcc --version  -> Suppose it says 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**D. Skip the CUDA extension entirely** (runtime falls back to the pure-PyTorch matrix multiplication track under `nssmpc/infra/utils/cuda.py`, which is still GPU-accelerated but lacks Cutlass optimization)
```bash
export NSSMPC_SKIP_CUTLASS=1
```

If you skip Step 0, `setup.py` will still try to auto-detect and align
`CUDA_HOME`, and on failure it prints the same four options before exiting.

### Step 1: Clone (with submodules) and install
```bash
git clone --recursive https://github.com/XidianNSS/NssMPClib.git
cd NssMPClib
# If you forgot --recursive:
#   git submodule update --init --recursive

pip install -e . --no-build-isolation
```

`--no-build-isolation` is required because the build needs the already-installed
torch (otherwise pip would set up a clean env without it). The `setup.py`
auto-detects `TORCH_CUDA_ARCH_LIST` from the visible GPUs and aligns `CUDA_HOME`
with `torch.version.cuda` when possible; both can be overridden via env vars.

To install without the CUDA extension (CPU-only, or when no matching toolkit is
available):
```bash
NSSMPC_SKIP_CUTLASS=1 pip install -e . --no-build-isolation
```

### Step 2: Generate Cryptographic Parameters
Generate essential precomputed parameters for MPC operations:
```bash
python scripts/offline_parameter_generation.py
```

**Note**: Parameters are saved to `~/NssMPClib/data/` (32-bit in `data/32/`, 64-bit in `data/64/`).

## Quick Start: 2-Party Computation Example

**Party 0 - `party_0.py`**:

```python
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
import torch

party = Party2PC(0, SEMI_HONEST)
with PartyRuntime(party):
    party.online()
    x = torch.rand([10, 10])
    share_x = SecretTensor(tensor=x)
    result = share_x.recon().convert_to_real_field()
    print("Server result:", result)
```

**Party 1 - `party_1.py`**:

```python
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor

client = Party2PC(1, SEMI_HONEST)
with PartyRuntime(client):
    client.online()
    share_x = SecretTensor(src_id=0)
    result = share_x.recon().convert_to_real_field()
    print("Client result:", result)
```

**Execution**:
```bash
# Terminal 1: Start server
python party_0.py

# Terminal 2: Start client (in separate terminal)
python party_1.py
```

## Running Built-in Examples

### 1. Arithmetic Secret Sharing (2-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1:
python -m unittest test_ass_p0.py
# Terminal 2:
python -m unittest test_ass_p1.py
```

### 2. Neural Network Inference (2-Party)
```bash
cd tests/application/neural_network/2pc/
# Terminal 1:
python neural_network_P0.py
# Terminal 2:
python neural_network_P1.py
```

### 3. Replicated Secret Sharing (3-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1: python -m unittest test_rss_p0.py
# Terminal 2: python -m unittest test_rss_p1.py  
# Terminal 3: python -m unittest test_rss_p2.py
```

## Configuration

Configure the library in `nssmpc/config/configs.json`:
```json
{
    "BIT_LEN": 32,           // Ring size: 32 or 64 bits
    "DEVICE": "cuda",        // Compute device: "cpu" or "cuda"
    "DTYPE": "float",        // Data type: "float" or "int"
    "SCALE_BIT": 8,          // Fixed-point scaling bits
    "DEBUG_LEVEL": 2         // Debug level: 0-Secure, 1-Testing, 2-Development
}
```

**DEBUG_LEVEL Details**:
- **0 (Secure Mode)**: Highest security. All pre-generated keys are destroyed after use, strictly following the One-Time Pad principle.
- **1 (Testing Mode)**: Performance-optimized.  Inputs with the same dimensions reuse the same set of keys, facilitating performance testing and batch operations.
- **2 (Development Mode)**: Convenient for development. Uses a single globally-shared pre-generated key for all operations. **ONLY for non-sensitive development environments**.

**Usage Scenarios**:
- `DEBUG_LEVEL: 0` - Production environments with real sensitive data
- `DEBUG_LEVEL: 1` - Performance testing environments, evaluating performance across different input sizes
- `DEBUG_LEVEL: 2` - Protocol development environments, quickly verifying functional correctness

## Project Structure

```
NssMPClib/
├── nssmpc/                   # Main library source
│   ├── application/          # Privacy-preserving applications
│   ├── config/              # Configuration files
│   ├── infra/               # Infrastructure components
│   ├── primitives/          # Cryptographic primitives
│   ├── protocols/           # MPC protocols
│   └── runtime/             # Runtime coordination
├── data/                     # Precomputed cryptographic parameters
├── tests/                   # Test suite and examples
├── tutorials/               # Detailed tutorials
└── scripts/                 # Utility scripts
```

## Precomputed Cryptographic Parameters

The library uses pre-generated parameters for efficiency. Key types include:

| Parameter Type | Purpose | Typical Use |
|----------------|---------|-------------|
| **AssMulTriples** | Multiplication in Arithmetic Secret Sharing | 2-party computation |
| **BooleanTriples** | AND operations in Boolean Secret Sharing | Secure comparison |
| **RssMulTriples** | Multiplication in Replicated Secret Sharing | 3-party computation |
| **DICFKey** | Distributed Interval Containment Function | Secure comparison |
| **GeLUKey** | Gaussian Error Linear Unit activation | Neural networks |

and so on...

## Tutorials

Detailed tutorials are available in the `tutorials/` directory:

| Tutorial | Description |
|----------|-------------|
| **Tutorial 0** | Library setup and configuration |
| **Tutorial 1** | 2-party secure computation |
| **Tutorial 2** | 3-party secure computation |
| **Tutorial 3** | Privacy-preserving neural network inference |
| **Tutorial 4** | Advanced internal components |

## Best Practices

1. **Separate Processes**: Each party must run in separate terminals
2. **Use Runtime Context**: Always wrap operations in `with PartyRuntime(party):`
3. **Parameter Management**: Generate parameters before first use
4. **Security Selection**: Use DEBUG_LEVEL=0 for production, DEBUG_LEVEL=2 for development

## Troubleshooting

### Common Issues:

1. **"Parameters not found" Error**:
   ```bash
   python scripts/offline_parameter_generation.py
   ```

2. **Port Already in Use**:
   Change base port in `configs.json` or kill existing processes.

3. **CUDA Errors**:
   Set `DEVICE: "cpu"` in config or check CUDA installation.

4. **`RuntimeError: The detected CUDA version (X.Y) mismatches ...` at install**:
   Your system `nvcc` (`/usr/local/cuda/bin/nvcc`) does not match the CUDA
   version torch was built against. Either install the matching CUDA toolkit
   (then re-run `pip install -e . --no-build-isolation`), set `CUDA_HOME` to a
   directory whose `bin/nvcc` matches, reinstall torch from the matching
   `https://download.pytorch.org/whl/cu*` index, or skip the CUDA extension
   with `NSSMPC_SKIP_CUTLASS=1`.

5. **`fatal error: cutlass/...: No such file or directory` during build**:
   Submodules weren't pulled. Run
   `git submodule update --init --recursive` and reinstall.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use NssMPClib in your research, please cite:
```
@software{nssmpclib,
  title = {NssMPClib: Secure Multi-Party Computation Library},
  author = {Xidian University NSS Lab},
  year = {2024},
  url = {https://github.com/XidianNSS/NssMPClib}
}
```

## License

NssMPClib is released under the MIT License. See the LICENSE file for details.

## Contact

- **Email**: xidiannss@gmail.com
- **GitHub**: https://github.com/XidianNSS/NssMPClib
- **Issues**: https://github.com/XidianNSS/NssMPClib/issues

## Acknowledgements

Maintained by the Network and System Security (NSS) Laboratory at Xidian University.