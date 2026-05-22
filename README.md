# HPC Parallel Vector Similarity Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: Linux/WSL](https://img.shields.io/badge/Platform-Linux%2FWSL-blue.svg)](https://docs.microsoft.com/en-us/windows/wsl/)
[![Language: C/CUDA](https://img.shields.io/badge/Language-C%2FCUDA-brightgreen.svg)](https://www.cprogramming.com/)
[![Dataset: SIFT 1M](https://img.shields.io/badge/Dataset-SIFT%201M-orange.svg)](http://corpus-texmex.irisa.fr/)

> **Comprehensive performance comparison of parallel k-NN search implementations**

This project implements exact brute-force **k-Nearest Neighbor (k-NN)** search on the **SIFT 1M** dataset and compares multiple parallelization strategies:

- 🔹 **Serial C** - Baseline sequential implementation
- 🔹 **OpenMP** - Shared-memory parallelism with `#pragma omp parallel for`
- 🔹 **POSIX Threads** - Manual thread-based query partitioning
- 🔹 **MPI** - Distributed-memory with base dataset partitioning
- 🔹 **Hybrid (MPI + OpenMP)** - Distributed + shared-memory combination
- 🔹 **CUDA** - GPU acceleration with streamed query batches

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Overview](#-overview)
- [Implementations](#-implementations)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Prerequisites](#-prerequisites)
- [Build](#-build)
- [Running Benchmarks](#-running-benchmarks)
- [Results & Analysis](#-results--analysis)
- [Design Notes](#-design-notes)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Quick Start

```bash
# Clone and navigate
git clone https://github.com/KesharaGunathilaka/hpc-parallel-vector-similarity-search.git
cd hpc-parallel-vector-similarity-search

# Download dataset (must be in data/ directory)
# See Dataset section below

# Build all CPU targets
make all

# Run serial baseline
./bin/serial_search | tee results/serial.txt

# Run OpenMP with 4 threads
./bin/openmp_search 4 | tee results/openmp_4t.txt

# For CUDA (requires NVIDIA GPU + CUDA Toolkit)
make cuda
./bin/cuda_search 512 | tee results/cuda_512.txt
```

See [Running Benchmarks](#-running-benchmarks) for detailed instructions.

---

## Overview

| Feature | Detail |
|---|---|
| **Dataset** | SIFT 1M: 1,000,000 base vectors, 10,000 queries, 128 dimensions |
| **Algorithm** | Exact brute-force k-NN using a max-heap of size k |
| **Distance Metric** | Squared Euclidean (L2) |
| **k** | 10 nearest neighbors |
| **Accuracy Metric** | Recall@10 against provided ground truth |
| **Memory** | ~512 MB data + code |
| **Execution Time** | Serial baseline: ~645 seconds on 4-core system |

---

## Implementations

| Implementation | Parallelism | Strategy | File |
|---|---|---|---|
| **Serial** | Single-threaded | Baseline sequential k-NN | `src/serial/serial_search.c` |
| **OpenMP** | Shared-memory | Parallelizes query loop with `#pragma omp parallel for` | `src/parallel/shared/openmp/openmp_search.c` |
| **POSIX Threads** | Shared-memory | Manual thread pool, each thread processes query subset | `src/parallel/shared/posix/pthread_search.c` |
| **MPI** | Distributed-memory | Partitions base dataset across ranks, merges results on rank 0 | `src/parallel/distributed/mpi/mpi_search.c` |
| **Hybrid (MPI + OpenMP)** | Distributed + Shared | MPI base distribution with OpenMP query parallelism per rank | `src/parallel/hybrid/mpi_openmp/mpi_openmp_search.c` |
| **CUDA** | GPU acceleration | Streams query batches through single GPU kernel | `src/parallel/gpu/cuda/cuda_search.cu` |

---

## Project Structure

```
hpc-parallel-vector-similarity-search/
├── src/                                  # Source code
│   ├── common/                          # Shared headers
│   │   ├── distance.h                  # L2 distance computation
│   │   ├── io.h                        # File I/O utilities
│   │   ├── knn.h                       # k-NN heap implementation
│   │   └── progress.h                  # Progress bar
│   ├── serial/
│   │   └── serial_search.c             # Serial baseline
│   └── parallel/
│       ├── shared/                     # Shared-memory implementations
│       │   ├── openmp/
│       │   │   └── openmp_search.c
│       │   └── posix/
│       │       └── pthread_search.c
│       ├── distributed/                # Distributed-memory
│       │   └── mpi/
│       │       └── mpi_search.c
│       ├── hybrid/                     # Hybrid approaches
│       │   └── mpi_openmp/
│       │       └── mpi_openmp_search.c
│       └── gpu/                        # GPU acceleration
│           └── cuda/
│               └── cuda_search.cu
├── data/                                # Dataset (not in repo, download separately)
│   ├── sift_base.fvecs
│   ├── sift_query.fvecs
│   └── sift_groundtruth.ivecs
├── results/                             # Benchmark results
├── bin/                                 # Compiled binaries (generated)
├── Makefile                            # Build configuration
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## Dataset

This project uses the **SIFT 1M** benchmark dataset from the [ANN Benchmarks / TexMex collection](http://corpus-texmex.irisa.fr/).

| File | Format | Size | Contents |
|---|---|---|---|
| `sift_base.fvecs` | Binary `.fvecs` | ~512 MB | 1,000,000 base vectors (128-dim) |
| `sift_query.fvecs` | Binary `.fvecs` | ~5.1 MB | 10,000 query vectors (128-dim) |
| `sift_groundtruth.ivecs` | Binary `.ivecs` | ~4 MB | Ground truth: 10,000 rows × 100 neighbors |

### File Format

- **`.fvecs`**: `[dim (4-byte int)] [dim × float32]` repeated
- **`.ivecs`**: `[dim (4-byte int)] [dim × int32]` repeated

### Download Instructions

```bash
# Create data directory
mkdir -p data

# Download from ANN Benchmarks (requires ~530 MB disk space)
cd data
wget http://corpus-texmex.irisa.fr/getfile.php?file=sift/sift_base.fvecs.gz
wget http://corpus-texmex.irisa.fr/getfile.php?file=sift/sift_query.fvecs.gz
wget http://corpus-texmex.irisa.fr/getfile.php?file=sift/sift_groundtruth.ivecs.gz

# Decompress
gunzip sift_*.gz
cd ..

# Verify
ls -lh data/sift_*
```

**Dataset must be in `data/` directory before running any benchmark.**

---

## Prerequisites

### CPU & MPI Targets

- **OS:** WSL2 or Linux (Ubuntu 20.04+)
- **Compiler:** GCC 9+ / G++ 9+
- **Build Tools:** GNU Make
- **MPI Library:** Open MPI or MPICH

Install on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y build-essential make gcc g++
sudo apt install -y openmpi-bin libopenmpi-dev
```

Verify installation:

```bash
gcc --version
mpicc --version
```

### CUDA Target (Optional)

- **GPU:** NVIDIA GPU with Compute Capability 3.5+
- **CUDA Toolkit:** 11.0 or newer
- **nvcc compiler:** Must be in `PATH`

Install CUDA Toolkit on WSL:

```bash
# Follow official NVIDIA docs for WSL CUDA installation
# https://docs.nvidia.com/cuda/wsl-user-guide/

# Verify
nvcc --version
```

### Hardware Requirements

| Target | Minimum | Recommended |
|---|---|---|
| **Serial** | Any | Intel i5+ / AMD Ryzen 5+ |
| **OpenMP/Threads** | 4 cores | 8+ cores, 16 GB RAM |
| **MPI** | 2 nodes | 4+ nodes, 10 Gbps network |
| **CUDA** | GTX 1050 | RTX 2080+, 6+ GB VRAM |

### System Information

Check your system:

```bash
# CPU cores
nproc
lscpu | grep -E 'CPU\(s\)|Thread|Core|Socket'

# Memory
free -h

# GPU (if installed)
nvidia-smi
```

---

## Build

### Using Makefile (Recommended)

Build individual targets:

```bash
make serial        # Serial baseline
make openmp        # OpenMP shared-memory
make pthread       # POSIX Threads
make mpi           # MPI distributed-memory
make hybrid        # MPI + OpenMP hybrid
make cuda          # CUDA GPU (requires CUDA Toolkit)
```

Build all CPU targets at once:

```bash
make all
```

Clean build artifacts:

```bash
make clean
```

### Manual Compilation

If you prefer direct compilation:

```bash
mkdir -p bin

# Serial
gcc -O3 -Wall -Wextra -I src/common \
  -o bin/serial_search src/serial/serial_search.c -lm

# OpenMP
gcc -O3 -Wall -Wextra -fopenmp -I src/common \
  -o bin/openmp_search src/parallel/shared/openmp/openmp_search.c -lm

# POSIX Threads
gcc -O3 -Wall -Wextra -I src/common \
  -o bin/pthread_search src/parallel/shared/posix/pthread_search.c -lpthread -lm

# MPI
mpicc -O3 -Wall -Wextra -I src/common \
  -o bin/mpi_search src/parallel/distributed/mpi/mpi_search.c -lm

# Hybrid MPI + OpenMP
mpicc -O3 -Wall -Wextra -fopenmp -I src/common \
  -o bin/mpi_openmp_search src/parallel/hybrid/mpi_openmp/mpi_openmp_search.c -lm

# CUDA
nvcc -O3 -std=c++14 -I src/common \
  -o bin/cuda_search src/parallel/gpu/cuda/cuda_search.cu
```

### Build Verification

Check that binaries were created:

```bash
ls -lh bin/
```

---

## Running Benchmarks

### Prerequisites

Ensure dataset files exist in `data/` directory:

```bash
ls -lh data/sift_*.{fvecs,ivecs}
```

Create results directory:

```bash
mkdir -p results
```

### Running Individual Implementations

#### Serial (Baseline)

```bash
./bin/serial_search | tee results/serial.txt
```

#### OpenMP

Run with different thread counts:

```bash
./bin/openmp_search 4 | tee results/openmp_4t.txt
./bin/openmp_search 8 | tee results/openmp_8t.txt
```

#### POSIX Threads

```bash
./bin/pthread_search 4 | tee results/pthread_4t.txt
./bin/pthread_search 8 | tee results/pthread_8t.txt
```

#### MPI

Start with 2 ranks, then scale up:

```bash
mpirun -np 2 ./bin/mpi_search | tee results/mpi_2r.txt
mpirun -np 4 ./bin/mpi_search | tee results/mpi_4r.txt
mpirun -np 8 ./bin/mpi_search | tee results/mpi_8r.txt
```

**Note:** Number of ranks cannot exceed available CPU cores. Check with `nproc`.

#### Hybrid MPI + OpenMP

Pass OpenMP thread count as argument:

```bash
mpirun -np 2 ./bin/mpi_openmp_search 4 | tee results/hybrid_2r_4t.txt
mpirun -np 4 ./bin/mpi_openmp_search 4 | tee results/hybrid_4r_4t.txt
```

#### CUDA

Tune batch size for your GPU:

```bash
./bin/cuda_search 256 | tee results/cuda_256.txt
./bin/cuda_search 512 | tee results/cuda_512.txt
./bin/cuda_search 1024 | tee results/cuda_1024.txt
```

**Tip:** Larger batch sizes maximize GPU utilization but increase memory usage.

### Running All Benchmarks

```bash
mkdir -p results

# Serial baseline
./bin/serial_search | tee results/serial.txt

# CPU shared-memory
./bin/openmp_search 4 | tee results/openmp_4t.txt
./bin/openmp_search 8 | tee results/openmp_8t.txt
./bin/pthread_search 4 | tee results/pthread_4t.txt
./bin/pthread_search 8 | tee results/pthread_8t.txt

# MPI distributed-memory
mpirun -np 2 ./bin/mpi_search | tee results/mpi_2r.txt
mpirun -np 4 ./bin/mpi_search | tee results/mpi_4r.txt

# Hybrid
mpirun -np 2 ./bin/mpi_openmp_search 4 | tee results/hybrid_2r_4t.txt
mpirun -np 4 ./bin/mpi_openmp_search 4 | tee results/hybrid_4r_4t.txt

# GPU
./bin/cuda_search 256 | tee results/cuda_256.txt
./bin/cuda_search 512 | tee results/cuda_512.txt
```

## Results & Analysis

All benchmark results are saved in the `results/` directory as text files.

### Performance Summary (WSL2, GCC 13.3.0, SIFT 1M)

| Implementation | Config | Time (s) | Throughput (QPS) | Recall@10 | Speedup |
|---|---|---:|---:|---:|---:|
| Serial | 1 core | 645.43 | 15.49 | 99.93% | 1.00x |
| OpenMP | 4 threads | 183.87 | 54.39 | 99.93% | 3.51x |
| OpenMP | 8 threads | 190.63 | 52.46 | 99.93% | 3.38x |
| POSIX Threads | 4 threads | 187.97 | 53.20 | 99.93% | 3.43x |
| POSIX Threads | 8 threads | 189.12 | 52.88 | 99.93% | 3.41x |
| MPI | 2 ranks | 347.38 | 28.79 | 99.94% | 1.86x |
| MPI | 4 ranks | 187.15 | 53.43 | 99.94% | 3.45x |
| Hybrid MPI+OpenMP | 2r + 4t | 188.35 | 53.09 | 99.94% | 3.42x |
| Hybrid MPI+OpenMP | 4r + 4t | 189.65 | 52.73 | 99.94% | 3.40x |
| CUDA | Batch 256 | 353.34 | 28.30 | 99.94% | 1.83x |
| **CUDA** | **Batch 512** | **179.57** | **55.69** | **99.94%** | **3.59x** |

### Key Insights

 **CUDA with batch 512** achieves the best performance (3.59x speedup)  
 **OpenMP with 4 threads** is the best CPU-only option (3.51x speedup)  
 **8 threads slower than 4** due to cache contention  
 **MPI with 2 ranks** underperforms (communication overhead)  
 **Hybrid doesn't improve over pure OpenMP** on single machine  

**See [COMPARISON_RESULTS.txt](results/COMPARISON_RESULTS.txt) for detailed analysis.**

---

## Design Notes

### Serial / OpenMP / POSIX Threads

All CPU implementations use exact brute-force k-NN search:

1. Compute squared L2 distance from query to every base vector
2. Maintain a max-heap of size k
3. Keep only the best k candidates
4. Report Recall@10 against ground truth

**OpenMP Strategy:** Uses `#pragma omp parallel for` to parallelize the query loop.  
**Threads Strategy:** Manual thread pool with query partitioning across threads.

### MPI

Distributed-memory parallelization:

- Partitions base dataset across ranks
- Each rank computes local top-k for all queries using its slice
- Rank 0 gathers and merges local results into final k-NN

**Scalability:** Works best with 4+ ranks when computation dominates communication.

### Hybrid MPI + OpenMP

Combines distributed and shared-memory:

- MPI partitions base dataset across nodes
- OpenMP parallelizes query loop within each rank
- Good for clusters with multi-core nodes (not recommended for single machine)

### CUDA

GPU-based exact search:

- Keeps full base dataset on GPU
- Streams query batches through single kernel
- Batch size is critical for performance

**Batch Size Tuning:**
- Small (256): Low GPU occupancy, high overhead
- Medium (512): Optimal for RTX 3060 (6 GB VRAM)
- Large (1024+): Maximum throughput, requires more VRAM

---

## Troubleshooting

### File Not Found

Ensure you're in the project root and dataset exists:

```bash
pwd
ls -lh data/
```

### MPI Issues

Check MPI installation:

```bash
mpicc --version
mpirun --version
which mpirun
```

**Error:** "Not enough slots"
```bash
# Use --oversubscribe flag if you have fewer cores than ranks
mpirun --oversubscribe -np 8 ./bin/mpi_search
```

### CUDA Issues

Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

**Error:** "nvcc: not found"
- Install CUDA Toolkit from NVIDIA
- Add CUDA bin to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### Build Errors

Clean and rebuild:

```bash
make clean
make all
```

Check compiler version:

```bash
gcc --version
g++ --version
```

---

## Contributors

- **Keshara Gunathilaka:** [KesharaGunathilaka](https://github.com/KesharaGunathilaka)
- **Kanishka Gunasinghe:** [kanishkapg](https://github.com/kanishkapg)
- **Yashodha Gunawardhana:** [Yashcat2](https://github.com/Yashcat2)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## References

- **SIFT Dataset:** [TexMex / ANN Benchmarks](http://corpus-texmex.irisa.fr/)
- **OpenMP Specification:** [OpenMP.org](https://www.openmp.org/)
- **Open MPI Documentation:** [Open MPI](https://www.open-mpi.org/)
- **CUDA Programming Guide:** [NVIDIA Developers](https://docs.nvidia.com/cuda/)

---

<div align="center">

### Performance Champion: CUDA with Batch Size 512

**3.59x speedup** | **55.69 QPS** | **99.94% Recall@10**

**Repository:** [GitHub](https://github.com/KesharaGunathilaka/hpc-parallel-vector-similarity-search)  
**Last Updated:** May 2026

</div>
