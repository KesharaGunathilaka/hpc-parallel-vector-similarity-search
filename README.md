# HPC Parallel Vector Similarity Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Linux/WSL](https://img.shields.io/badge/Platform-Linux%2FWSL-blue.svg)](https://learn.microsoft.com/en-us/windows/wsl/)
[![Language: C/CUDA](https://img.shields.io/badge/Language-C%2FCUDA-brightgreen.svg)](https://developer.nvidia.com/cuda-zone)
[![Dataset: SIFT 1M](https://img.shields.io/badge/Dataset-SIFT%201M-orange.svg)](http://corpus-texmex.irisa.fr/)

Exact brute-force **k-nearest-neighbour search** over one million 128-dimensional vectors, parallelised six ways in C and CUDA and benchmarked on one laptop.

On an RTX 3060 Laptop GPU the CUDA version answers 10,000 queries in **27.8 s**, against **645.2 s** for the serial baseline: a **23.2x speedup** with the same exact search.

A group project for **EC7207 High Performance Computing** (Semester 7), BSc (Hons) in Computer Engineering, University of Ruhuna.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Implementations](#implementations)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Running Benchmarks](#running-benchmarks)
- [Checking Correctness](#checking-correctness)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

---

## Overview

| Feature | Detail |
|---|---|
| **Dataset** | SIFT 1M: 1,000,000 base vectors, 10,000 queries, 128 dimensions |
| **Algorithm** | Exact brute-force k-NN (every query is compared with every base vector) |
| **Distance** | Squared Euclidean (L2) |
| **k** | 10 |
| **Accuracy measure** | Recall@10 against the dataset's ground truth |
| **Implementations** | Serial C, OpenMP, POSIX threads, MPI, MPI + OpenMP, MPI + POSIX threads, CUDA |

---

## Results

### Test machine

| Part | Detail |
|---|---|
| **Laptop** | Lenovo 82RD, on AC power |
| **CPU** | AMD Ryzen 7 6800H (8 cores, 16 threads) |
| **RAM** | 16 GB (2 x 8 GB, 4800 MT/s) |
| **GPU** | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB (driver 621.01) |
| **OS** | Windows 11 with WSL2, Ubuntu 24.04 |
| **WSL2 limits** | **4 logical processors**, 6 GB RAM (set in `.wslconfig`) |
| **Toolchain** | GCC 13.3.0, Open MPI 4.1.6, CUDA 12.2 (host compiler GCC 12.4.0) |

All CPU figures are for the 4 logical processors given to WSL2, not the whole CPU.

### Timings

Times cover the search itself, not loading the dataset (or, for CUDA, the one-time copy of the base set to the GPU). Speedup is relative to the serial run.

| Implementation | Configuration | Time (s) | Queries/s | Speedup | Recall@10 |
|---|---|---:|---:|---:|---:|
| Serial | 1 thread | 645.24 | 15.50 | 1.00x | 99.93% |
| OpenMP | 2 threads | 338.00 | 29.59 | 1.91x | 99.93% |
| OpenMP | 4 threads | 189.47 | 52.78 | 3.41x | 99.93% |
| OpenMP | 8 threads | 196.84 | 50.80 | 3.28x | 99.93% |
| POSIX threads | 2 threads | 358.51 | 27.89 | 1.80x | 99.93% |
| POSIX threads | 4 threads | 192.80 | 51.87 | 3.35x | 99.93% |
| POSIX threads | 8 threads | 192.85 | 51.86 | 3.35x | 99.93% |
| MPI | 2 ranks | 342.27 | 29.22 | 1.89x | 99.94% |
| MPI | 4 ranks | 185.64 | 53.87 | 3.48x | 99.94% |
| MPI + OpenMP | 2 ranks x 2 threads | 189.73 | 52.71 | 3.40x | 99.94% |
| MPI + OpenMP | 2 ranks x 4 threads | 188.10 | 53.16 | 3.43x | 99.94% |
| MPI + OpenMP | 4 ranks x 4 threads | 185.89 | 53.80 | 3.47x | 99.94% |
| MPI + POSIX | 2 ranks x 2 threads | 190.11 | 52.60 | 3.39x | 99.94% |
| MPI + POSIX | 2 ranks x 4 threads | 189.58 | 52.75 | 3.40x | 99.94% |
| MPI + POSIX | 4 ranks x 4 threads | 186.56 | 53.60 | 3.46x | 99.94% |
| CUDA | batch 256 | 191.71 | 52.16 | 3.37x | 99.94% |
| CUDA | batch 512 | 96.28 | 103.86 | 6.70x | 99.94% |
| CUDA | batch 1024 | 47.95 | 208.55 | 13.46x | 99.94% |
| CUDA | batch 2048 | 36.09 | 277.08 | 17.88x | 99.94% |
| CUDA | batch 4096 | 43.55 | 229.62 | 14.82x | 99.94% |
| CUDA | batch 8192 | 36.60 | 273.25 | 17.63x | 99.94% |
| **CUDA** | **batch 10000** | **27.80** | **359.74** | **23.21x** | **99.94%** |

Each row is one run. The batch sizes 4096, 8192 and 10000 were repeated and agreed within 0.6%. Raw program output is in [`results/`](results/).

### What the numbers show

- **The GPU batch size decides everything.** Each query is one GPU thread, in blocks of 128. A batch of 256 is only 2 blocks, which leaves most of the GPU idle. Going from batch 256 to batch 10000 cuts the time from 191.7 s to 27.8 s.
- **Batch times do not fall smoothly.** 4096 was slower than 2048 in both of its runs, so measure the batch size instead of assuming bigger is better.
- **CPU versions scale with the processors available.** With 4 logical processors, OpenMP, POSIX threads, MPI and both hybrids all reach about 3.4x to 3.5x with 4 workers. Running more threads than processors adds nothing.
- **MPI needs very little communication here.** The ranks only exchange data once, at the end, so 2 ranks already give 1.89x.

See [Comparison Results.txt](results/Comparison%20Results.txt) for the full analysis. The [original project report](docs/Analysis%20Report.pdf) describes the May 2026 runs; the table above is a later re-measurement of the current code.

---

## Implementations

| Implementation | What is parallelised | Source |
|---|---|---|
| **Serial** | Nothing (baseline) | `src/serial/serial_search.c` |
| **OpenMP** | Query loop, `#pragma omp parallel for` | `src/parallel/shared/openmp/openmp_search.c` |
| **POSIX threads** | Queries split into one contiguous range per thread | `src/parallel/shared/posix/pthread_search.c` |
| **MPI** | Base set split across ranks, results merged on rank 0 | `src/parallel/distributed/mpi/mpi_search.c` |
| **MPI + OpenMP** | MPI splits the base set, OpenMP splits the queries in each rank | `src/parallel/hybrid/mpi_openmp/mpi_openmp_search.c` |
| **MPI + POSIX threads** | Same, with hand-managed threads | `src/parallel/hybrid/mpi_posix/mpi_posix_search.c` |
| **CUDA** | One GPU thread per query, queries streamed in batches | `src/parallel/gpu/cuda/cuda_search.cu` |

---

## How It Works

### CPU search

For each query:

1. Compute the squared L2 distance to every base vector.
2. Keep the best 10 candidates in a max-heap, so the worst of them can be replaced quickly.
3. Sort the 10 results by distance.

OpenMP and POSIX threads share the base set in memory and split the queries between threads.

### MPI and hybrid

- Each rank loads only its own slice of the base set and computes a local top 10 for **every** query.
- Local indices are shifted by the slice offset to become global indices.
- Rank 0 collects distances and indices with `MPI_Gather` and keeps the best 10 of all candidates.
- The hybrid versions initialise MPI with `MPI_THREAD_FUNNELED`, because only the main thread calls MPI.

### CUDA

- The whole base set (about 512 MB) is copied to GPU memory once.
- Queries are sent in batches; one GPU thread scans the full base set for one query and keeps its top 10 in a small sorted array.
- Two CUDA streams with pinned host buffers let one batch be prepared while another is running.
- `__ldg` and `__restrict__` mark the vector data as read-only.

---

## Project Structure

```
hpc-parallel-vector-similarity-search/
├── src/
│   ├── common/                  # Shared headers
│   │   ├── distance.h           # Squared L2 distance
│   │   ├── io.h                 # .fvecs / .ivecs readers
│   │   ├── knn.h                # Top-k heap, merge and recall
│   │   └── progress.h           # Progress bar
│   ├── serial/                  # Serial baseline
│   ├── parallel/
│   │   ├── shared/openmp/
│   │   ├── shared/posix/
│   │   ├── distributed/mpi/
│   │   ├── hybrid/mpi_openmp/
│   │   ├── hybrid/mpi_posix/
│   │   └── gpu/cuda/
│   └── tools/
│       └── verify_recall.c      # Recall and tie check
├── docs/                        # Project report
├── results/                     # Benchmark output
├── data/                        # Dataset (not in the repository)
├── Makefile
└── README.md
```

---

## Dataset

[SIFT 1M](http://corpus-texmex.irisa.fr/) (ANN_SIFT1M) from the TexMex corpus.

| File | Size | Contents |
|---|---|---|
| `sift_base.fvecs` | ~516 MB | 1,000,000 base vectors (128 floats each) |
| `sift_query.fvecs` | ~5 MB | 10,000 query vectors |
| `sift_groundtruth.ivecs` | ~4 MB | 100 true neighbours for each query |

Format: each record is a 4-byte integer dimension followed by that many `float32` (`.fvecs`) or `int32` (`.ivecs`) values.

### Download

From the repository root (the archive is about 160 MB):

```bash
mkdir -p data
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs data/
rm -r sift sift.tar.gz
```

All programs read from `data/` relative to the current directory, so run them from the repository root.

---

## Prerequisites

- Linux or WSL2
- GCC and GNU Make
- Open MPI (for the MPI and hybrid versions)
- About 1 GB of free RAM for the CPU versions
- For CUDA: an NVIDIA GPU with room for the ~512 MB base set, and the CUDA Toolkit

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev
```

For CUDA on WSL2, follow NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/). CUDA releases only support certain host compiler versions. CUDA 12.2 supports GCC up to 12, so on Ubuntu 24.04 (GCC 13) also install:

```bash
sudo apt install -y g++-12
```

The Makefile uses `g++-12` for CUDA automatically when it is installed. If `nvcc` is not at `/usr/local/cuda/bin/nvcc`, run `make cuda NVCC=/path/to/nvcc`.

---

## Build

```bash
make all            # all CPU versions
make cuda           # CUDA version
```

Individual targets: `serial`, `openmp`, `pthread`, `mpi`, `hybrid`, `hybrid-posix`, `cuda`. Binaries are written to `bin/`; `make clean` removes them.

---

## Running Benchmarks

```bash
mkdir -p results

# Serial baseline
./bin/serial_search | tee results/serial.txt

# Shared memory: the argument is the thread count
./bin/openmp_search 4  | tee results/openmp_4t.txt
./bin/pthread_search 4 | tee results/pthread_4t.txt

# MPI
mpirun -np 2 ./bin/mpi_search | tee results/mpi_2r.txt
mpirun --use-hwthread-cpus -np 4 ./bin/mpi_search | tee results/mpi_4r.txt

# Hybrid: the argument is threads per rank
mpirun -np 2 ./bin/mpi_openmp_search 2 | tee results/hybrid_2r_2t.txt
mpirun -np 2 ./bin/mpi_posix_search 2  | tee results/hybrid_posix_2r_2t.txt

# CUDA: the argument is the number of queries per batch
./bin/cuda_search 10000 | tee results/cuda_10000.txt
```

`--use-hwthread-cpus` lets Open MPI count hardware threads as slots; without it, 4 ranks do not fit on a 2-core / 4-thread WSL2 setup. `make run-mpi`, `make run-hybrid` and `make run-cuda` are shortcuts (`RANKS`, `OMP_THREADS` and `BATCH` can be overridden).

The programs also append a summary line to a file in `results/`.

To give WSL2 more or fewer processors, set `processors=` in `%UserProfile%\.wslconfig` and run `wsl --shutdown`. CPU timings will change with it.

---

## Checking Correctness

The search is exact, yet Recall@10 reads 99.93% or 99.94% instead of 100%. The reason is ties: when several base vectors are exactly the same distance from a query, any of them is a correct answer, but matching by index accepts only the one listed in the ground truth.

`make verify` checks this by comparing distances as well as indices:

```
Recall@10 by neighbour index    : 99.93% (70 mismatches)
Recall@10 by neighbour distance : 100.00% (0 mismatches)
```

A one-off run of the same distance check on the full CUDA and 4-rank MPI neighbour lists also found no mismatches (that dump step is not part of the repository). Their index-based figure differs by 0.01 points because they break ties in a different order.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open file` / missing data | Run from the repository root and check that `data/` holds the three files |
| MPI: "not enough slots" | Add `--use-hwthread-cpus`, or `--oversubscribe` for more ranks than threads |
| `nvcc: not found` | Install the CUDA Toolkit, or pass `NVCC=` to `make` |
| CUDA build fails with `_Float32 is undefined` | The host compiler is too new: install `g++-12` |

---

## Team

- **Keshara Gunathilaka** ([KesharaGunathilaka](https://github.com/KesharaGunathilaka))
- **Kanishka Gunasinghe** ([kanishkapg](https://github.com/kanishkapg))
- **Yashodha Gunawardhana** ([Yashcat2](https://github.com/Yashcat2))

---

## License

MIT License. See [LICENSE](LICENSE).

---

## References

- [TexMex corpus (SIFT 1M)](http://corpus-texmex.irisa.fr/)
- [OpenMP](https://www.openmp.org/)
- [Open MPI](https://www.open-mpi.org/)
- [CUDA documentation](https://docs.nvidia.com/cuda/)
