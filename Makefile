CC ?= gcc
MPICC ?= mpicc
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS ?= -O3 -Wall -Wextra
CPPFLAGS ?= -I src/common
OPENMP_FLAGS ?= -fopenmp
CUDA_HOST_COMPILER ?= g++
CUDA_FLAGS ?= -O3 -std=c++14 -D__STRICT_ANSI__ -I src/common -ccbin $(CUDA_HOST_COMPILER) -allow-unsupported-compiler
BIN_DIR := bin

SERIAL_SRC := src/serial/serial_search.c
OPENMP_SRC := src/parallel/shared/openmp/openmp_search.c
PTHREAD_SRC := src/parallel/shared/posix/pthread_search.c
MPI_SRC := src/parallel/distributed/mpi/mpi_search.c
HYBRID_SRC := src/parallel/hybrid/mpi_openmp/mpi_openmp_search.c
CUDA_SRC := src/parallel/gpu/cuda/cuda_search.cu

RANKS ?= 2
OMP_THREADS ?= 4
BATCH ?= 256

.PHONY: all serial openmp pthread mpi hybrid cuda run-mpi run-hybrid run-cuda clean

all: serial openmp pthread mpi hybrid

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

serial: $(BIN_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $(BIN_DIR)/serial_search $(SERIAL_SRC) -lm

openmp: $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(CPPFLAGS) -o $(BIN_DIR)/openmp_search $(OPENMP_SRC) -lm

pthread: $(BIN_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $(BIN_DIR)/pthread_search $(PTHREAD_SRC) -lpthread -lm

mpi: $(BIN_DIR)
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -o $(BIN_DIR)/mpi_search $(MPI_SRC) -lm

hybrid: $(BIN_DIR)
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(CPPFLAGS) -o $(BIN_DIR)/mpi_openmp_search $(HYBRID_SRC) -lm

cuda: $(BIN_DIR)
	$(NVCC) $(CUDA_FLAGS) -o $(BIN_DIR)/cuda_search $(CUDA_SRC)

run-mpi: mpi
	mpirun -np $(RANKS) ./$(BIN_DIR)/mpi_search

run-hybrid: hybrid
	mpirun -np $(RANKS) ./$(BIN_DIR)/mpi_openmp_search $(OMP_THREADS)

run-cuda: cuda
	./$(BIN_DIR)/cuda_search $(BATCH)

clean:
	rm -f $(BIN_DIR)/serial_search $(BIN_DIR)/openmp_search $(BIN_DIR)/pthread_search \
	      $(BIN_DIR)/mpi_search $(BIN_DIR)/mpi_openmp_search $(BIN_DIR)/cuda_search
