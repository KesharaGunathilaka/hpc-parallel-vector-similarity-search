// cuda_search.cu — CUDA GPU k-NN Search (SIFT 1M)
//
// Strategy: one CUDA thread per query vector.
//   - All query threads run in parallel on the GPU.
//   - Each thread scans the entire base dataset independently.
//   - A max-heap of size K is maintained in registers.
//   - Results are transferred back to CPU for recall evaluation.
//
// Parallelism model (vs other implementations):
//   Serial      → 1 CPU thread, all queries sequential
//   OpenMP      → N CPU threads, queries split across threads
//   POSIX       → N CPU threads, queries partitioned by range
//   CUDA (this) → thousands of GPU threads, one per query, all concurrent
//
// Compile:
//   nvcc -O2 -arch=sm_86 -I src/common \
//        -o bin/cuda_search \
//        src/parallel/gpu/cuda/cuda_search.cu
//
// Run:
//   ./bin/cuda_search
//
// RTX 3050 specs (sm_86): 4 GB GDDR6, 20 SMs, 2048 threads/SM

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "io.h"

// compute_recall inlined here to avoid knn.h VLA (Neighbor heap[k])
// which MSVC doesn't support
static double compute_recall(const int *results, const int *ground_truth,
                             int query_n, int k, int gt_dim) {
    int correct = 0;
    for (int q = 0; q < query_n; q++)
        for (int i = 0; i < k; i++) {
            int pred = results[(long)q * k + i];
            for (int j = 0; j < k && j < gt_dim; j++)
                if (pred == ground_truth[(long)q * gt_dim + j]) { correct++; break; }
        }
    return (double)correct / (query_n * k) * 100.0;
}

// ============================================================
//  Configuration
// ============================================================
#define K           10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

#define THREADS_PER_BLOCK  256   // tunable: try 128, 256, 512

// ============================================================
//  CUDA error checking
// ============================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================
//  Device helpers
// ============================================================

/**
 * Squared Euclidean distance between two vectors (device-side).
 * Using __restrict__ lets the compiler assume no aliasing,
 * enabling better memory access optimisations.
 * __ldg() loads via the read-only (texture) cache — good for
 * data that many threads read but no thread writes.
 */
__device__ __forceinline__
float d_l2_sq(const float* __restrict__ a,
              const float* __restrict__ b,
              int dim)
{
    float dist = 0.0f;
    #pragma unroll 8          // unroll for the common SIFT dim=128 case
    for (int i = 0; i < dim; i++) {
        float diff = __ldg(&a[i]) - __ldg(&b[i]);
        dist += diff * diff;
    }
    return dist;
}

/** Sift-down for a max-heap stored in two parallel arrays. */
__device__
void d_heap_sift_down(float *h_dist, int *h_idx, int size, int i)
{
    while (1) {
        int largest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        if (l < size && h_dist[l] > h_dist[largest]) largest = l;
        if (r < size && h_dist[r] > h_dist[largest]) largest = r;
        if (largest == i) break;

        float td = h_dist[i]; h_dist[i] = h_dist[largest]; h_dist[largest] = td;
        int   ti = h_idx[i];  h_idx[i]  = h_idx[largest];  h_idx[largest]  = ti;
        i = largest;
    }
}

/** Build a max-heap in-place (bottom-up). */
__device__
void d_heap_build(float *h_dist, int *h_idx, int size)
{
    for (int i = size / 2 - 1; i >= 0; i--)
        d_heap_sift_down(h_dist, h_idx, size, i);
}

// ============================================================
//  Main kernel: one thread per query
// ============================================================
/**
 * Each thread (global id = q) handles query q:
 *   1. Seeds a max-heap with the first K base vectors.
 *   2. Scans the remaining base_n-K vectors; if a closer one is
 *      found, replaces the heap root and re-heapifies.
 *   3. Insertion-sorts the heap (K=10, so sort cost is negligible).
 *   4. Writes K result indices to global memory.
 *
 * Register usage: 2 * K floats/ints for the heap — fine for K=10.
 */
__global__
void knn_kernel(const float* __restrict__ d_base,
                const float* __restrict__ d_queries,
                int base_n,
                int query_n,
                int dim,
                int k,
                int* __restrict__ d_results)
{
    int q = (int)(blockIdx.x * blockDim.x) + threadIdx.x;
    if (q >= query_n) return;

    const float *query = d_queries + (long)q * dim;

    // Heap lives in registers (k=10 → tiny)
    float h_dist[10];
    int   h_idx[10];

    // ---- Phase 1: seed heap with first k base vectors ----
    for (int i = 0; i < k; i++) {
        h_dist[i] = d_l2_sq(query, d_base + (long)i * dim, dim);
        h_idx[i]  = i;
    }
    d_heap_build(h_dist, h_idx, k);

    // ---- Phase 2: scan remaining base vectors ----
    for (long i = k; i < base_n; i++) {
        float d = d_l2_sq(query, d_base + i * dim, dim);
        if (d < h_dist[0]) {          // closer than current worst
            h_dist[0] = d;
            h_idx[0]  = (int)i;
            d_heap_sift_down(h_dist, h_idx, k, 0);
        }
    }

    // ---- Phase 3: insertion-sort ascending by distance ----
    for (int i = 1; i < k; i++) {
        float kd = h_dist[i];
        int   ki = h_idx[i];
        int j = i - 1;
        while (j >= 0 && h_dist[j] > kd) {
            h_dist[j + 1] = h_dist[j];
            h_idx[j + 1]  = h_idx[j];
            j--;
        }
        h_dist[j + 1] = kd;
        h_idx[j + 1]  = ki;
    }

    // ---- Phase 4: write result indices to global memory ----
    int *out = d_results + (long)q * k;
    for (int i = 0; i < k; i++)
        out[i] = h_idx[i];
}

// ============================================================
//  Host utility: GPU info
// ============================================================
static void print_gpu_info(void)
{
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU            : %s\n", prop.name);
    printf("VRAM           : %.0f MB\n", prop.totalGlobalMem / 1.0e6);
    printf("SMs            : %d\n",  prop.multiProcessorCount);
    printf("Max threads/SM : %d\n",  prop.maxThreadsPerMultiProcessor);
    printf("Compute cap.   : %d.%d\n", prop.major, prop.minor);
}

// ============================================================
//  Host utility: wall-clock time
// ============================================================
#include <windows.h>
static double now_sec(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

// ============================================================
//  Main
// ============================================================
int main(void)
{
    printf("============================================\n");
    printf("     CUDA GPU k-NN Search (SIFT 1M)\n");
    printf("============================================\n\n");

    // ---- Print GPU info ----
    print_gpu_info();
    printf("\n");

    // ---- Load dataset on CPU ----
    int base_dim, base_n;
    int query_dim, query_n;
    int gt_dim, gt_n;

    fprintf(stderr, "Loading base vectors ...\n");
    float *h_base = fvecs_read(BASE_PATH, &base_dim, &base_n);
    printf("Base vectors   : %d  (dim = %d)\n", base_n, base_dim);

    fprintf(stderr, "Loading query vectors ...\n");
    float *h_queries = fvecs_read(QUERY_PATH, &query_dim, &query_n);
    printf("Query vectors  : %d  (dim = %d)\n", query_n, query_dim);

    fprintf(stderr, "Loading ground truth ...\n");
    int *h_gt = ivecs_read(GT_PATH, &gt_dim, &gt_n);
    printf("Ground truth   : %d  entries  (neighbors/entry = %d)\n", gt_n, gt_dim);
    printf("k              : %d\n", K);

    if (base_dim != query_dim) {
        fprintf(stderr, "Error: dimension mismatch (base=%d, query=%d)\n",
                base_dim, query_dim);
        return EXIT_FAILURE;
    }

    // ---- Allocate host result buffer ----
    int *h_results = (int *)malloc((long)query_n * K * sizeof(int));
    if (!h_results) {
        fprintf(stderr, "Error: cannot allocate host results\n");
        return EXIT_FAILURE;
    }

    // ---- Allocate GPU memory ----
    size_t base_bytes    = (long)base_n    * base_dim  * sizeof(float);
    size_t query_bytes   = (long)query_n   * query_dim * sizeof(float);
    size_t results_bytes = (long)query_n   * K         * sizeof(int);

    printf("\nGPU memory required:\n");
    printf("  base    : %.1f MB\n", base_bytes   / 1.0e6);
    printf("  queries : %.2f MB\n", query_bytes  / 1.0e6);
    printf("  results : %.3f MB\n", results_bytes / 1.0e6);
    printf("\n");

    float *d_base, *d_queries;
    int   *d_results;

    CUDA_CHECK(cudaMalloc((void **)&d_base,    base_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_queries,  query_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_results,  results_bytes));

    // ---- Copy data to GPU (timed) ----
    fprintf(stderr, "Copying data to GPU ...\n");
    double t_h2d_start = now_sec();

    CUDA_CHECK(cudaMemcpy(d_base,    h_base,    base_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, query_bytes, cudaMemcpyHostToDevice));

    double t_h2d = now_sec() - t_h2d_start;
    printf("H→D transfer   : %.4f seconds  (%.1f MB/s)\n",
           t_h2d, (base_bytes + query_bytes) / 1.0e6 / t_h2d);

    // ---- Configure kernel launch ----
    int threads = THREADS_PER_BLOCK;
    int blocks  = (query_n + threads - 1) / threads;

    printf("Grid           : %d blocks × %d threads = %d total threads\n\n",
           blocks, threads, blocks * threads);

    // ---- Launch kernel (timed with CUDA events for precision) ----
    fprintf(stderr, "Launching CUDA kernel ...\n");

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    knn_kernel<<<blocks, threads>>>(
        d_base, d_queries, base_n, query_n, base_dim, K, d_results
    );
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));   // blocks until kernel finishes

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    double kernel_sec = kernel_ms / 1000.0;

    printf("Kernel time    : %.4f seconds\n", kernel_sec);

    // ---- Copy results back to CPU ----
    double t_d2h_start = now_sec();
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_bytes, cudaMemcpyDeviceToHost));
    double t_d2h = now_sec() - t_d2h_start;
    printf("D→H transfer   : %.4f seconds\n", t_d2h);

    // Total wall time (H2D + kernel + D2H) — what matters for real workloads
    double total_sec = t_h2d + kernel_sec + t_d2h;

    // ---- Compute recall ----
    double recall = compute_recall(h_results, h_gt, query_n, K, gt_dim);

    // ---- Print results ----
    printf("\n--------------------------------------------\n");
    printf("            RESULTS (CUDA GPU)\n");
    printf("--------------------------------------------\n");
    printf("Kernel search time    : %.4f seconds\n", kernel_sec);
    printf("Total wall time       : %.4f seconds\n", total_sec);
    printf("Avg query time        : %.6f seconds\n", kernel_sec / query_n);
    printf("Throughput            : %.2f queries/sec\n", query_n / kernel_sec);
    printf("Recall@%d             : %.2f%%\n", K, recall);
    printf("--------------------------------------------\n");

    // Machine-readable (matches other implementations' CSV format)
    printf("CSV:cuda,1,%.4f,%.2f,%.2f\n",
           kernel_sec, query_n / kernel_sec, recall);

    // ---- Sample results ----
    printf("\nSample results (first 3 queries):\n");
    for (int q = 0; q < 3 && q < query_n; q++) {
        printf("  Query %d: ", q);
        for (int i = 0; i < K; i++)
            printf("%d ", h_results[(long)q * K + i]);
        printf("\n");
        printf("  Truth %d: ", q);
        for (int i = 0; i < K && i < gt_dim; i++)
            printf("%d ", h_gt[(long)q * gt_dim + i]);
        printf("\n");
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free(h_base);
    free(h_queries);
    free(h_gt);
    free(h_results);

    return EXIT_SUCCESS;
}
