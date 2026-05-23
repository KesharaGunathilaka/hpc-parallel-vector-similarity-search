/**
 * cuda_search.cu - Exact single-GPU brute-force k-NN search with streaming.
 * Strategy:
 *   - Copy the full base dataset to GPU memory once
 *   - Stream query batches through pinned host buffers and CUDA streams
 *   - One CUDA thread processes one query and keeps a small top-k list in registers
 */


#ifdef _MSC_VER
  #define __attribute__(x)
#endif


#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "io.h"
#include "distance.h"
#include "knn.h"
#include "progress.h"

#define K                      10
#define BASE_PATH              "data/sift_base.fvecs"
#define QUERY_PATH             "data/sift_query.fvecs"
#define GT_PATH                "data/sift_groundtruth.ivecs"
#define DEFAULT_BATCH_QUERIES   256
#define THREADS_PER_BLOCK       128

#define CUDA_CHECK(call) do { \
    cudaError_t cuda_status = (call); \
    if (cuda_status != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ float l2_distance_squared_device(const float *a, const float *b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = __ldg(&a[i]) - __ldg(&b[i]);
        dist += diff * diff;
    }
    return dist;
}

__device__ __forceinline__ void insert_candidate(float distance,
                                                 int index,
                                                 float *best_dist,
                                                 int *best_idx,
                                                 int k) {
    if (distance >= best_dist[k - 1]) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && distance < best_dist[pos - 1]) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        pos--;
    }

    best_dist[pos] = distance;
    best_idx[pos] = index;
}

__global__ void knn_kernel(const float *base,
                           int base_n,
                           int dim,
                           const float *queries,
                           int query_n,
                           int k,
                           int *out_indices) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_n) {
        return;
    }

    const float *query = queries + (long)q * dim;

    float best_dist[K];
    int best_idx[K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        best_dist[i] = CUDART_INF_F;
        best_idx[i] = -1;
    }

    for (int i = 0; i < k && i < base_n; i++) {
        float distance = l2_distance_squared_device(query, base + (long)i * dim, dim);
        insert_candidate(distance, i, best_dist, best_idx, k);
    }

    for (int i = k; i < base_n; i++) {
        float distance = l2_distance_squared_device(query, base + (long)i * dim, dim);
        if (distance < best_dist[k - 1]) {
            insert_candidate(distance, i, best_dist, best_idx, k);
        }
    }

    for (int i = 0; i < k; i++) {
        out_indices[(long)q * k + i] = best_idx[i];
    }
}

static void copy_batch_to_pinned(float *dst, const float *src, int count, int dim) {
    memcpy(dst, src, (size_t)count * dim * sizeof(float));
}

int main(int argc, char *argv[]) {
    int batch_queries = DEFAULT_BATCH_QUERIES;
    if (argc >= 2) {
        batch_queries = atoi(argv[1]);
        if (batch_queries < 1) {
            batch_queries = DEFAULT_BATCH_QUERIES;
        }
    }

    CUDA_CHECK(cudaSetDevice(0));

    int base_dim = 0;
    int base_n = 0;
    float *base = fvecs_read(BASE_PATH, &base_dim, &base_n);

    int query_dim = 0;
    int query_n = 0;
    float *queries = fvecs_read(QUERY_PATH, &query_dim, &query_n);

    int gt_dim = 0;
    int gt_n = 0;
    int *ground_truth = ivecs_read(GT_PATH, &gt_dim, &gt_n);

    if (base_dim != query_dim) {
        fprintf(stderr, "Error: dimension mismatch (base=%d, query=%d)\n", base_dim, query_dim);
        free(base);
        free(queries);
        free(ground_truth);
        return EXIT_FAILURE;
    }

    printf("============================================\n");
    printf("     CUDA Exact k-NN Search (SIFT 1M)\n");
    printf("============================================\n\n");
    printf("Base vectors   : %d  (dim = %d)\n", base_n, base_dim);
    printf("Query vectors  : %d  (dim = %d)\n", query_n, query_dim);
    printf("Ground truth   : %d  entries  (neighbors/entry = %d)\n", gt_n, gt_dim);
    printf("k              : %d\n", K);
    printf("Batch size     : %d\n", batch_queries);
    printf("\n");

    float *device_base = NULL;
    CUDA_CHECK(cudaMalloc((void **)&device_base, (size_t)base_n * base_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_base, base, (size_t)base_n * base_dim * sizeof(float), cudaMemcpyHostToDevice));

    float *host_query_buffers[2] = {NULL, NULL};
    int *host_result_buffers[2] = {NULL, NULL};
    float *device_query_buffers[2] = {NULL, NULL};
    int *device_result_buffers[2] = {NULL, NULL};
    cudaStream_t streams[2];

    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaMallocHost((void **)&host_query_buffers[s], (size_t)batch_queries * base_dim * sizeof(float)));
        CUDA_CHECK(cudaMallocHost((void **)&host_result_buffers[s], (size_t)batch_queries * K * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&device_query_buffers[s], (size_t)batch_queries * base_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&device_result_buffers[s], (size_t)batch_queries * K * sizeof(int)));
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    int *results = (int *)malloc((long)query_n * K * sizeof(int));
    if (!results) {
        fprintf(stderr, "Error: cannot allocate results buffer\n");
        return EXIT_FAILURE;
    }

#ifdef _WIN32
    LARGE_INTEGER freq, ts_start, ts_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ts_start);
#else
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif

    int slot_start[2] = {-1, -1};
    int slot_count[2] = {0, 0};

    for (int batch = 0; batch * batch_queries < query_n; batch++) {
        int slot = batch % 2;

        if (batch >= 2) {
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            memcpy(results + (long)slot_start[slot] * K,
                   host_result_buffers[slot],
                   (size_t)slot_count[slot] * K * sizeof(int));
        }

        int start = batch * batch_queries;
        int count = query_n - start;
        if (count > batch_queries) {
            count = batch_queries;
        }

        copy_batch_to_pinned(host_query_buffers[slot], queries + (long)start * base_dim, count, base_dim);

        CUDA_CHECK(cudaMemcpyAsync(device_query_buffers[slot],
                                   host_query_buffers[slot],
                                   (size_t)count * base_dim * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   streams[slot]));

        int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        knn_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[slot]>>>(device_base,
                                                                     base_n,
                                                                     base_dim,
                                                                     device_query_buffers[slot],
                                                                     count,
                                                                     K,
                                                                     device_result_buffers[slot]);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(host_result_buffers[slot],
                                   device_result_buffers[slot],
                                   (size_t)count * K * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   streams[slot]));

        int completed = start + count;
        progress_bar_print("CUDA", completed, query_n);

        slot_start[slot] = start;
        slot_count[slot] = count;
    }

    for (int s = 0; s < 2; s++) {
        if (slot_start[s] >= 0) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            memcpy(results + (long)slot_start[s] * K,
                   host_result_buffers[s],
                   (size_t)slot_count[s] * K * sizeof(int));
        }
    }

    progress_bar_finish("CUDA", query_n);

#ifdef _WIN32
    QueryPerformanceCounter(&ts_end);
    double elapsed = (double)(ts_end.QuadPart - ts_start.QuadPart) / freq.QuadPart;
#else
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
#endif

    double recall = compute_recall(results, ground_truth, query_n, K, gt_dim);

    printf("--------------------------------------------\n");
    printf("        RESULTS (CUDA, 1 GPU)\n");
    printf("--------------------------------------------\n");
    printf("Total search time     : %.4f seconds\n", elapsed);
    printf("Avg query time        : %.6f seconds\n", elapsed / query_n);
    printf("Throughput            : %.2f queries/sec\n", query_n / elapsed);
    printf("Recall@%d             : %.2f%%\n", K, recall);
    printf("--------------------------------------------\n");
    printf("CSV:cuda,1,%.4f,%.2f,%.2f\n", elapsed, query_n / elapsed, recall);


    FILE *outf = fopen("results/cuda.txt", "a");
    if (outf) {
        fprintf(outf, "CSV:cuda,1,%.4f,%.2f,%.2f\n", elapsed, query_n / elapsed, recall);
        fprintf(outf, "Sample results (first 3 queries):\n");
        for (int q = 0; q < 3 && q < query_n; q++) {
            fprintf(outf, "Query %d: ", q);
            for (int i = 0; i < K; i++) {
                fprintf(outf, "%d ", results[(long)q * K + i]);
            }
            fprintf(outf, "\n");
            fprintf(outf, "Truth %d: ", q);
            for (int i = 0; i < K && i < gt_dim; i++) {
                fprintf(outf, "%d ", ground_truth[(long)q * gt_dim + i]);
            }
            fprintf(outf, "\n");
        }
        fclose(outf);
    }

    printf("\nSample results (first 3 queries):\n");
    for (int q = 0; q < 3 && q < query_n; q++) {
        printf("  Query %d: ", q);
        for (int i = 0; i < K; i++) {
            printf("%d ", results[(long)q * K + i]);
        }
        printf("\n");
        printf("  Truth %d: ", q);
        for (int i = 0; i < K && i < gt_dim; i++) {
            printf("%d ", ground_truth[(long)q * gt_dim + i]);
        }
        printf("\n");
    }

    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(device_query_buffers[s]));
        CUDA_CHECK(cudaFree(device_result_buffers[s]));
        CUDA_CHECK(cudaFreeHost(host_query_buffers[s]));
        CUDA_CHECK(cudaFreeHost(host_result_buffers[s]));
    }

    CUDA_CHECK(cudaFree(device_base));

    free(base);
    free(queries);
    free(ground_truth);
    free(results);

    return EXIT_SUCCESS;
}