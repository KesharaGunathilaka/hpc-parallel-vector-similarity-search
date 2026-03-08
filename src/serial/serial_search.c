// serial_search.c — Serial (single-threaded) brute-force k-NN search

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "io.h"
#include "distance.h"
#include "knn.h"

//Configuration
#define K           10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

int main(void) {
    int base_dim, base_n;
    int query_dim, query_n;
    int gt_dim, gt_n;

    printf("   Serial Brute-Force k-NN Search (SIFT 1M)\n");

    fprintf(stderr, "Loading base vectors ...\n");
    float *base = fvecs_read(BASE_PATH, &base_dim, &base_n);
    printf("Base vectors   : %d  (dim = %d)\n", base_n, base_dim);

    fprintf(stderr, "Loading query vectors ...\n");
    float *queries = fvecs_read(QUERY_PATH, &query_dim, &query_n);
    printf("Query vectors  : %d  (dim = %d)\n", query_n, query_dim);

    fprintf(stderr, "Loading ground truth ...\n");
    int *ground_truth = ivecs_read(GT_PATH, &gt_dim, &gt_n);
    printf("Ground truth   : %d  entries  (neighbors/entry = %d)\n", gt_n, gt_dim);
    printf("k              : %d\n", K);
    printf("\n");

    if (base_dim != query_dim) {
        fprintf(stderr, "Error: dimension mismatch (base=%d, query=%d)\n",
                base_dim, query_dim);
        return EXIT_FAILURE;
    }

    // Allocate result storage
    int *results = (int *)malloc((long)query_n * K * sizeof(int));
    if (!results) {
        fprintf(stderr, "Error: cannot allocate results array\n");
        return EXIT_FAILURE;
    }

    // Serial search
    fprintf(stderr, "Starting serial search ...\n");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int q = 0; q < query_n; q++) {
        const float *query = queries + (long)q * query_dim;
        knn_search(query, base, base_n, base_dim, K,
                   results + (long)q * K);

        /* Progress every 1000 queries */
        if ((q + 1) % 1000 == 0 || q + 1 == query_n) {
            fprintf(stderr, "\r  Progress: %d / %d queries (%.1f%%)",
                    q + 1, query_n, (q + 1) * 100.0 / query_n);
        }
    }
    fprintf(stderr, "\n");

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    // Compute recall
    double recall = compute_recall(results, ground_truth, query_n, K, gt_dim);

    // Print results
    printf("--------------------------------------------\n");
    printf("            RESULTS (Serial)\n");
    printf("--------------------------------------------\n");
    printf("Total search time     : %.4f seconds\n", elapsed);
    printf("Avg query time        : %.6f seconds\n", elapsed / query_n);
    printf("Throughput            : %.2f queries/sec\n", query_n / elapsed);
    printf("Recall@%d             : %.2f%%\n", K, recall);
    printf("--------------------------------------------\n");

    // Machine-readable line for scripts
    printf("CSV:serial,1,%.4f,%.2f,%.2f\n", elapsed, query_n / elapsed, recall);

    // Sample results (first 3 queries)
    printf("\nSample results (first 3 queries):\n");
    for (int q = 0; q < 3 && q < query_n; q++) {
        printf("  Query %d: ", q);
        for (int i = 0; i < K; i++)
            printf("%d ", results[(long)q * K + i]);
        printf("\n");
        printf("  Truth %d: ", q);
        for (int i = 0; i < K && i < gt_dim; i++)
            printf("%d ", ground_truth[(long)q * gt_dim + i]);
        printf("\n");
    }

    // Cleanup
    free(base);
    free(queries);
    free(ground_truth);
    free(results);

    return EXIT_SUCCESS;
}
