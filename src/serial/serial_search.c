#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "io.h"
#include "distance.h"
#include "knn.h"
#include "progress.h"

//Configuration
#define K           10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

int main(void) {
    int base_dim, base_n;
    int query_dim, query_n;
    int gt_dim, gt_n;

    printf("   Serial k-NN Search \n");

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

    const int progress_step = 250;

    // Serial search
    fprintf(stderr, "Starting serial search ...\n");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int q = 0; q < query_n; q++) {
        const float *query = queries + (long)q * query_dim;
        knn_search(query, base, base_n, base_dim, K,
                   results + (long)q * K);

        if ((q + 1) % progress_step == 0 || q + 1 == query_n) {
            progress_bar_print("Serial", q + 1, query_n);
        }
    }
    progress_bar_finish("Serial", query_n);

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

    /* Save results to file */
    FILE *outf = fopen("results/serial.txt", "a");
    if (outf) {
        fprintf(outf, "CSV:serial,1,%.4f,%.2f,%.2f\n", elapsed, query_n / elapsed, recall);
        fprintf(outf, "Sample results (first 3 queries):\n");
        for (int q = 0; q < 3 && q < query_n; q++) {
            fprintf(outf, "Query %d: ", q);
            for (int i = 0; i < K; i++)
                fprintf(outf, "%d ", results[(long)q * K + i]);
            fprintf(outf, "\n");
            fprintf(outf, "Truth %d: ", q);
            for (int i = 0; i < K && i < gt_dim; i++)
                fprintf(outf, "%d ", ground_truth[(long)q * gt_dim + i]);
            fprintf(outf, "\n");
        }
        fclose(outf);
    }

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
