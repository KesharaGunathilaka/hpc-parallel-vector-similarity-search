/*
 * Recall verification.
 *
 * The search programs score Recall@10 by comparing neighbour indices with the
 * ground truth. When two base vectors are exactly the same distance from a
 * query, either one is a correct answer, but index matching counts only one.
 *
 * This tool re-runs the exact search and compares distances as well, to show
 * whether every index mismatch is such a tie.
 */
#include <stdio.h>
#include <stdlib.h>

#include "io.h"
#include "distance.h"
#include "knn.h"

#define K           10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

int main(void) {
    int dim, base_n, query_dim, query_n, gt_dim, gt_n;
    float *base = fvecs_read(BASE_PATH, &dim, &base_n);
    float *queries = fvecs_read(QUERY_PATH, &query_dim, &query_n);
    int *gt = ivecs_read(GT_PATH, &gt_dim, &gt_n);

    if (dim != query_dim || gt_n < query_n || gt_dim < K) {
        fprintf(stderr, "Error: dataset files do not match\n");
        return EXIT_FAILURE;
    }

    int *results = (int *)malloc((long)query_n * K * sizeof(int));
    float *distances = (float *)malloc((long)query_n * K * sizeof(float));
    if (!results || !distances) {
        fprintf(stderr, "Error: cannot allocate result buffers\n");
        return EXIT_FAILURE;
    }

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < query_n; q++) {
        knn_search_with_distances(queries + (long)q * dim, base, base_n, dim, K,
                                  distances + (long)q * K, results + (long)q * K);
    }

    long total = (long)query_n * K;
    double index_recall = compute_recall(results, gt, query_n, K, gt_dim);

    /* Both result lists are sorted by distance, so compare position by position */
    long distance_mismatches = 0;
    for (int q = 0; q < query_n; q++) {
        for (int i = 0; i < K; i++) {
            int truth = gt[(long)q * gt_dim + i];
            float truth_distance = l2_distance_squared(queries + (long)q * dim,
                                                       base + (long)truth * dim, dim);
            if (distances[(long)q * K + i] != truth_distance) {
                distance_mismatches++;
            }
        }
    }

    printf("Queries                         : %d (k = %d)\n", query_n, K);
    printf("Recall@%d by neighbour index    : %.2f%% (%.0f mismatches)\n",
           K, index_recall, total * (100.0 - index_recall) / 100.0);
    printf("Recall@%d by neighbour distance : %.2f%% (%ld mismatches)\n",
           K, 100.0 * (total - distance_mismatches) / total, distance_mismatches);

    free(base);
    free(queries);
    free(gt);
    free(results);
    free(distances);

    return distance_mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
