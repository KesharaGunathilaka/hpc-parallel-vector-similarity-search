#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "io.h"
#include "distance.h"
#include "knn.h"
#include "progress.h"

#define K          10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

static void compute_partition(long total, int rank, int size, long *start, long *count) {
    long begin = (total * rank) / size;
    long end = (total * (rank + 1)) / size;
    *start = begin;
    *count = end - begin;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int omp_threads = omp_get_max_threads();
    if (argc >= 2) {
        omp_threads = atoi(argv[1]);
        if (omp_threads < 1) {
            omp_threads = 1;
        }
    }
    omp_set_num_threads(omp_threads);

    int base_dim = 0;
    long base_total = 0;
    fvecs_metadata(BASE_PATH, &base_dim, &base_total);

    long base_start = 0;
    long base_count = 0;
    compute_partition(base_total, rank, size, &base_start, &base_count);

    int local_base_dim = 0;
    int local_base_n = 0;
    float *base = fvecs_read_range(BASE_PATH, base_start, base_count, &local_base_dim, &local_base_n);

    int query_dim = 0;
    int query_n = 0;
    float *queries = fvecs_read(QUERY_PATH, &query_dim, &query_n);

    int gt_dim = 0;
    int gt_n = 0;
    int *ground_truth = NULL;
    if (rank == 0) {
        ground_truth = ivecs_read(GT_PATH, &gt_dim, &gt_n);
    }

    if (query_dim != base_dim || local_base_dim != base_dim) {
        if (rank == 0) {
            fprintf(stderr, "Error: dimension mismatch (base=%d, query=%d, local_base=%d)\n",
                    base_dim, query_dim, local_base_dim);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("============================================\n");
        printf("   Hybrid MPI + OpenMP k-NN Search (SIFT 1M)\n");
        printf("============================================\n\n");
        printf("Base vectors   : %ld total\n", base_total);
        printf("Query vectors  : %d\n", query_n);
        printf("Ground truth   : %d entries (neighbors/entry = %d)\n", gt_n, gt_dim);
        printf("k              : %d\n", K);
        printf("MPI ranks      : %d\n", size);
        printf("OMP threads    : %d\n", omp_threads);
        printf("\n");
    }

    int *local_indices = (int *)malloc((long)query_n * K * sizeof(int));
    float *local_distances = (float *)malloc((long)query_n * K * sizeof(float));
    if (!local_indices || !local_distances) {
        fprintf(stderr, "Rank %d: cannot allocate local result buffers\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_done = 0;
    const int progress_step = 250;
    char progress_label[32];
    snprintf(progress_label, sizeof(progress_label), "Hybrid r%d", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < query_n; q++) {
        const float *query = queries + (long)q * query_dim;
        knn_search_with_distances(query,
                                  base,
                                  local_base_n,
                                  base_dim,
                                  K,
                                  local_distances + (long)q * K,
                                  local_indices + (long)q * K);

        #pragma omp atomic
        local_done++;

        if ((local_done % progress_step == 0) || (local_done == query_n)) {
            if (omp_get_thread_num() == 0) {
                progress_bar_print(progress_label, local_done, query_n);
            }
        }
    }

    /* Convert local indices to global indices using base_start offset */
    for (long idx = 0; idx < (long)query_n * K; idx++) {
        local_indices[idx] += (int)base_start;
    }

    float *gathered_distances = NULL;
    int *gathered_indices = NULL;
    int *results = NULL;
    Neighbor *candidates = NULL;

    if (rank == 0) {
        gathered_distances = (float *)malloc((long)size * query_n * K * sizeof(float));
        gathered_indices = (int *)malloc((long)size * query_n * K * sizeof(int));
        results = (int *)malloc((long)query_n * K * sizeof(int));
        candidates = (Neighbor *)malloc((long)size * K * sizeof(Neighbor));
        if (!gathered_distances || !gathered_indices || !results || !candidates) {
            fprintf(stderr, "Rank 0: cannot allocate gather/merge buffers\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    MPI_Gather(local_distances, query_n * K, MPI_FLOAT,
               gathered_distances, query_n * K, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_indices, query_n * K, MPI_INT,
               gathered_indices, query_n * K, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int q = 0; q < query_n; q++) {
            for (int r = 0; r < size; r++) {
                for (int i = 0; i < K; i++) {
                    long src = ((long)r * query_n + q) * K + i;
                    candidates[r * K + i].distance = gathered_distances[src];
                    candidates[r * K + i].index = gathered_indices[src];
                }
            }

            knn_select_topk_from_candidates(candidates, size * K, K, results + (long)q * K);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        progress_bar_finish(progress_label, query_n);
        double recall = compute_recall(results, ground_truth, query_n, K, gt_dim);

        printf("--------------------------------------------\n");
        printf("    RESULTS (Hybrid MPI + OpenMP)\n");
        printf("--------------------------------------------\n");
        printf("Total search time     : %.4f seconds\n", elapsed);
        printf("Avg query time        : %.6f seconds\n", elapsed / query_n);
        printf("Throughput            : %.2f queries/sec\n", query_n / elapsed);
        printf("Recall@%d             : %.2f%%\n", K, recall);
        printf("--------------------------------------------\n");

        printf("CSV:hybrid,%d,%d,%.4f,%.2f,%.2f\n", size, omp_threads, elapsed, query_n / elapsed, recall);

        /* Save results to file */
        FILE *outf = fopen("results/hybrid.txt", "a");
        if (outf) {
            fprintf(outf, "CSV:hybrid,%d,%d,%.4f,%.2f,%.2f\n", size, omp_threads, elapsed, query_n / elapsed, recall);
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


    }

    free(base);
    free(queries);
    free(ground_truth);
    free(local_indices);
    free(local_distances);
    free(gathered_distances);
    free(gathered_indices);
    free(results);
    free(candidates);

    MPI_Finalize();
    return EXIT_SUCCESS;
}