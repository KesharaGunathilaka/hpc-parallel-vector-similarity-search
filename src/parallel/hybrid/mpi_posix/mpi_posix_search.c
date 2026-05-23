#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "io.h"
#include "distance.h"
#include "knn.h"
#include "progress.h"

#define K          10
#define BASE_PATH   "data/sift_base.fvecs"
#define QUERY_PATH  "data/sift_query.fvecs"
#define GT_PATH     "data/sift_groundtruth.ivecs"

/* ---- Per-thread argument block ---- */
typedef struct {
    int          thread_id;
    int          start_query;   /* inclusive */
    int          end_query;     /* exclusive */
    const float *base;
    const float *queries;
    float       *local_distances;
    int         *local_indices;
    int          base_n;
    int          query_n;
    int          dim;
    int          k;
    int         *progress;
    int          total;
    int          step;
    pthread_mutex_t *progress_mutex;
    int          queries_done;  /* written only by this thread */
} ThreadArg;

static void compute_partition(long total, int rank, int size, long *start, long *count) {
    long begin = (total * rank) / size;
    long end = (total * (rank + 1)) / size;
    *start = begin;
    *count = end - begin;
}

/* ---- Thread worker function ---- */
static void *search_worker(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;

    int count = 0;
    for (int q = ta->start_query; q < ta->end_query; q++) {
        const float *query = ta->queries + (long)q * ta->dim;
        knn_search_with_distances(query,
                                  ta->base,
                                  ta->base_n,
                                  ta->dim,
                                  ta->k,
                                  ta->local_distances + (long)q * ta->k,
                                  ta->local_indices + (long)q * ta->k);
        count++;

        pthread_mutex_lock(ta->progress_mutex);
        (*ta->progress)++;
        if ((*ta->progress) % ta->step == 0 || *ta->progress == ta->total) {
            progress_bar_print("Hybrid", *ta->progress, ta->total);
        }
        pthread_mutex_unlock(ta->progress_mutex);
    }
    ta->queries_done = count;
    return NULL;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pthread_count = 4;
    if (argc >= 2) {
        pthread_count = atoi(argv[1]);
        if (pthread_count < 1) {
            pthread_count = 1;
        }
    }

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
        printf("  Hybrid MPI + POSIX k-NN Search (SIFT 1M)\n");
        printf("============================================\n\n");
        printf("Base vectors   : %ld total\n", base_total);
        printf("Query vectors  : %d\n", query_n);
        printf("Ground truth   : %d entries (neighbors/entry = %d)\n", gt_n, gt_dim);
        printf("k              : %d\n", K);
        printf("MPI ranks      : %d\n", size);
        printf("POSIX threads  : %d\n", pthread_count);
        printf("\n");
    }

    int *local_indices = (int *)malloc((long)query_n * K * sizeof(int));
    float *local_distances = (float *)malloc((long)query_n * K * sizeof(float));
    if (!local_indices || !local_distances) {
        fprintf(stderr, "Rank %d: cannot allocate local result buffers\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ---- Allocate and initialize thread resources ---- */
    pthread_t *threads = (pthread_t *)malloc(pthread_count * sizeof(pthread_t));
    ThreadArg *args = (ThreadArg *)malloc(pthread_count * sizeof(ThreadArg));
    
    if (!threads || !args) {
        fprintf(stderr, "Rank %d: cannot allocate thread resources\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int *progress = (int *)malloc(sizeof(int));
    pthread_mutex_t *progress_mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
    
    if (!progress || !progress_mutex) {
        fprintf(stderr, "Rank %d: cannot allocate progress tracking\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    *progress = 0;
    pthread_mutex_init(progress_mutex, NULL);

    const int progress_step = 250;
    
    /* ---- Partition queries among threads ---- */
    int queries_per_thread = query_n / pthread_count;
    int extra = query_n % pthread_count;
    int offset = 0;

    for (int t = 0; t < pthread_count; t++) {
        int count = queries_per_thread + (t < extra ? 1 : 0);
        args[t].thread_id = t;
        args[t].start_query = offset;
        args[t].end_query = offset + count;
        args[t].base = base;
        args[t].queries = queries;
        args[t].local_distances = local_distances;
        args[t].local_indices = local_indices;
        args[t].base_n = local_base_n;
        args[t].query_n = query_n;
        args[t].dim = base_dim;
        args[t].k = K;
        args[t].progress = progress;
        args[t].total = query_n;
        args[t].step = progress_step;
        args[t].progress_mutex = progress_mutex;
        args[t].queries_done = 0;
        offset += count;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* ---- Launch threads ---- */
    for (int t = 0; t < pthread_count; t++) {
        int rc = pthread_create(&threads[t], NULL, search_worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "Rank %d: pthread_create failed for thread %d (rc=%d)\n", rank, t, rc);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    /* ---- Wait for all threads ---- */
    for (int t = 0; t < pthread_count; t++) {
        pthread_join(threads[t], NULL);
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
        progress_bar_finish("Hybrid", query_n);
        double recall = compute_recall(results, ground_truth, query_n, K, gt_dim);

        printf("--------------------------------------------\n");
        printf("    RESULTS (Hybrid MPI + POSIX)\n");
        printf("--------------------------------------------\n");
        printf("Total search time     : %.4f seconds\n", elapsed);
        printf("Avg query time        : %.6f seconds\n", elapsed / query_n);
        printf("Throughput            : %.2f queries/sec\n", query_n / elapsed);
        printf("Recall@%d             : %.2f%%\n", K, recall);
        printf("--------------------------------------------\n");

        printf("CSV:hybrid_posix,%d,%d,%.4f,%.2f,%.2f\n", size, pthread_count, elapsed, query_n / elapsed, recall);

        /* Save results to file */
        FILE *outf = fopen("results/hybrid_posix.txt", "a");
        if (outf) {
            fprintf(outf, "CSV:hybrid_posix,%d,%d,%.4f,%.2f,%.2f\n", size, pthread_count, elapsed, query_n / elapsed, recall);
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

    /* ---- Cleanup ---- */
    free(base);
    free(queries);
    free(ground_truth);
    free(local_indices);
    free(local_distances);
    free(gathered_distances);
    free(gathered_indices);
    free(results);
    free(candidates);
    free(threads);
    free(args);
    free(progress);
    pthread_mutex_destroy(progress_mutex);
    free(progress_mutex);

    MPI_Finalize();
    return EXIT_SUCCESS;
}