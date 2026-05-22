#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "io.h"
#include "distance.h"
#include "knn.h"
#include "progress.h"

/* ---- Configuration ---- */
#define K               10
#define DEFAULT_THREADS 4
#define BASE_PATH       "data/sift_base.fvecs"
#define QUERY_PATH      "data/sift_query.fvecs"
#define GT_PATH         "data/sift_groundtruth.ivecs"

/* ---- Shared read-only data (set before thread creation) ---- */
typedef struct {
    const float *base;
    const float *queries;
    int         *results;       /* each thread writes to its own region */
    int          base_n;
    int          query_n;
    int          dim;
    int          k;
} SharedData;

/* ---- Per-thread argument block ---- */
typedef struct {
    int          thread_id;
    int          start_query;   /* inclusive */
    int          end_query;     /* exclusive */
    SharedData  *shared;
    struct ProgressState *progress_state;
    /* progress tracking */
    int          queries_done;  /* written only by this thread */
} ThreadArg;

typedef struct ProgressState {
    int *progress;
    int total;
    int step;
    pthread_mutex_t *mutex;
} ProgressState;

/* ---- Thread worker function ---- */
static void *search_worker(void *arg) {
    ThreadArg  *ta = (ThreadArg *)arg;
    SharedData *sd = ta->shared;
    ProgressState *ps = ta->progress_state;

    int count = 0;
    for (int q = ta->start_query; q < ta->end_query; q++) {
        const float *query = sd->queries + (long)q * sd->dim;
        knn_search(query, sd->base, sd->base_n, sd->dim, sd->k,
                   sd->results + (long)q * sd->k);
        count++;

        pthread_mutex_lock(ps->mutex);
        (*ps->progress)++;
        if ((*ps->progress) % ps->step == 0 || *ps->progress == ps->total) {
            progress_bar_print("Pthreads", *ps->progress, ps->total);
        }
        pthread_mutex_unlock(ps->mutex);
    }
    ta->queries_done = count;
    return NULL;
}

int main(int argc, char *argv[]) {
    /* ---- Thread count ---- */
    int num_threads = DEFAULT_THREADS;
    if (argc >= 2) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) num_threads = 1;
    }

    /* ---- Load dataset ---- */
    int base_dim, base_n;
    int query_dim, query_n;
    int gt_dim, gt_n;

    printf("============================================\n");
    printf("  POSIX Threads k-NN Search (SIFT 1M)\n");
    printf("============================================\n\n");

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
    printf("Threads        : %d\n", num_threads);
    printf("\n");

    if (base_dim != query_dim) {
        fprintf(stderr, "Error: dimension mismatch (base=%d, query=%d)\n",
                base_dim, query_dim);
        return EXIT_FAILURE;
    }

    /* ---- Allocate result storage ---- */
    int *results = (int *)malloc((long)query_n * K * sizeof(int));
    if (!results) {
        fprintf(stderr, "Error: cannot allocate results array\n");
        return EXIT_FAILURE;
    }

    int progress = 0;
    pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;
    ProgressState progress_state = {
        .progress = &progress,
        .total = query_n,
        .step = 250,
        .mutex = &progress_mutex
    };

    /* ---- Prepare shared data ---- */
    SharedData shared = {
        .base    = base,
        .queries = queries,
        .results = results,
        .base_n  = base_n,
        .query_n = query_n,
        .dim     = base_dim,
        .k       = K
    };

    /* ---- Partition queries among threads ---- */
    pthread_t  *threads = (pthread_t  *)malloc(num_threads * sizeof(pthread_t));
    ThreadArg  *args    = (ThreadArg  *)malloc(num_threads * sizeof(ThreadArg));

    int queries_per_thread = query_n / num_threads;
    int extra              = query_n % num_threads;
    int offset             = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = queries_per_thread + (t < extra ? 1 : 0);
        args[t].thread_id   = t;
        args[t].start_query = offset;
        args[t].end_query   = offset + count;
        args[t].shared      = &shared;
        args[t].progress_state = &progress_state;
        args[t].queries_done = 0;
        offset += count;
    }

    /* ---- Launch threads ---- */
    fprintf(stderr, "Starting POSIX threads search with %d threads ...\n", num_threads);
    fprintf(stderr, "  Query distribution: ");
    for (int t = 0; t < num_threads; t++) {
        fprintf(stderr, "T%d[%d-%d] ", t, args[t].start_query, args[t].end_query - 1);
    }
    fprintf(stderr, "\n");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_create(&threads[t], NULL, search_worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d (rc=%d)\n", t, rc);
            return EXIT_FAILURE;
        }
    }

    /* ---- Wait for all threads ---- */
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    progress_bar_finish("Pthreads", query_n);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    /* ---- Per-thread stats ---- */
    fprintf(stderr, "  Per-thread completed: ");
    for (int t = 0; t < num_threads; t++)
        fprintf(stderr, "T%d=%d ", t, args[t].queries_done);
    fprintf(stderr, "\n");

    /* ---- Compute recall ---- */
    double recall = compute_recall(results, ground_truth, query_n, K, gt_dim);

    /* ---- Print results ---- */
    printf("--------------------------------------------\n");
    printf("    RESULTS (POSIX Threads, %d threads)\n", num_threads);
    printf("--------------------------------------------\n");
    printf("Total search time     : %.4f seconds\n", elapsed);
    printf("Avg query time        : %.6f seconds\n", elapsed / query_n);
    printf("Throughput            : %.2f queries/sec\n", query_n / elapsed);
    printf("Recall@%d             : %.2f%%\n", K, recall);
    printf("--------------------------------------------\n");

    /* Machine-readable */
    printf("CSV:pthread,%d,%.4f,%.2f,%.2f\n", num_threads, elapsed,
           query_n / elapsed, recall);

    /* Save results to file */
    FILE *outf = fopen("results/pthread.txt", "a");
    if (outf) {
        fprintf(outf, "CSV:pthread,%d,%.4f,%.2f,%.2f\n", num_threads, elapsed,
                query_n / elapsed, recall);
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

    /* ---- Sample results ---- */
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

    /* ---- Cleanup ---- */
    free(base);
    free(queries);
    free(ground_truth);
    free(results);
    free(threads);
    free(args);

    pthread_mutex_destroy(&progress_mutex);

    return EXIT_SUCCESS;
}
