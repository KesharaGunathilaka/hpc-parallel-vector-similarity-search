
#ifndef KNN_H
#define KNN_H

#include "distance.h"


typedef struct {
    float distance;   
    int   index;   
} Neighbor;

static void heap_sift_down(Neighbor *heap, int size, int i) {
    while (1) {
        int largest = i;
        int left    = 2 * i + 1;
        int right   = 2 * i + 2;

        if (left  < size && heap[left].distance  > heap[largest].distance)
            largest = left;
        if (right < size && heap[right].distance > heap[largest].distance)
            largest = right;

        if (largest == i) break;

        Neighbor tmp   = heap[i];
        heap[i]        = heap[largest];
        heap[largest]  = tmp;
        i = largest;
    }
}

static void heap_build(Neighbor *heap, int size) {
    for (int i = size / 2 - 1; i >= 0; i--)
        heap_sift_down(heap, size, i);
}

/* ---------- k-NN search for a single query ---------- */

/**
 * Brute-force k-NN search for one query vector.
 *
 * @param query    Query vector (length = dim)
 * @param base     Base dataset  (base_n * dim contiguous floats)
 * @param base_n   Number of base vectors
 * @param dim      Dimensionality
 * @param k        Number of neighbors to find
 * @param results  Output array of k indices (sorted ascending by distance)
 */
static void __attribute__((unused)) knn_search(const float *query,
                       const float *base,
                       int base_n,
                       int dim,
                       int k,
                       int *results) {
    // Max-heap holding the k best candidates seen so far
    Neighbor *heap = (Neighbor *)malloc(k * sizeof(Neighbor));

    // Seed the heap with the first k base vectors
    for (int i = 0; i < k; i++) {
        heap[i].distance = l2_distance_squared(query, base + (long)i * dim, dim);
        heap[i].index    = i;
    }
    heap_build(heap, k);

    // Scan remaining base vectors
    for (int i = k; i < base_n; i++) {
        float dist = l2_distance_squared(query, base + (long)i * dim, dim);
        if (dist < heap[0].distance) {
            heap[0].distance = dist;
            heap[0].index    = i;
            heap_sift_down(heap, k, 0);
        }
    }

    // Sort the k results by distance
    for (int i = 1; i < k; i++) {
        Neighbor key = heap[i];
        int j = i - 1;
        while (j >= 0 && heap[j].distance > key.distance) {
            heap[j + 1] = heap[j];
            j--;
        }
        heap[j + 1] = key;
    }

    // Copy indices to output
    for (int i = 0; i < k; i++)
        results[i] = heap[i].index;
    free(heap);
}

/**
 * Brute-force k-NN search for one query vector, preserving distances.
 */
static void __attribute__((unused)) knn_search_with_distances(const float *query,
                                      const float *base,
                                      int base_n,
                                      int dim,
                                      int k,
                                      float *out_distances,
                                      int *out_indices) {
    Neighbor *heap = (Neighbor *)malloc(k * sizeof(Neighbor));


    for (int i = 0; i < k; i++) {
        heap[i].distance = l2_distance_squared(query, base + (long)i * dim, dim);
        heap[i].index = i;
    }
    heap_build(heap, k);

    for (int i = k; i < base_n; i++) {
        float dist = l2_distance_squared(query, base + (long)i * dim, dim);
        if (dist < heap[0].distance) {
            heap[0].distance = dist;
            heap[0].index = i;
            heap_sift_down(heap, k, 0);
        }
    }

    for (int i = 1; i < k; i++) {
        Neighbor key = heap[i];
        int j = i - 1;
        while (j >= 0 && heap[j].distance > key.distance) {
            heap[j + 1] = heap[j];
            j--;
        }
        heap[j + 1] = key;
    }

    for (int i = 0; i < k; i++) {
        if (out_distances) {
            out_distances[i] = heap[i].distance;
        }
        if (out_indices) {
            out_indices[i] = heap[i].index;
        }
    }
    free(heap);
}

/**
 * Select the top-k candidates from an already gathered candidate pool.
 */
static void __attribute__((unused)) knn_select_topk_from_candidates(const Neighbor *candidates,
                                            int candidate_count,
                                            int k,
                                            int *results) {
    if (candidate_count <= 0 || k <= 0) {
        return;
    }

    int heap_size = candidate_count < k ? candidate_count : k;
    Neighbor *heap = (Neighbor *)malloc(k * sizeof(Neighbor));


    for (int i = 0; i < heap_size; i++) {
        heap[i] = candidates[i];
    }

    heap_build(heap, heap_size);

    for (int i = heap_size; i < candidate_count; i++) {
        if (candidates[i].distance < heap[0].distance) {
            heap[0] = candidates[i];
            heap_sift_down(heap, heap_size, 0);
        }
    }

    for (int i = 1; i < heap_size; i++) {
        Neighbor key = heap[i];
        int j = i - 1;
        while (j >= 0 && heap[j].distance > key.distance) {
            heap[j + 1] = heap[j];
            j--;
        }
        heap[j + 1] = key;
    }

    for (int i = 0; i < heap_size; i++) {
        results[i] = heap[i].index;
    }
    for (int i = heap_size; i < k; i++) {
        results[i] = -1;
    }
    free(heap);
}



/**
 * Compute Recall@k: fraction of predicted neighbors that appear in
 * the ground truth's top-k list.
 *
 * @param results        Predicted indices (query_n * k)
 * @param ground_truth   Ground truth indices (query_n * gt_dim)
 * @param query_n        Number of queries
 * @param k              Number of neighbors predicted
 * @param gt_dim         Number of ground truth neighbors per query
 * @return               Recall percentage (0–100)
 */
static double compute_recall(const int *results,
                             const int *ground_truth,
                             int query_n,
                             int k,
                             int gt_dim) {
    int correct = 0;
    int total   = query_n * k;

    for (int q = 0; q < query_n; q++) {
        for (int i = 0; i < k; i++) {
            int pred = results[(long)q * k + i];
            for (int j = 0; j < k && j < gt_dim; j++) {
                if (pred == ground_truth[(long)q * gt_dim + j]) {
                    correct++;
                    break;
                }
            }
        }
    }

    return (double)correct / total * 100.0;
}



#endif
