/**
 * distance.h - Distance computation functions for vector similarity search
 *
 * Uses squared Euclidean (L2) distance — standard for the SIFT dataset.
 * The square root is omitted because it is a monotonic function and does
 * not affect the ranking of nearest neighbors.
 */

#ifndef DISTANCE_H
#define DISTANCE_H

/**
 * Compute squared Euclidean distance between two vectors.
 *
 * @param a    First vector  (length = dim)
 * @param b    Second vector (length = dim)
 * @param dim  Dimensionality
 * @return     sum of (a[i] - b[i])^2
 */
static inline float l2_distance_squared(const float *a, const float *b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

#endif /* DISTANCE_H */
