#ifndef IO_H
#define IO_H

#include <stdio.h>
#include <stdlib.h>

/**
 * Read a .fvecs file (float vectors).
 *
 * @param filename  Path to the .fvecs file
 * @param dim       Output: dimensionality of each vector
 * @param n         Output: number of vectors read
 * @return          Pointer to flat float array (n * dim elements)
 */
static float *fvecs_read(const char *filename, int *dim, int *n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    /* Read dimension from the first vector */
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: cannot read dimension from '%s'\n", filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Determine number of vectors from file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long vec_size = (long)sizeof(int) + (long)d * sizeof(float);
    long num_vecs = file_size / vec_size;

    *dim = d;
    *n = (int)num_vecs;

    /* Allocate contiguous array */
    float *data = (float *)malloc((long)num_vecs * d * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: cannot allocate %.2f MB for %ld vectors\n",
                (double)num_vecs * d * sizeof(float) / (1024.0 * 1024.0), num_vecs);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Read all vectors */
    for (long i = 0; i < num_vecs; i++) {
        int tmp_d;
        if (fread(&tmp_d, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error: read failed at vector %ld\n", i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        if (fread(data + i * d, sizeof(float), d, f) != (size_t)d) {
            fprintf(stderr, "Error: read failed at vector %ld data\n", i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
    return data;
}

/**
 * Read a .ivecs file (integer vectors).
 *
 * @param filename  Path to the .ivecs file
 * @param dim       Output: dimensionality of each vector
 * @param n         Output: number of vectors read
 * @return          Pointer to flat int array (n * dim elements)
 */
static int *ivecs_read(const char *filename, int *dim, int *n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: cannot read dimension from '%s'\n", filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long vec_size = (long)sizeof(int) + (long)d * sizeof(int);
    long num_vecs = file_size / vec_size;

    *dim = d;
    *n = (int)num_vecs;

    int *data = (int *)malloc((long)num_vecs * d * sizeof(int));
    if (!data) {
        fprintf(stderr, "Error: cannot allocate %.2f MB for %ld vectors\n",
                (double)num_vecs * d * sizeof(int) / (1024.0 * 1024.0), num_vecs);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    for (long i = 0; i < num_vecs; i++) {
        int tmp_d;
        if (fread(&tmp_d, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error: read failed at vector %ld\n", i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        if (fread(data + i * d, sizeof(int), d, f) != (size_t)d) {
            fprintf(stderr, "Error: read failed at vector %ld data\n", i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
    return data;
}

#endif