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
 * Read a contiguous range from a .fvecs file.
 *
 * @param filename  Path to the .fvecs file
 * @param start     Zero-based vector offset
 * @param count     Number of vectors to read
 * @param dim       Output: dimensionality of each vector
 * @param n         Output: number of vectors actually read
 * @return          Pointer to flat float array (n * dim elements)
 */
static __attribute__((unused)) float *fvecs_read_range(const char *filename,
                               long start,
                               long count,
                               int *dim,
                               int *n) {
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

    long vec_size = (long)sizeof(int) + (long)d * sizeof(float);
    long total_vecs = file_size / vec_size;

    if (start < 0 || start > total_vecs) {
        fprintf(stderr, "Error: invalid start offset %ld for '%s'\n", start, filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (count < 0 || start + count > total_vecs) {
        count = total_vecs - start;
    }

    if (count < 0) {
        count = 0;
    }

    *dim = d;
    *n = (int)count;

    float *data = (float *)malloc((long)count * d * sizeof(float));
    if (!data && count > 0) {
        fprintf(stderr, "Error: cannot allocate %.2f MB for %ld vectors\n",
                (double)count * d * sizeof(float) / (1024.0 * 1024.0), count);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (fseek(f, start * vec_size, SEEK_SET) != 0) {
        fprintf(stderr, "Error: cannot seek to vector %ld in '%s'\n", start, filename);
        free(data);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    for (long i = 0; i < count; i++) {
        int tmp_d;
        if (fread(&tmp_d, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error: read failed at vector %ld\n", start + i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        if (fread(data + i * d, sizeof(float), d, f) != (size_t)d) {
            fprintf(stderr, "Error: read failed at vector %ld data\n", start + i);
            free(data);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
    return data;
}

/**
 * Read only the metadata from a .fvecs file.
 *
 * @param filename  Path to the .fvecs file
 * @param dim       Output: dimensionality of each vector
 * @param n         Output: number of vectors in the file
 */
static void __attribute__((unused)) fvecs_metadata(const char *filename, int *dim, long *n) {
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
    fclose(f);

    long vec_size = (long)sizeof(int) + (long)d * sizeof(float);
    *dim = d;
    *n = file_size / vec_size;
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