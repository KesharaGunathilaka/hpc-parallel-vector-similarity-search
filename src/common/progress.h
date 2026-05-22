#ifndef PROGRESS_H
#define PROGRESS_H

#include <stdio.h>

static inline void progress_bar_print(const char *label, int current, int total) {
    const int width = 30;
    if (total <= 0) {
        return;
    }

    if (current < 0) {
        current = 0;
    }
    if (current > total) {
        current = total;
    }

    int filled = (int)((long long)current * width / total);
    if (!label) {
        label = "Progress";
    }

    fprintf(stderr, "\r%s [", label);
    for (int i = 0; i < width; i++) {
        fputc(i < filled ? '#' : '-', stderr);
    }
    fprintf(stderr, "] %d/%d (%.1f%%)", current, total, (double)current * 100.0 / total);
    fflush(stderr);
}

static inline void progress_bar_finish(const char *label, int total) {
    progress_bar_print(label, total, total);
    fprintf(stderr, "\n");
    fflush(stderr);
}

#endif