#include "data/adj.h"
#include "data/feat.h"
#include "data/idx_test.h"
#include "data/idx_train.h"
#include "data/idx_val.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GRAPH_SIZE 2708
#define FEATURE_DEPTH 1433
#define DEGREE 2

void full_matrix_mult () {
    float *new_features = malloc(GRAPH_SIZE*FEATURE_DEPTH*sizeof(float));
    int d, i, j, k;
    for(d = 0; d < DEGREE; d++){
        printf("Degree: %d\n", d);
        for(i = 0; i < GRAPH_SIZE; i++) {
            //printf("Outer: %d\n", i);
            for(j = 0; j < FEATURE_DEPTH; j++) {
                new_features[i*FEATURE_DEPTH +j] = 0;
                for(k = 0; k < GRAPH_SIZE; k++) {
                    new_features[i*FEATURE_DEPTH + j] += ADJ[i][k] * FEAT[k][j];
                }
            }
        }
    }
    free(new_features);
}

int main() {
    clock_t begin = clock();
    
    full_matrix_mult();

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Precomputation time: %lf\n", time_spent);
    return 0;
}
