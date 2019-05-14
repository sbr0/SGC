#include "data/adj.h"
#include "data/feat.h"
#include "data/idx_test.h"
#include "data/idx_train.h"
#include "data/idx_val.h"
//#include "data/simple.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GRAPH_SIZE 2708
#define FEATURE_DEPTH 1433
#define DEGREE 2

void copy_matrix(uint32_t dim1, uint32_t dim2, float src[dim1*dim2], float dest[dim1][dim2]) {
    uint32_t i, j;
    for(i = 0; i < dim1; i++) {
        for(j = 0; j < dim2; j++) {
            dest[i][j] = src[i*dim2 + j];
        }
    }
}
void print_matrix_to_file(char* filename, uint32_t dim1, uint32_t dim2, float matrix[dim1][dim2]) {
    uint32_t i, j;
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(i = 0; i < dim1; i++) {
        for(j = 0; j < dim2; j++) {
            fwrite(&matrix[i][j], sizeof(float), 1, f);
        }
    }
    fclose(f);
}

void full_matrix_mult () {
    float *new_features = malloc(GRAPH_SIZE*FEATURE_DEPTH*sizeof(float));
    uint16_t d;
    uint32_t i, j, k;
    for(d = 0; d < DEGREE; d++){
        printf("Degree: %d\n", d);
        for(i = 0; i < GRAPH_SIZE; i++) {
            for(j = 0; j < FEATURE_DEPTH; j++) {
                new_features[i*FEATURE_DEPTH +j] = 0;
                for(k = 0; k < GRAPH_SIZE; k++) {
                    new_features[i*FEATURE_DEPTH + j] += ADJ[i][k] * FEAT[k][j];
                }
            }
        }
        copy_matrix((uint32_t)GRAPH_SIZE, (uint32_t)FEATURE_DEPTH, new_features, FEAT);
    }
    free(new_features);
}

void read_sparse_array() {
    FILE *f;
    float *new_features = calloc(GRAPH_SIZE*FEATURE_DEPTH, sizeof(float));
    uint16_t d;
    uint32_t i, j, k, array_size, neighbor_number;
    float adj_weight;
    f = fopen("sparce.bin", "rb");
    for(d = 0; d < DEGREE; d++) {
        fseek(f, 0, SEEK_SET);  // reset file pointer to start of file
        fread(&array_size, sizeof(uint32_t), 1, f);
        for(i = 0; i < array_size; i++) {
            fread(&neighbor_number, sizeof(uint32_t), 1, f);
            new_features[i*FEATURE_DEPTH +j] = 0;
            for(k = 0; k < neighbor_number; j++) {
                fread(&adj_weight, sizeof(uint32_t), 1, f);
                for(j = 0; j < FEATURE_DEPTH; j++) {
                    // TODO See how adj array is created, we need coeffs for multiplication.
                }
            }
        }
    }
} 

int main() {
    clock_t begin = clock();
    
    full_matrix_mult();

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Precomputation time: %lf\n", time_spent);

    print_matrix_to_file("preprocess.bin", GRAPH_SIZE, FEATURE_DEPTH, FEAT);
    return 0;
}
