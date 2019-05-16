#define SIMPLE

#ifndef SIMPLE
    #include "data/adj.h"
    #include "data/feat.h"
    #include "data/idx_test.h"
    #include "data/idx_train.h"
    #include "data/idx_val.h"
    #define GRAPH_SIZE 2708
    #define FEATURE_DEPTH 1433
#endif
#ifdef SIMPLE
    #include "data/simple.h"
    #define GRAPH_SIZE 5
    #define FEATURE_DEPTH 3
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

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

// mem_file must be freed by caller
uint32_t* read_file(char* filename) {
    FILE *f;
    uint32_t file_size;
    f = fopen(filename, "rb");
    fseek(f, 0L, SEEK_END);
    file_size = ftell(f);
    rewind(f);
    uint32_t *mem_file = malloc(file_size);
    fread(mem_file, 1, file_size, f);
    fclose(f);
    
    return mem_file;
} 

float* generate_degree_matrix(uint32_t* adj) {
    uint32_t i = 0;
    uint32_t size = adj[i++];
    uint32_t node = 0;
    uint32_t node_l;
    float *degree_matrix = calloc(size, sizeof(float));
    while(node < size) {
        node_l = adj[i];
        i += node_l + 1; // Skip adjacent node numbers
        degree_matrix[node] = 1 / (float)sqrt(node_l + 1); // +1 for the added self_loop
        // printf("Node: %zu :%f\t", node, degree_matrix[node]);
        node++;
    }
    
    FILE *f = fopen("c_degree.bin", "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(i = 0; i < size; i++) {
        fwrite(&degree_matrix[i], sizeof(float), 1, f);
    }
    fclose(f);
    return degree_matrix;
}

void* sgc_precompute (uint32_t* adj, float* deg) {
    printf("SGC_precomp\n");
    uint32_t i = 0, j = 0, m = 0;
    uint32_t nb_nodes = adj[i++];
    size_t s_capacity = 4 * nb_nodes * sizeof(float); 
    float * s = (float *) malloc(s_capacity);
    uint32_t base_node = 0, dest_node, neighbor_nb;
    bool self_loop;
    s[j++] = nb_nodes; 
    printf("nb_nodes = %zu\n", nb_nodes);
    printf("s_capacity: %zu\n", s_capacity);
    while (base_node < nb_nodes) {
        printf("[%zu]:[", base_node);
        self_loop = false;
        neighbor_nb = adj[i++];
        s[j++] = neighbor_nb + 1; // +1 for self-loop
        for (m = 0; m < neighbor_nb; m++) {
            // Check if s still has room
            if (j*sizeof(uint32_t) > s_capacity - 6*sizeof(uint32_t)) {
                s_capacity *= 2;
                s = (float *) realloc(s, s_capacity);
                printf("\nREALLOC_SUCCESS\n");
            }
            dest_node = adj[i++];
            // add self loop
            if (dest_node > base_node && !self_loop) {
                s[j++] = base_node;
                s[j++] = (float)(deg[base_node] * deg[base_node]);
                self_loop = true;
            }
            // Normalise
            s[j++] = dest_node; // copy over second node number 
            s[j++] = (float)(deg[base_node] * deg[dest_node]);
            /* printf(" [%zu]: %.3f", dest_node, s[j-1]); */
        } 
        base_node ++;
        printf("%zu]",j);
    }
    printf("SUCCESS\n");
    s = realloc(s, j); // fit memory allocation to final size of s
    return s;
    // TODO Write result to file and compare with python results
}


/* void read_sparse_array() { */
/*     FILE *f; */
/*     float *new_features = calloc(GRAPH_SIZE*FEATURE_DEPTH, sizeof(float)); */
/*     uint16_t d; */
/*     uint32_t i, j, k, array_size, neighbor_number; */
/*     float adj_weight; */
/*     f = fopen("sparce.bin", "rb"); */
/*     for(d = 0; d < DEGREE; d++) { */
/*         fseek(f, 0, SEEK_SET);  // reset file pointer to start of file */
/*         fread(&array_size, sizeof(uint32_t), 1, f); */
/*         for(i = 0; i < array_size; i++) { */
/*             fread(&neighbor_number, sizeof(uint32_t), 1, f); */
/*             new_features[i*FEATURE_DEPTH +j] = 0; */
/*             for(k = 0; k < neighbor_number; j++) { */
/*                 fread(&adj_weight, sizeof(uint32_t), 1, f); */
/*                 for(j = 0; j < FEATURE_DEPTH; j++) { */
/*                     // TODO See how adj array is created, we need coeffs for multiplication. */
/*                 } */
/*             } */
/*         } */
/*     } */
/*     fclose(f); */
/* } */ 

int main() {
    clock_t begin = clock();
    
    uint32_t * adj = read_file("sparce.bin");
    float * degree = generate_degree_matrix(adj);
    void * s = sgc_precompute(adj, degree);

    /* full_matrix_mult(); */

    /* clock_t end = clock(); */
    /* double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; */
    /* printf("Precomputation time: %lf\n", time_spent); */

    /* print_matrix_to_file("preprocess.bin", GRAPH_SIZE, FEATURE_DEPTH, FEAT); */

    free(adj);
    free(degree);
    free(s);
    return 0;
}
