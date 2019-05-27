#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define GRAPH_SIZE 2708
#define FEATURE_DEPTH 1433
#define LABELS 7
#define DEGREE 2

typedef union Data_union {
    uint32_t u;
    float f;
} dataUnion;

/* void copy_matrix(uint32_t dim1, uint32_t dim2, float src[dim1*dim2], float dest[dim1][dim2]) { */
/*     uint32_t i, j; */
/*     for(i = 0; i < dim1; i++) { */
/*         for(j = 0; j < dim2; j++) { */
/*             dest[i][j] = src[i*dim2 + j]; */
/*         } */
/*     } */
/* } */

/* void print_matrix_to_file(char* filename, uint32_t dim1, uint32_t dim2, float matrix[dim1][dim2]) { */
/*     uint32_t i, j; */
/*     FILE *f = fopen(filename, "wb"); */
/*     if (f == NULL) { */
/*         printf("Error opening file!\n"); */
/*         exit(1); */
/*     } */
/*     for(i = 0; i < dim1; i++) { */
/*         for(j = 0; j < dim2; j++) { */
/*             fwrite(&matrix[i][j], sizeof(float), 1, f); */
/*         } */
/*     } */
/*     fclose(f); */
/* } */


/* void full_matrix_mult () { */
/*     float *new_features = malloc(GRAPH_SIZE*FEATURE_DEPTH*sizeof(float)); */
/*     uint16_t d; */
/*     uint32_t i, j, k; */
/*     for(d = 0; d < DEGREE; d++){ */
/*         printf("Degree: %d\n", d); */
/*         for(i = 0; i < GRAPH_SIZE; i++) { */
/*             for(j = 0; j < FEATURE_DEPTH; j++) { */
/*                 new_features[i*FEATURE_DEPTH +j] = 0; */
/*                 for(k = 0; k < GRAPH_SIZE; k++) { */
/*                     new_features[i*FEATURE_DEPTH + j] += ADJ[i][k] * FEAT[k][j]; */
/*                 } */
/*             } */
/*         } */
/*         copy_matrix((uint32_t)GRAPH_SIZE, (uint32_t)FEATURE_DEPTH, new_features, FEAT); */
/*     } */
/*     free(new_features); */
/* } */

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

// mem_file must be freed by caller
float* read_float_file(char* filename) {
    FILE *f;
    uint32_t file_size;
    f = fopen(filename, "rb");
    fseek(f, 0L, SEEK_END);
    file_size = ftell(f);
    rewind(f);
    float *mem_file = malloc(file_size);
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

void* generate_normalised_adj_matrix (uint32_t* adj, float* adj_weights, float* deg) {
    uint32_t i = 0, j = 0, m = 0;
    uint32_t nb_nodes = adj[i++];
    size_t s_capacity = 4 * nb_nodes * sizeof(float); 
    dataUnion * s = malloc(s_capacity);
    uint32_t base_node = 0, dest_node, neighbor_nb;
    float weight;
    bool self_loop;
    s[j++].u = nb_nodes; 
    printf("nb_nodes = %zu\n", nb_nodes);
    printf("s_capacity: %zu\n", s_capacity);
    while (base_node < nb_nodes) {
        self_loop = false;
        neighbor_nb = adj[i++];
        s[j++].u = neighbor_nb + 1; // +1 for self-loop
        for (m = 0; m < neighbor_nb; m++) {
            // Check if s still has room
            if (j*sizeof(uint32_t) > s_capacity - 6*sizeof(uint32_t)) {
                s_capacity *= 2;
                s = realloc(s, s_capacity);
            }
            dest_node = adj[i];
            weight = adj_weights[i++];
            // add self loop
            if (dest_node > base_node && !self_loop) {
                s[j++].u = base_node;
                // self_loop weight of 1 assumed
                s[j++].f = (float)(deg[base_node] * deg[base_node]); 
                self_loop = true;
            }
            // Normalise
            s[j++].u = dest_node; // copy over second node number 
            s[j++].f = (float)(deg[base_node] * deg[dest_node] * weight);
        } 
        // If self loop condition was never reached because all dest nodes < base node
        if (!self_loop) {
            s[j++].u = base_node;
            s[j++].f = (float)(deg[base_node] * deg[base_node]);
            self_loop = true;
        }
        base_node++;
    }
    s = realloc(s, j * sizeof(float)); // fit memory allocation to final size of s

    // Print result to file
    FILE *f = fopen("c_norm_adj.bin", "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(i = 0; i < j; i++) {
        fwrite(&s[i], sizeof(float), 1, f);
    }
    fclose(f);

    return s;
}

float * sparse_matrix_mult(dataUnion * s, float * features) {
    uint32_t i = 0, base_node = 0, dest_node = 0, neighbor_nb;
    uint32_t nb_nodes = s[i++].u;  
    float adj_weight;
    float * new_features = calloc(nb_nodes * FEATURE_DEPTH, sizeof(float));
    while (base_node < nb_nodes) {
        neighbor_nb = s[i++].u;
        for (uint32_t k = 0; k < neighbor_nb; k++) {
            dest_node = s[i++].u;
            adj_weight = s[i++].f;
            // apply weight to all relevant features (Weight stationary)
            for (uint32_t m = 0; m < FEATURE_DEPTH; m++) {
                new_features[base_node*FEATURE_DEPTH + m] += adj_weight * features[dest_node*FEATURE_DEPTH + m];
            }
        }
        base_node++;
    }
    // Print result to file
    FILE *f = fopen("c_precomp.bin", "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(i = 0; i < nb_nodes*FEATURE_DEPTH; i++) {
        fwrite(&new_features[i], sizeof(float), 1, f);
    }
    fclose(f);
    free(features);
    return new_features;
}

// calculates inference for a given node 
void infer (float* features, float* weights, float* infered_res, uint32_t n) {
    uint32_t i, j;
    for (i = 0; i < LABELS; i++) {
        infered_res[n] = 0;
        for (j = 0; j < FEATURE_DEPTH; j++) {
            infered_res[n*LABELS + i] += features[n*FEATURE_DEPTH + j] * weights[i*FEATURE_DEPTH + j];
        }
    }
}

void soft_max (float* matrix, uint32_t n) {
    float max = 0, sum = 0;
    uint32_t i;
    for (i = 0; i < LABELS; i++) {
        float temp = matrix[n*LABELS + i];
        if (temp > max)
            max = temp;
    }
    for (i = 0; i < LABELS; i++) {
        matrix[n*LABELS + i] -= max;
        sum += matrix[n*LABELS + i];
    }
    for (i = 0; i < LABELS; i++) {
        matrix[n*LABELS + i] /= sum;
    }
}


int main() {
    clock_t begin = clock();
    
    uint32_t * adj = read_file("adj.bin");
    float * adj_weights = read_float_file("adj_weights.bin"); // Should actually be dataUnion type
    float * features = read_float_file("features.bin");
    float * degree = generate_degree_matrix(adj);
    dataUnion * s = generate_normalised_adj_matrix(adj, adj_weights, degree);
    for (uint32_t i = 0; i < DEGREE; i++) {
        features = sparse_matrix_mult(s, features);
    }

    /* full_matrix_mult(); */

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Precomputation time: %lf\n", time_spent);

    /* print_matrix_to_file("preprocess.bin", GRAPH_SIZE, FEATURE_DEPTH, FEAT); */
    
    begin = clock();
    // Weights are transposed from python: LABELS x FEATURE_DEPTH
    float * weights = read_float_file("python_starting_weights.bin"); // transposed
    float * infered_res = malloc(LABELS * GRAPH_SIZE * sizeof(float));
    // TODO create randomised training sub-goup
    for (uint32_t i = 0; i < GRAPH_SIZE; i++) {
        infer(features, weights, infered_res, i);
    }

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Inference time: %lf\n", time_spent);


    free(adj);
    free(degree);
    free(s);
    free(features);
    return 0;
}
