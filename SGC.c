#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define GRAPH_SIZE 2708
#define FEATURE_DEPTH 1433
#define LABELS 7
#define DEGREE 3
#define ALPHA 0.2
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 0.00000001
#define WEIGHT_DECAY 5e-2
/* #define WEIGHT_DECAY 0.001 */
#define EPOCHS 100

#define TRAIN_START 0
#define TRAIN_END 139
#define VAL_START 140
#define VAL_END 639
#define TEST_START 1708
#define TEST_END 2707

typedef union Data_union {
    uint32_t u;
    float f;
} dataUnion;

struct CSR {
    float * val;
    uint32_t * idx;
    uint32_t val_length;
    uint32_t * ptr;
} CSR;

struct float_array_with_size {
    uint32_t size;
    float * array;
} float_array_with_size;

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
    if (f == NULL) {
        fprintf(stderr, "file %s could not be opended, aborting\n", filename);
    }
    fseek(f, 0L, SEEK_END);
    file_size = ftell(f);
    rewind(f);
    float *mem_file = malloc(file_size);
    fread(mem_file, 1, file_size, f);
    fclose(f);
    
    return mem_file;
} 

// mem_file must be freed by caller
// alternate version of read_float_file that preserves the
// size information of the array
struct float_array_with_size read_float_file_with_size(char* filename) {
    FILE *f;
    struct float_array_with_size array;
    f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "file %s could not be opended, aborting\n", filename);
    }
    fseek(f, 0L, SEEK_END);
    array.size = ftell(f);
    rewind(f);
    array.array = malloc(array.size);
    fread(array.array, 1, array.size, f);
    fclose(f);
    
    return array;
} 

void write_matrix(char* filename, uint32_t* matrix, uint32_t size) {
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(int i = 0; i < size; i++) {
        fwrite(&matrix[i], sizeof(uint32_t), 1, f);
    }
    fclose(f);
}

void write_float_matrix(char* filename, float* matrix, uint32_t size) {
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(int i = 0; i < size; i++) {
        fwrite(&matrix[i], sizeof(float), 1, f);
    }
    fclose(f);
}

void write_res(char* filename, struct CSR new_features) {

    // Print result to file to compare result
    // Converts back to dense format for a 1 to 1 comparison
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(int i = 0; i < GRAPH_SIZE; i++) {
        uint32_t nb_nodes, nf_pos;
        float zero = 0.0;
        nf_pos = new_features.ptr[i];
        if ( i < GRAPH_SIZE - 1 ) {
            nb_nodes = new_features.ptr[i+1] - nf_pos;
        } else {
            nb_nodes = new_features.val_length - nf_pos;
        }
        for (uint32_t j = 0; j < FEATURE_DEPTH; j++) {
            if (j == new_features.idx[nf_pos]) {
                fwrite(&new_features.val[nf_pos], sizeof(float), 1, f);
                if (nf_pos < new_features.ptr[i] + nb_nodes) {
                    nf_pos++;
                }
            } else {
                fwrite(&zero, sizeof(float), 1, f);
            } 
        }
    }
    fclose(f);
}

/* Generates a Compressed Sparse Row (CSR) struct from
 * its val, idx and ptr binary data files
 * */
struct CSR read_CSR(char* filename_val, char* filename_idx, char* filename_ptr) {
    struct CSR new_CSR;
    struct float_array_with_size val_array;

    val_array = read_float_file_with_size(filename_val);

    new_CSR.val = val_array.array;
    new_CSR.val_length = val_array.size / sizeof(float);
    new_CSR.idx = read_file(filename_idx);
    new_CSR.ptr = read_file(filename_ptr);
    
    return new_CSR;
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
    
    #ifndef ARM_PLATFORM
        write_float_matrix("c_degree.bin", degree_matrix, size);
    #endif

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
    // Assumptions that dataunion types of s will both be 32bit are made here
    s = realloc(s, j * sizeof(float)); // fit memory allocation to final size of s

    #ifndef ARM_PLATFORM
        write_float_matrix("c_norm_adj.bin", (float*) s, j);
    #endif
        
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
    #ifndef ARM_PLATFORM
        write_float_matrix("c_precomp.bin", new_features, nb_nodes*FEATURE_DEPTH);
    #endif
    return new_features;
}

struct CSR sparse_matrix_mult_CSR(dataUnion * s, struct CSR features) {
    uint32_t i = 0, base_node = 0, dest_node = 0, neighbor_nb;
    uint32_t nb_nodes = s[i++].u;  
    float adj_weight;
    struct CSR new_features;
    size_t nf_size = 1000 * sizeof(float); // Warning: float and uint32_t must be same size
    new_features.val = malloc(nf_size); 
    new_features.idx = malloc(nf_size); 
    new_features.ptr = malloc(GRAPH_SIZE * sizeof(uint32_t)); 
    uint32_t nf_count = 0;

    while (base_node < nb_nodes) {
        /* printf("ALIVE %d i: %d nf_count: %d\n", base_node, i, nf_count); */
        float* CSR_dense_row = calloc(FEATURE_DEPTH, sizeof(float));
        
        neighbor_nb = s[i++].u;
        for (uint32_t k = 0; k < neighbor_nb; k++) {
            uint32_t m, next_row, row_start;
            dest_node = s[i++].u;
            adj_weight = s[i++].f;
            // apply weight to all relevant features (Weight stationary)
            row_start = features.ptr[dest_node];
            if (dest_node < GRAPH_SIZE - 1) {
                next_row = features.ptr[dest_node + 1];
            } else {
                next_row = features.val_length;
            }
            m = next_row - row_start;
            for (uint32_t j = 0; j < m; j++) {
                CSR_dense_row[features.idx[row_start + j]] += adj_weight * features.val[row_start + j]; 
            }
        }
        // Dense row is now complete, convert to CSR format.
        new_features.ptr[base_node] = nf_count;
        for (uint32_t j = 0; j < FEATURE_DEPTH; j++) {
            if (CSR_dense_row[j] != 0) {
                new_features.val[nf_count] = CSR_dense_row[j];
                new_features.idx[nf_count] = j;
                nf_count ++;
            }
            if (nf_count == nf_size / sizeof(uint32_t)) {
                nf_size *= 2;
                new_features.val = realloc(new_features.val, nf_size);
                new_features.idx = realloc(new_features.idx, nf_size);
            }
        }
        free(CSR_dense_row);
        base_node++;
    }

    new_features.val = realloc(new_features.val, nf_count * sizeof(float));
    new_features.idx = realloc(new_features.idx, nf_count * sizeof(float));
    new_features.val_length = nf_count;
    
    #ifndef ARM_PLATFORM
        write_res("c_precomp_CSR.bin", new_features);
    #endif

    free(features.ptr);
    free(features.val);
    free(features.idx);
    return new_features;
}

// calculates inference for a given node 
void infer (float* features, float* weights, float* biases, float* infered_res, uint32_t n) {
    uint32_t i, j;
    for (i = 0; i < LABELS; i++) {
        infered_res[n*LABELS + i] = biases[i];
        for (j = 0; j < FEATURE_DEPTH; j++) {
            infered_res[n*LABELS + i] += features[n*FEATURE_DEPTH + j] * weights[i*FEATURE_DEPTH + j];
        }
    }
}

// calculates inference for a range of nodes, with features in CSR format
// infered_res must be pre-allocated and zeroed.
void infer_CSR (struct CSR features, float* weights, float* biases, float* infered_res, uint32_t start, uint32_t end) {
    uint32_t node, label, base_pos, nb_features, feat_pos, i;
    for (node = start; node <= end; node++) {
        base_pos = features.ptr[node];
        if (node < GRAPH_SIZE - 1) {
            nb_features = features.ptr[node + 1] - base_pos;
        } else {
            nb_features = features.val_length - base_pos;
        }
        for (label = 0; label < LABELS; label++ ) {
            infered_res[node*LABELS + label] = biases[label];
            for (i = 0; i < nb_features; i++) {
                feat_pos = base_pos + i;
                infered_res[node*LABELS + label] += weights[label*FEATURE_DEPTH + features.idx[feat_pos]] * features.val[feat_pos];
            }
        }
    }
}

void soft_max (float* infered_res, uint32_t start, uint32_t end) {
    uint32_t node, i;
    for (node = start; node <= end; node++) {
        float max = 0, sum = 0;
        for (i = 0; i < LABELS; i++) {
            float temp = infered_res[node*LABELS + i];
            if (temp > max)
                max = temp;
        }
        for (i = 0; i < LABELS; i++) {
            infered_res[node*LABELS + i] -= max;
            sum += exp(infered_res[node*LABELS + i]);
        }
        for (i = 0; i < LABELS; i++) {
            infered_res[node*LABELS + i] = exp(infered_res[node*LABELS + i]) / sum;
        }
    }
}
void soft_max_old (float* vector, uint32_t n) {
    float max = 0, sum = 0;
    uint32_t i;
    for (i = 0; i < LABELS; i++) {
        float temp = vector[n*LABELS + i];
        if (temp > max)
            max = temp;
    }
    for (i = 0; i < LABELS; i++) {
        vector[n*LABELS + i] -= max;
        sum += exp(vector[n*LABELS + i]);
    }
    for (i = 0; i < LABELS; i++) {
        vector[n*LABELS + i] = exp(vector[n*LABELS + i]) / sum;
    }
}

float cross_entropy (float* vector, uint32_t* labels) {
    // cross_entropy = - 1 * log(vector(x)) ; with x the correct label
    float cross_entropy = 0; 
    uint32_t i;
    for (i = TRAIN_START; i <= TRAIN_END; i++) {
        cross_entropy -= log(vector[i*LABELS + labels[i]]);
    }
    return cross_entropy / (TRAIN_END - TRAIN_START +1);
}

// traverse sparse feature matrix only once
// grad and bias_grad arguments must be zeroed before function call!
// gradients [LABELS][FEATURE_DEPTH]
void CSR_gradients (struct CSR features, float* weights, float* biases, float* infered_res, uint32_t* labels, float* grad, float* bias_grad, uint32_t start, uint32_t end) {
    for (uint32_t node = start; node <= end ; node++) {
        uint32_t base_pos, nb_features;
        base_pos = features.ptr[node];
        if (node < GRAPH_SIZE - 1) {
            nb_features = features.ptr[node + 1] - base_pos;
        } else {
            nb_features = features.val_length - base_pos;
        }
        for (uint32_t feat = 0; feat < nb_features; feat++) {
            uint32_t feat_idx = features.idx[base_pos + feat];
            for (uint32_t label = 0; label < LABELS; label++) {
                if (labels[node] == label) {
                    grad[label*FEATURE_DEPTH + feat_idx] += (infered_res[node*LABELS + label] - 1) * features.val[base_pos + feat];
                } else {
                    grad[label*FEATURE_DEPTH + feat_idx] += infered_res[node*LABELS + label] * features.val[base_pos + feat];
                }
            } 
        }
        // Biases
        for (uint32_t label = 0; label < LABELS; label++) {
            if (labels[node] == label) {
                bias_grad[label] += infered_res[node*LABELS + label] - 1;
            } else {
                bias_grad[label] += infered_res[node*LABELS + label];
            }
        }
    }
    for (uint32_t i = 0; i < LABELS * FEATURE_DEPTH; i++) {
        /* if (i < 5) */
        /*     printf("grad %d : %f\n", i, grad[i]); */
        grad[i] += WEIGHT_DECAY * weights[i];
    }
    for (uint32_t label = 0; label < LABELS; label++) {
        bias_grad[label] += WEIGHT_DECAY * biases[label];
    }
}

// OG
// gradients [LABELS][FEATURE_DEPTH]
void gradients (float* features, float* weights, float* biases, float* infered_res, uint32_t* labels, float* grad, float* bias_grad){
    for (uint32_t i = 0; i < LABELS; i++) {
        for (uint32_t j = 0; j < FEATURE_DEPTH; j++) {
            float sum_grad = 0, sum_bias = 0;
            for (uint32_t b = TRAIN_START; b <= TRAIN_END; b++) {
                if (labels[b] == i) {
                    sum_grad += (infered_res[b*LABELS + i] - 1) * features[b*FEATURE_DEPTH + j];
                    if (!j) // Do not iterate over features
                        sum_bias += infered_res[b*LABELS + i] - 1;
                } else {
                    sum_grad += infered_res[b*LABELS + i] * features[b*FEATURE_DEPTH + j]; 
                    if (!j)
                        sum_bias += infered_res[b*LABELS + i]; 
                }
            }
            grad[i*FEATURE_DEPTH + j] = sum_grad + WEIGHT_DECAY * weights[i*FEATURE_DEPTH + j];
            if (!j)
                bias_grad[i] = sum_bias + WEIGHT_DECAY * biases[i];
        }
    }

}

// TODO implement gradient decay(L2)
void adam (float* grad, float* m, float* v, float* weights, uint32_t t) {
    uint32_t i = 0;
    float weight_sum = 0;
    for (i = 0; i < LABELS*FEATURE_DEPTH; i++) {
        /* grad[i] -= WEIGHT_DECAY * weights[i]; // L2 linearization. Doesn't seem to work */
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1 - BETA2) * pow(grad[i], 2.0);
    }
    float* m_ = malloc(LABELS*FEATURE_DEPTH*sizeof(float));
    float* v_ = malloc(LABELS*FEATURE_DEPTH*sizeof(float));
    for (i = 0; i < LABELS*FEATURE_DEPTH; i++) {
        m_[i] = m[i] / (1- pow(BETA1, t));
        v_[i] = v[i] / (1- pow(BETA2, t));
        weights[i] -= (ALPHA * m_[i] / (sqrt(v_[i]) + EPSILON));
        weight_sum += weights[i];
    }
    /* printf("Tot weight @%3d:%f\n", t, weight_sum); */
}

// Note: This function could be removed by extending the weight matrix by LABELS
//       to include biases. Adam optimization would then be automatic, but gradient
//       calculation and inference would need to be modified accordingly
void adam_biases (float* grad_b, float* m_b, float* v_b, float* biases, uint32_t t) {
    uint32_t i = 0;
    for (i = 0; i < LABELS; i++) {
        /* biases[i] += WEIGHT_DECAY * weights[i]; // L2 linearization. Doesn't seem to work */
        m_b[i] = BETA1 * m_b[i] + (1 - BETA1) * grad_b[i];
        v_b[i] = BETA2 * v_b[i] + (1 - BETA2) * pow(grad_b[i], 2.0);
    }
    float* m_b_ = malloc(LABELS*sizeof(float));
    float* v_b_ = malloc(LABELS*sizeof(float));
    for (i = 0; i < LABELS; i++) {
        m_b_[i] = m_b[i] / (1- pow(BETA1, t));
        v_b_[i] = v_b[i] / (1- pow(BETA2, t));
        biases[i] -= (ALPHA * m_b_[i] / (sqrt(v_b_[i]) + EPSILON));
    }
}

// prediction[GRAPH_SIZE][LABELS]  truth[GRAPH_SIZE]
float accuracy (float* prediction, uint32_t* truth, uint32_t start_idx, uint32_t stop_idx) {
    uint32_t i, j, max_idx, sum = 0;
    float max;
    for (i = start_idx; i <= stop_idx; i++ ) {
        max = 0;
        for (j = 0; j < LABELS; j++) {
            if (prediction[i * LABELS + j] > max) {
                max = prediction[i * LABELS + j];
                max_idx = j;
            }
        }
        if (max_idx == truth[i])
            sum ++;
    }
    return ((float)sum / (stop_idx - start_idx + 1));
}

int main() {
    clock_t begin = clock();
    
    uint32_t * adj = read_file("adj.bin");
    float * adj_weights = read_float_file("adj_weights.bin"); // Should actually be dataUnion type
    float * features = read_float_file("features.bin");
    struct CSR CSR_features = read_CSR("CSR_values.bin", "CSR_Idx.bin", "CSR_Ptr.bin");
    uint32_t * labels = read_file("labels.bin");
    float * degree = generate_degree_matrix(adj);
    dataUnion * s = generate_normalised_adj_matrix(adj, adj_weights, degree);
    for (uint32_t i = 0; i < DEGREE; i++) {
        features = sparse_matrix_mult(s, features);
        CSR_features = sparse_matrix_mult_CSR(s, CSR_features);
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Precomputation time: %lf\n", time_spent);

    /* print_matrix_to_file("preprocess.bin", GRAPH_SIZE, FEATURE_DEPTH, FEAT); */
    
    begin = clock();
    // Weights are transposed from python: LABELS x FEATURE_DEPTH
    float * weights = read_float_file("python_starting_weights.bin"); // transposed
    float * biases = read_float_file("python_starting_biases.bin");
    float * infered_res = calloc(LABELS * GRAPH_SIZE, sizeof(float));
    float * infered_res_CSR = calloc(LABELS * GRAPH_SIZE, sizeof(float));
    // TODO create randomised training sub-goup
    float * m = calloc(LABELS * FEATURE_DEPTH, sizeof(float));
    float * v = calloc(LABELS * FEATURE_DEPTH, sizeof(float));
    float * m_b = calloc(LABELS, sizeof(float));
    float * v_b = calloc(LABELS, sizeof(float));
    /* for (uint32_t i = 0; i < 30; i++) { */
    /*     printf("row %2d", i); */
    /*     for (uint32_t j = 0; j < LABELS; j++) { */
    /*         printf(" %f ", infered_res[LABELS*i + j]); */
    /*     } */
    /*     printf("\n"); */
    /* } */

    printf("starting biases: ");
    for (uint32_t i = 0; i < LABELS; i++){
        printf("%f ", biases[i]);
    }
    printf("\n");

    infer_CSR (CSR_features, weights, biases, infered_res_CSR, TRAIN_START, TRAIN_END);
    soft_max(infered_res_CSR, TRAIN_START, TRAIN_END);
    for (uint32_t i = TRAIN_START; i <= TRAIN_END; i++) {
        infer(features, weights, biases, infered_res, i);
        soft_max_old(infered_res, i);
    }

    FILE *f = fopen("infered_res_CSR.bin", "wb");
    FILE *f2 = fopen("infered_res.bin", "wb");
    if (f == NULL || f2 == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    for(uint32_t i = 0; i < LABELS * GRAPH_SIZE; i++) {
        fwrite(&infered_res_CSR[i], sizeof(float), 1, f);
        fwrite(&infered_res[i], sizeof(float), 1, f2);
    }
    fclose(f);
    fclose(f2);


    for (uint32_t epoch = 1; epoch < EPOCHS+1; epoch++) {
        infer_CSR (CSR_features, weights, biases, infered_res_CSR, TRAIN_START, TRAIN_END);
        soft_max(infered_res_CSR, TRAIN_START, TRAIN_END);
        /* for (uint32_t i = TRAIN_START; i <= TRAIN_END; i++) { */
        /*     infer(features, weights, biases, infered_res, i); */
        /*     soft_max_old(infered_res, i); */
        /* } */
        if (epoch == 1 || epoch == EPOCHS)
            printf("Cross entropy %d: %f \n", epoch, cross_entropy(infered_res_CSR, labels));
        float * grad = calloc(LABELS * FEATURE_DEPTH, sizeof(float));
        float * bias_grad = calloc(LABELS, sizeof(float));
        /* gradients (features, weights, biases, infered_res, labels, grad, bias_grad); */
        CSR_gradients (CSR_features, weights, biases, infered_res_CSR, labels, grad, bias_grad, TRAIN_START, TRAIN_END);
        printf("grad 30: %f weight 30: %f\n", grad[30], weights[30]);
        adam(grad, m, v, weights, epoch);
        adam_biases(bias_grad, m_b, v_b, biases, epoch);
        free(grad);
        free(bias_grad);
    }


    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Inference time: %lf\n", time_spent);

    for (uint32_t i = VAL_START; i <= VAL_END; i++) {
        infer(features, weights, biases, infered_res, i);
        soft_max_old(infered_res, i);
    }
    printf("Validation ACCURACY: %f\n", accuracy(infered_res, labels, VAL_START, VAL_END));

    for (uint32_t i = TEST_START; i <= TEST_END; i++) {
        infer(features, weights, biases, infered_res, i);
        soft_max_old(infered_res, i);
    }
    printf("Test ACCURACY: %f\n", accuracy(infered_res, labels, TEST_START, TEST_END));

    printf("ending biases: ");
    for (uint32_t i = 0; i < LABELS; i++){
        printf("%f ", biases[i]);
    }
    printf("\n");

    free(adj);
    free(degree);
    free(s);
    free(features);
    free(infered_res);
    free(infered_res_CSR);
    return 0;
}
