#include "data/adj.h"
#include "data/feat.h"
#include "data/idx_test.h"
#include "data/idx_train.h"
#include "data/idx_val.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("in main\n");
    float *new_features = malloc(2708*1433*sizeof(float));
    int d, i, j, k;
    printf("var init\n");
     for(d = 0; d < 2; d++){
         printf("Degree: %d\n", d);
         for(i = 0; i < 2708; i++) {
             printf("Outer: %d\n", i);
             for(j = 0; j < 1433; j++) {
                 //printf("Inner: %d\n", j);
                 new_features[i*1433 +j] = 0;
                 for(k = 0; k < 2708; k++) {
                     new_features[i*1433 + j] += ADJ[i][k] * FEAT[k][j];
                }
            }
        }
    }
    free(new_features);
    return 0;
}

