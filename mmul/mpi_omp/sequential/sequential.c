
#include <omp.h>
#include <math.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdio.h>
#include <unistd.h>
#include <time.h>

float *A, *B, *C;

void alloc_arrays(int x, int y, int z) {
    // A[X, Y], B[Y, Z], C[X, Z]

    A = (float*)malloc(x*y*sizeof(float));
    B = (float*)malloc(y*z*sizeof(float));
    C = (float*)malloc(x*z*sizeof(float));
}

void init_data_arrays(int x, int y, int z) {
    // A[X, Y], B[Y, Z], C[X, Z]

    int i, j;
    
    // Setting the seed for random number generation
    srandom(time(NULL));

    // A is declared regularly
    for(i = 0; i < x; i++)
        for(j = 0; j < y; j++)
            A[i*y+j] = (float)rand() / (float)RAND_MAX; 

    // B is transposed on declaration
    for(i = 0; i < y; i++)
        for(j = 0; j < z; j++)
            B[j*y+i]=(float)rand() / (float)RAND_MAX; 
}

int main(int argc, char *argv[]) {

    // Time measuring variables 
    clock_t start, end;
    double total;

    // User arguments 
    int x, y, z;

    // Loop control variables
    int i, j, k;

    if(argc != 4){
        printf("Usage: ./mmul x y z\n"
               "A[x, y] * B[y, z] = C[x, z]\n");
        exit(-1);
    }

    x = atoi(argv[1]); y = atoi(argv[2]); z = atoi(argv[3]);
    
    start = clock();

    alloc_arrays(x, y, z);
    init_data_arrays(x, y, z);

    for(i = 0; i < x; i++) {
        for(j = 0; j < z; j++) {
            C[i*z+j] = 0;
            for(k = 0; k < y; k++)  {
                C[i*z+j] += A[i*y+k] * B[j*y+k];
            }
        }
    }

    end = clock();

    total = (double)(end - start) / CLOCKS_PER_SEC;;

    printf("(s_mmul) Execution time for A[%d][%d] x B[%d][%d] = C[%d][%d] is %.6lf\n",
                x, y, y, z, x, z, total);

    free(A); free(B); free(C);

    return 0;
}
