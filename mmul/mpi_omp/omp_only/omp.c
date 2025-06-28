
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

    // User arguments 
    int x, y, z;

    // Loop control variables
    int i, j, k;

    // Execution time variables
    clock_t start, end;
    double total;

    if(argc != 5){
        printf("Usage: ./mmul x y z T\n"
               "A[x, y] * B[y, z] = C[x, z]\n"
               "T is the number of OpenMP Threads");
        exit(-1);
    }

    x = atoi(argv[1]); y = atoi(argv[2]); z = atoi(argv[3]);
    
    omp_set_num_threads(atoi(argv[4]));

    start = clock();

    alloc_arrays(x, y, z);
    init_data_arrays(x, y, z);

    double sum;
    #pragma omp parallel for schedule(static) private(j, k, sum)
    for (i = 0; i < x; i++) {
        for (j = 0; j < z; j++) {
            sum = 0.0;
            for (k = 0; k < y; k++) {
                sum += A[i*y + k] * B[j*y + k];
            }
            C[i*z + j] = sum;
        }
    }

    end = clock();

    total = (double)(end - start) / CLOCKS_PER_SEC;;

    printf("(o_omp_mmul) Execution time for A[%d][%d] x B[%d][%d] = C[%d][%d] using %d OpenMP threads is %.6lf\n",
                x, y, y, z, x, z, atoi(argv[4]), total);

    return 0;
}
