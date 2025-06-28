
#include "mpi.h"
#include <omp.h>
#include <math.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdio.h>
#include <unistd.h>
#include <time.h>

float *A, *local_A, *B, *local_C, *C;

void alloc_arrays(int x, int y, int z) {
    // A[X, Y], B[Y, Z], C[X, Z]

    local_A = (float*)malloc(x*y*sizeof(float));
    B       = (float*)malloc(y*z*sizeof(float));
    local_C = (float*)malloc(x*z*sizeof(float));
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
    
    // MPI control variables
    int rank, size, res;
    int *size_vector;

    // User arguments 
    int x, y, z, chunk_size = 1;

    // Loop control variables
    int i, j, k;

    // Matrix operation variables
    int rows_per_task, remainder, counter = 0;

    // Execution time variables
    double start, end, local_total, total;

    if(argc != 5){
        printf("Usage: ./mmul x y z T\n"
               "A[x, y] * B[y, z] = C[x, z]\n"
               "T is the number of OpenMP Threads\n");
        exit(-1);
    }

    //"S is the number of rows each OpenMP thread computes\n"

    x = atoi(argv[1]); y = atoi(argv[2]); z = atoi(argv[3]);

    omp_set_num_threads(atoi(argv[4]));

    // --------------------------------- MPI Initialization ---------------------------------

    res = MPI_Init(&argc, &argv);       
    
    if(res != MPI_SUCCESS) {
        fprintf(stderr, "Erro inciaindo MPI: %d\n", res);
        MPI_Abort(MPI_COMM_WORLD, res);
    }

    start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

        // --------------------------------- Variable Initialization ---------------------------------  
        
        size_vector = (int*)malloc(size*sizeof(int));

        // Divides the rows between the processes            
        if(rank == 0) {

            rows_per_task = x / size;
            remainder = x % size;

            // Assigns how many rows each process takes
            for (i = 0; i < size; i++) {
                size_vector[i] = rows_per_task;
                if (i < remainder)
                    size_vector[i] += 1;
                printf("size_vector[%d] = %d\n", i, size_vector[i]);
            }
        }

        // Broadcasts the workload vector between the processes
        MPI_Bcast(size_vector, size, MPI_INT, 0, MPI_COMM_WORLD);

        // A matrix allocation and A and B Matrices initialization
        if (rank == 0) {
            A = (float*)malloc(x*y*sizeof(float));
            B = (float*)malloc(y*z*sizeof(float));
            C = (float*)malloc(x*z*sizeof(float));

            local_A = A;
            local_C = C;

            init_data_arrays(x, y, z);
        } else {
            // Computation processes matrix allocation
            alloc_arrays(size_vector[rank], y, z);    
        }                

        // --------------------------------- A and B matrices distribution ---------------------------------

    
        if(rank == 0) {
            
            // Sends the substrix to the Ith process. Counter is a variable that acumulates
            //      offset of each submatrix iteratively, as some configurations may have 
            //      unbalanced work division
            for(i = 1; i < size; i++) { 
                
                MPI_Send(&A[counter*y], size_vector[i]*y, MPI_FLOAT, i, 0, MPI_COMM_WORLD);

                counter += size_vector[i];
            }
        } else {
            MPI_Recv(local_A, size_vector[rank]*y, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Bcast(B, y*z, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // --------------------------------- Submatrix Multiplication ---------------------------------

        // Instead of doing local_A x B, I'm doing local_A x Bt (where Bt is B transposed). So instead of 
        //      multiplying a row by a column, I'm multiplying a row in local_A by a row in Bt, which
        //      renders continuous memory access, instead of doing a offset to access each column in B. 
        //      This increases data locality, cache hits and improves performance.   
        #pragma omp parallel for collapse(2) schedule(static, chunk_size) 
        for(i = 0; i < size_vector[rank]; i++) {
            for(j = 0; j < z; j++) {
                local_C[i*z+j] = 0;
                for(k = 0; k < y; k++)  {
                    local_C[i*z+j] += local_A[i*y+k] * B[j*y+k];
                }
            }
        }

        // --------------------------------- Combining the Submatrices into a final Matrix ---------------------------------

        if(rank == 0) {
            // The first submatrix, from rank 0, is already stored in C, so the next one is selected.
            counter = size_vector[0];

            // Gets the local_C from all the other processes
            for(i = 1; i < size; i++) {
                MPI_Recv(&C[counter*z], size_vector[i]*z, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                counter += size_vector[i];
            }
        } else {
            MPI_Send(local_C, size_vector[rank]*z, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // --------------------------------- Results ---------------------------------

        if(rank == 0 && 0) {
            for(i = 0; i < x; i++){
                for(j = 0; j < z; j++) {
                    printf("%.2f ", C[i*z+j]);
                }
                printf("\n");
            }
        }

        // --------------------------------- Memory management ---------------------------------

        free(B); 
        if(rank == 0) {
            free(A); free(C); local_A = NULL; local_C = NULL;
        } else {
            free(local_A); free(local_C); 
        }
        free(size_vector);

    end = MPI_Wtime();

    local_total = end - start;

    MPI_Reduce(&local_total, &total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if(rank == 0)
        printf("(rr_omp_mpi) Execution time for A[%d][%d] x B[%d][%d] = C[%d][%d], using %d MPI processes and %d OpenMP threads per process is %.6lf\n",
                x, y, y, z, x, z, size, atoi(argv[4]), total);

    return 0;
}


