/*
    This program solves Laplace's equation on a regular 2D grid using simple Jacobi iteration.

    The stencil calculation stops when  ( err >= CONV_THRESHOLD OR  iter > ITER_MAX )
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <mpi.h>

#define ITER_MAX 3000 // number of maximum iterations
#define CONV_THRESHOLD 1.0e-5f // threshold of convergence

struct custom_wavefield {
    double* top_halo,
          * bottom_halo,
          * interior;
};

// matrix to be solved
double *grid;

double *new_grid;
// auxiliary matrix

// size of each side of the grid
int size;

// return the maximum value
double max(double a, double b){
    if(a > b)
        return a;
    return b;
}

// return the absolute value of a number
double absolute(double num){
    if(num < 0)
        return -1.0 * num;
    return num;
}

void allocate_memory(){
    // allocate memory for the grid
    grid = (double *) malloc(size * size * sizeof(double));
    // new_grid = (double *) malloc(size * size * sizeof(double));
}

// initialize the grid
void initialize_grid(){
    // seed for random generator
    srand(10);

    int linf = size / 2;
    int lsup = linf + size / 10;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // inicializa região de calor no centro do grid
            if ( i>=linf && i < lsup && j>=linf && j<lsup)
                grid[i*size+j] = 100;
            else
               grid[i*size+j] = 0;
            // new_grid[i*size+j] = 0.0;
            // std::cout << grid[i*size+j] << " ";
        }
        // std::cout << std::endl;
    }
}

// save the grid in a file
void save_grid(){

    char file_name[30];
    sprintf(file_name, "grid_laplace.txt");

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            fprintf(file, "%lf ", grid[i*size+j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void print_grid(int size) {
    int i, j;

    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            printf("| %.1lf |", grid[i*size+j]);
        }
        printf("\n");
    }
}

// perform one Jacobi sweep on the local subgrid and return maximum error
double jacobi_substep(double *sub_grid, double *sub_new_grid, int size, int sub_size, int i_offset) {
    double err = 0.0;
    for (int i = i_offset; i < i_offset + sub_size; i++) {  
        for(int j = 1; j < size-1; j++) {
            sub_new_grid[i*size+j] = 0.25 * (sub_grid[i*size+j+1] + sub_grid[i*size+j-1] +
                                             sub_grid[(i-1)*size+j] + sub_grid[(i+1)*size+j]);
            err = max(err, absolute(sub_new_grid[i*size+j] - sub_grid[i*size+j]));
        }
    }
    return err;
}

int main(int argc, char *argv[]){

    if(argc != 2){
        printf("Usage: ./laplace_seq N\n");
        printf("N: The size of each side of the domain (grid)\n");
        exit(-1);
    }

    // variables to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    size = 1 << (atoi(argv[1]));    
    int stencil_radius = 1;

    // mpi initialization
    MPI_Init(&argc, &argv);

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // resize the grid to be divisible by the number of processes
    int slice[mpi_size],
        displs[mpi_size],
        base_slice = size / mpi_size,
        remainder = size % mpi_size,
        offset = 0;

    for(int i = 0; i < mpi_size; i++) {
        slice[i] = (base_slice + (i < remainder ? 1 : 0)) * size;
        displs[i] = offset;
        offset += slice[i];
        
        if(rank == 0)
            printf("Process %d will handle %d columns, displacement: %d\n", 
                   i, slice[i], displs[i]);
    }

    // // Calculate this process's column offset in the global grid
    int my_offset = 0;
    for(int i = 0; i < rank; i++) 
        my_offset += slice[i];

    if(rank == 0) {
        printf("Grid size: %d x %d\n", size, size);
        // allocate memory to the grid (matrix)
        grid = (double *) malloc(size * size * sizeof(double));

        // set grid initial conditions
        initialize_grid();
    }

    int domain = slice[rank];           // number of owned elements
    int halo = stencil_radius * size; // elements per halo
    int total = domain + 2*halo;

    // // Allocate memory for local slice + 2 halo buffers
    double *sub_grid = (double *) malloc(total * sizeof(double));
    double *sub_new_grid = (double *) malloc(total * sizeof(double));

    // Initializing custom datastructure
    custom_wavefield cw_subgrid = {
        &sub_grid[0],                               // top_halo
        &sub_grid[total - halo],        // bottom_halo
        &sub_grid[halo]                       // interior
    };

    custom_wavefield cw_new_subgrid = {
        &sub_new_grid[0],
        &sub_new_grid[total - halo],
        &sub_new_grid[halo]
    };

    // Fill new_sub_grid with zeros
    for(int i = 0; i < total; i++)
        sub_new_grid[i] = 0.0;

    // // Scatter: skip first ghost column (start at index 'size')
    MPI_Scatterv(rank == 0 ? grid : NULL, slice, displs, 
                MPI_DOUBLE, cw_subgrid.interior, domain, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(0)
        for(int i = 0; i < domain; i++)
            printf("(P%d) sub_grid[%d] = %lf\n", rank, i, cw_subgrid.interior[i]);

    // Freeing the original grid from P0 memory
    if(rank == 0) free(grid);

    double err_interior = 0.0,
           err_top = 0.0,
           err_bottom = 0.0,
           err = 1.0;
    int iter = 0;
    double *temp;

    printf("Jacobi relaxation calculation: %d x %d subgrid (Process %d)\n", size, size, rank);

    // get the start time
    gettimeofday(&time_start, NULL);

    // Jacobi iteration
    // This loop will end if either the maximum change reaches below a set threshold (convergence)
    // or a fixed number of maximum iterations have completed
    while ( err > CONV_THRESHOLD && iter <= ITER_MAX ) {

        MPI_Request send_requests[2];
        int num_requests = 0;

        // calculates the Laplace equation to determine each cell's next value
        err_interior = jacobi_substep(sub_grid, sub_new_grid, size, slice[rank]/size-2*stencil_radius, 2*stencil_radius);

        // Sending and receiving from top neighbor
        if(rank != 0) {
            // Send first interior row to top neighbor (rank - 1)
            MPI_Isend(cw_subgrid.interior, halo, MPI_DOUBLE, rank - 1, 0, 
                    MPI_COMM_WORLD, &send_requests[num_requests++]);
            
            // Receive into top halo from top neighbor
            MPI_Recv(cw_subgrid.top_halo, halo, MPI_DOUBLE, rank - 1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Process top boundary rows (that need the top halo)
            err_top = jacobi_substep(sub_grid, sub_new_grid, size, stencil_radius, stencil_radius);
        }

        // Sending and receiving from bottom neighbor
        if(rank != mpi_size - 1) {
            // Send last interior row to bottom neighbor
            int last_interior_offset = domain;  // Start of last interior row
            MPI_Isend(&sub_grid[last_interior_offset], halo, MPI_DOUBLE, rank + 1, 0, 
                    MPI_COMM_WORLD, &send_requests[num_requests++]);
            
            // Receive into bottom halo
            MPI_Recv(cw_subgrid.bottom_halo, halo, MPI_DOUBLE, rank + 1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Process the last stencil_radius interior rows (which need bottom halo)
            int total_interior_rows = domain / size;
            int bottom_start = stencil_radius + total_interior_rows - stencil_radius;
            err_bottom = jacobi_substep(sub_grid, sub_new_grid, size, stencil_radius, bottom_start);
        }

        printf("[Rank %d] {err_bottom: %lf, err_interior: %lf, err_top: %lf}\n", 
               rank, err_bottom,  err_interior, err_top );

        // Getting the maximum error among the entire subdomain
        err = max(err_interior, max(err_top, err_bottom));

        // // copie the next values into the working array for the next iteration
        
        custom_wavefield temp_cw = cw_subgrid;
        cw_subgrid = cw_new_subgrid;
        cw_new_subgrid = temp_cw;
        
        temp = sub_grid;
        sub_grid = sub_new_grid;
        sub_new_grid = temp;

        iter++;

        double global_err;
        MPI_Allreduce(&err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        err = global_err;

        printf("Error: %lf\n", err);
    }

    MPI_Finalize();

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                       (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    // //save the final grid in file
    // save_grid();

    free(sub_grid);
    free(sub_new_grid);

    printf("\nKernel executed in %lf seconds with %d iterations and error of %0.10lf\n", exec_time, iter, err);

    return 0;
}
