CC = gcc
CFLAGS = -fopenmp -g -Wall -Wextra -O3
CFLAG = -g -Wall -Wextra -O3
MPI = mpicc

all: clean new serial openmp mpi_openmp 

new:
	@mkdir bin

serial:

	@$(CC) sequential/sequential.c $(CFLAG) -o s_mmul
	@mv s_mmul bin

openmp:

	@$(CC) omp_only/omp.c $(CFLAGS) -o o_mmul_omp
	@mv o_mmul_omp bin

mpi_openmp:

	@cd mpi_and_omp && $(MPI) regular.c $(CFLAGS) -o r_mmul_mpi_omp && $(MPI) demand.c $(CFLAGS) -o d_mmul_mpi_omp && $(MPI) round_robin.c $(CFLAGS) -o rr_mmul_mpi_omp
	@cd mpi_and_omp && mv r_* rr_* d_* ../bin

clean:
	@find . -type f -executable -exec rm -f {} +
	@rm -rf bin
	