CC = gcc
CFLAGS = -fopenmp

all: clean openmp

openmp:

	@mkdir bin

	@cd regular && \
	$(CC) $(CFLAGS) omp.c -o r_laplace && \
	$(CC) $(CFLAGS) omp_lin.c -o r_laplace_lin && \
	$(CC) $(CFLAGS) omp_ptr.c -o r_laplace_ptr && \
	$(CC) $(CFLAGS) omp_linptr.c -o r_laplace_linptr
	@mv regular/r_* bin

	@cd demand && \
	$(CC) $(CFLAGS) omp.c -o d_laplace && \
	$(CC) $(CFLAGS) omp_lin.c -o d_laplace_lin && \
	$(CC) $(CFLAGS) omp_ptr.c -o d_laplace_ptr && \
	$(CC) $(CFLAGS) omp_linptr.c -o d_laplace_linptr
	@mv demand/d_* bin

	@cd round_robin && \
	$(CC) $(CFLAGS) omp.c -o rr_laplace && \
	$(CC) $(CFLAGS) omp_lin.c -o rr_laplace_lin && \
	$(CC) $(CFLAGS) omp_ptr.c -o rr_laplace_ptr && \
	$(CC) $(CFLAGS) omp_linptr.c -o rr_laplace_linptr
	@mv round_robin/rr_* bin

clean:
	@find . -type f -executable -exec rm -f {} +
	@rm -rf bin
