# PPD-UFSCar-2025.1

## Laplace Heat Dispersion Parallelization (laplace)

- This application was developed as a requisite for two class assignments ministered by Ph.D. Hermes Senger. 
- There are currently OpenMP and PThreads implementations, both using the C language. 
- Future attempts include porting this program to CUDA and also checking the performance difference between C and C++.  
- Current implementations have three different strategies (continuous workload balancing, round-robin and a pseudo-consumer/producer scheme). For each of these, there are 4 variations: the regular implementation (no additional optimizations), matrix to vector linearization (MtVL), pointer-swapping (PS) and MtVL-PS (which uses both methods). 
- Tests were run using Singularity containers in the UFSCar Cluster, using from 20 to 120 threads. The Singularity container.def file, as well as the Makefiles and jobs for Slurm are all included in this repository.   

## Matrix Multiplication (mmul)
- This application was also developed as a requisite for two class assignments ministered by Ph.D. Hermes Senger
- The first one solves a matrix multiplication (C = A x B) using OpenMP and MPI focusing on multi-Node decomposition. There's also a MPI only implementation for comparison tests. 
- The second one (which is not completed as of this date) does the same, but using a single GPU and CUDA.  
- Both use a scheme where the C matrix is not computed as A x B, but as A x Bt, where Bt is the transpose of B. This can render better results, as we don't have to multiply row by column, but row by row, which increases memory locality. 
- Tests are still being conducted in the UFSCar cluster for the OpenMP + MPI implementation using 2 nodes and 20 to 120 Threads in each node for a 10k dimension square matrix. The CUDA tests will run using WSL 2 (Ubuntu) with CUDA enabled for a GTX 1060 6GB.  

