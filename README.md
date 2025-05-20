# PPD-UFSCar-2025.1

## Laplace Heat Dispersion Parallelization

- This application was developed as a requisite for two class assignments ministered by Ph.D. Hermes Senger. 
- There are currently OpenMP and PThreads implementations, both using the C language. 
- Future attempts include porting this program to CUDA and also checking the performance difference between C and C++.  
- Current implementations have three different strategies (continuous workload balancing, round-robin and a pseudo-consumer/producer scheme). For each of these, there are 4 variations: the regular implementation (no additional optimizations), matrix to vector linearization (MtVL), pointer-swapping (PS) and MtVL-PS (which uses both methods). 
- Tests were run using Singularity containers in the UFSCar Cluster, using from 20 to 120 threads. The Singularity container.def file, as well as the Makefiles and jobs for Slurm are all included in this repository.   