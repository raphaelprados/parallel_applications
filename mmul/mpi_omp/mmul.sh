#!/bin/bash
#SBATCH -J mmul_mpi_omp             # Job name
#SBATCH -p slow                     # Job partition
#SBATCH -t 12:00:00                 # Run time (hh:mm:ss)
#SBATCH --mem=8G                    # Memory requirements
#SBATCH --nodes=2                   # Number of nodes needed
#SBATCH --ntasks=2                  # Number of total MPI processes
#SBATCH --cpus-per-task=40          # Number of OpenMP threads per process
#SBATCH --output=%x.%j.out          # Name of stdout output file - %j expands to jobId and %x to jobName
#SBATCH --error=%x.%j.err           # Name of stderr output file

sizes=(1000 2500 5000 10000 15000)
runs=(10 20 30 40)
schedulers=("d" "r" "rr")

echo "***************** SEQUENTIAL *****************"

echo "Running size=${size} run=${run} sched=${sched}"       
for size in "${sizes[@]}"; do
    singularity run mmul.sif s_mmul "$size" "$size" "$size"
done

echo "***************** PARALLEL (OMP) *****************"

echo "Running size=${size} run=${run} sched=${sched}"       
for size in "${sizes[@]}"; do
    for run in "${runs[@]}"; do
        singularity run mmul.sif o_mmul_omp "$size" "$size" "$size" "$run"
    done
done

echo "***************** PARALLEL (MPI+OMP) *****************"

for sched in "${schedulers[@]}"; do
    echo "Running size=${size} run=${run} sched=${sched}"       
    for size in "${sizes[@]}"; do
        for run in "${runs[@]}"; do
            srun --mpi=pmi2 singularity run mmul.sif ${sched}_mmul_mpi_omp "$size" "$size" "$size" "$run"
        done
    done
done

# OBS: if it is an MPI job
# use --mpi=pmi2
# srun --mpi=pmi2 singularity run container.sif mmul_seq 1000
