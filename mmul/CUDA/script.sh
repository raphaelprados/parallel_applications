#!/bin/bash

nvcc -arch=sm_70 -o b_mmul 10-mmul/b_mmul.cu

sizes=(1000 2500 5000 10000 15000 17000)
folds=(0 1 2 3 4 5)

for size in "${sizes[@]}"; do
    for fold in "${folds[@]}"; do
        ./b_mmul "$size" "$fold"
    done
done
