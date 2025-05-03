#!/bin/bash

# 기본값: 1000 (입력이 없을 경우)
MATRIX_SIZE=${1:-1000}
# NUM_THREADS=$(nproc)
NUM_THREADS=1

echo "▶ MATRIX_SIZE = $MATRIX_SIZE"
echo "▶ NUM_THREADS = $NUM_THREADS"
echo

echo "=== [OpenMP Parallel] ==="
./lu-omp-parallel $MATRIX_SIZE 1 $NUM_THREADS 0
echo

echo "=== [OpenMP Serial] ==="
./lu-omp-serial $MATRIX_SIZE 1 1 0
echo

echo "=== [Pthread Parallel] ==="
./lu-pthread-parallel $MATRIX_SIZE 1 $NUM_THREADS 0
echo

echo "=== [Pthread Serial] ==="
./lu-pthread-serial $MATRIX_SIZE 1 1 0
echo
