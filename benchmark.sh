#!/bin/bash

PAR_EXEC=./lu-omp
SER_EXEC=./lu-omp-serial
MATRIX_SIZES=(1000 2000)
THREADS=(8 12)
REPEAT=4

echo "MatrixSize,Threads,Type,Time(s),Residual,VerificationTime(s)" > results.csv

for size in "${MATRIX_SIZES[@]}"; do

  # #### Serial
  # for ((i=0; i<REPEAT; i++)); do
  #   output=$($SER_EXEC $size 1)

  #   time=$(echo "$output" | grep "LU decomposition time" | awk '{print $4}')
  #   res=$(echo "$output" | grep "L2,1 norm of residual" | awk '{print $5}')
  #   vtime=$(echo "$output" | grep "Verification time" | awk '{print $3}')

  #   echo "$size,1,Serial,$time,$res,$vtime" | tee -a results.csv
  # done

  #### Parallel
  for t in "${THREADS[@]}"; do
    for ((i=0; i<REPEAT; i++)); do
      output=$($PAR_EXEC $size $t)

      time=$(echo "$output" | grep "LU decomposition time" | awk '{print $4}')
      res=$(echo "$output" | grep "L2,1 norm of residual" | awk '{print $5}')
      vtime=$(echo "$output" | grep "Verification time" | awk '{print $3}')

      echo "$size,$t,Parallel,$time,$res,$vtime" | tee -a results.csv
    done
  done

done
