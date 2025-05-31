#!/bin/bash

# Navigate to the directory of the current script
ROOT="$(dirname "$0")"
cd "$ROOT" || exit 1

OUTPUT_CSV="results.csv"
echo "n_process,input_file,time_sec,validation" > "$OUTPUT_CSV"

for n in 1 2 4 8 16; do
  for i in {1..5}; do
    input_file="./sample/input${i}.txt"
    output_file="./sample/my_output${i}.txt"
    gt_file="./sample/output${i}.txt"

    # Start measuring time
    start_time=$(date +%s.%N)

    # Execute the program
    mpirun --oversubscribe -np "$n" ./project3 < "$input_file" > "$output_file"

    # End measuring time
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    # Compare the result with the ground truth
    if diff -bwi "$gt_file" "$output_file" > /dev/null; then
      result="PASS"
    else
      result="FAIL"
    fi

    # Save to CSV
    echo "$n,input${i}.txt,$elapsed,$result" >> "$OUTPUT_CSV"
    echo "[$n proc | input${i}] Time: $elapsed sec | Result: $result"
  done
done
