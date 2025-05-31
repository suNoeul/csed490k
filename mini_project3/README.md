# Mini Project 3: Conway’s Game of Life in MPI

This project implements **Conway’s Game of Life** in parallel using **MPI (Message Passing Interface)**. 
It includes correctness checking using sample input/output files, as well as performance measurement for various process counts.

### Environment
- **Platform**: WSL2 Ubuntu 22.04  
- **Compiler**: g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0  
- **MPI Library**: Open MPI 4.1.2


### Open MPI Installation

To install Open MPI, run the following commands:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```
Verify the installation:

```bash
mpirun --version    # e.g. mpirun (Open MPI) 4.1.2
which mpicc         # e.g. /usr/bin/mpicc
which mpirun        # e.g. /usr/bin/mpirun
```



<br>

g++ -O2 -std=c++17 -o project3-serial project3-serial.cpp
i=1; ./project3-serial < ./sample/input${i}.txt > ./sample/serial_output${i}.txt
i=1; diff -bwi ./sample/output${i}.txt ./sample/serial_output${i}.txt

## 🚀 How to Compile and Run
```bash
# 1. Compile the MPI-based Game of Life & Serial Baseline version
mpiCC -o project3 project3.cpp
g++ -O2 -std=c++17 -o project3-serial project3-serial.cpp 

# 2. Run a Sample Input with MPI (e.g., 4 processes)
i=1; mpirun -np 4 ./project3 < ./sample/input${i}.txt > ./sample/my_output${i}.txt

# 3. Verify Output Correctness
i=1; diff -bwi ./sample/output${i}.txt ./sample/my_output${i}.txt  
```
If the files are identical, `diff` will return no output (indicating a correct result).


<br>



## 📊 Performance Measurement Script
⚠️ **Note:** This script assumes that both `project3` (MPI version) and `project3-serial` (serial baseline) executables **already exist** in the same directory. Make sure you compile them before running the script.

To benchmark execution time across multiple process counts (`0, 1, 2, 4, 8`), use the provided shell script:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This script performs the following:
- Executes:
    - `project3-serial` when `n=0` (serves as the baseline serial version)
    - `project3` with `mpirun -np n` when `n > 0` (parallel MPI version)
- Runs across inputs `input1.txt` to `input5.txt`
- Measures and logs execution time
- Compares each output with the ground truth
- Saves all results to a CSV file named `results.csv`

