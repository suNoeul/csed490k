#### 과제 환경
- WSL2 Ubuntu 22.04 
- g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

#### OpenMPI 설치
```bash 
sudo apt update && sudo apt upgrade -y

sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

mpirun --version    #> mpirun (Open MPI) 4.1.2

which mpicc         #> /usr/bin/mpicc
which mpirun        #> /usr/bin/mpirun
```
```bash
mpiCC -o project3 project3.cpp
mpirun -np 4 ./project3 < ./sample/input1.txt > my_output.txt
```
