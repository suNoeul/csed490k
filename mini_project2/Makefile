EXEC_OMP=lu-omp
EXEC_PTHREAD=lu-pthread

OBJ =  $(EXEC_OMP)-parallel $(EXEC_OMP)-serial \
       $(EXEC_PTHREAD)-parallel $(EXEC_PTHREAD)-serial

MATRIX_SIZE=8000
W :=`grep processor /proc/cpuinfo | wc -l`

# flags
OPT=-O2 -g
OMP=-fopenmp
PTHREAD=-pthread

all: $(OBJ)

# =========================
# OpenMP Targets
# =========================

# Parallel (OpenMP enabled)
$(EXEC_OMP)-parallel: $(EXEC_OMP).cpp
	g++ $(OPT) $(OMP) -DPARALLEL -o $@ $< -lrt

# Serial (no -fopenmp, no PARALLEL macro)
$(EXEC_OMP)-serial: $(EXEC_OMP).cpp
	g++ $(OPT) -o $@ $< -lrt

run-omp-parallel: $(EXEC_OMP)-parallel
	./$(EXEC_OMP)-parallel $(MATRIX_SIZE) 1 $(W) 0

run-omp-serial: $(EXEC_OMP)-serial
	./$(EXEC_OMP)-serial $(MATRIX_SIZE) 1 1 0

# =========================
# Pthread Targets
# =========================

# Parallel (Pthreads + macro)
$(EXEC_PTHREAD)-parallel: $(EXEC_PTHREAD).cpp
	g++ $(OPT) $(PTHREAD) -DPARALLEL -o $@ $< -lrt

# Serial (no -pthread, no macro)
$(EXEC_PTHREAD)-serial: $(EXEC_PTHREAD).cpp
	g++ $(OPT) -o $@ $< -lrt

run-pthread-parallel: $(EXEC_PTHREAD)-parallel
	./$(EXEC_PTHREAD)-parallel $(MATRIX_SIZE) 1 $(W) 0

run-pthread-serial: $(EXEC_PTHREAD)-serial
	./$(EXEC_PTHREAD)-serial $(MATRIX_SIZE) 1 1 0

# =========================
# Common
# =========================

clean:
	rm -f $(OBJ) *.o
