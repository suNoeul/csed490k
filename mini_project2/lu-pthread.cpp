
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>       // EXIT_FAILURE
#include <unistd.h>     // sysconf()
#include <chrono>     
#include <pthread.h>

using Matrix = std::vector<std::vector<double>>;

struct args_t { 
    // Thread & Matrix info
    int id;             // Thread ID
    int t;              // Total number of threads
    int n;              // Matrix dimension

    // Shared matrices
    Matrix* A;          // Input matrix 
    Matrix* L;          // Lower triangular matrix
    Matrix* U;          // Upper triangular matrix

    // Synchronization primitive
    pthread_barrier_t* barrier;
};

int get_cache_line_size() {
    long size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    return (size == -1) ? 64 : static_cast<int>(size);  // fallback: 64 bytes
}

void* thread_worker(void* arg) {
    args_t* args = static_cast<args_t*>(arg);
    int id = args->id;
    int n = args->n;
    int t = args->t;
    Matrix& A = *(args->A);
    Matrix& L = *(args->L);
    Matrix& U = *(args->U);
    pthread_barrier_t* barrier = args->barrier;

    int cache_line_bytes = get_cache_line_size();
    int block_size = cache_line_bytes / sizeof(double); // e.g. 64byte / 8byte = 8개개

    for (int k = 0; k < n; ++k) {
        pthread_barrier_wait(barrier);  // wait for pivot

        int rows = n - (k + 1);
        int chunk = (rows + t - 1) / t;

        int start = k + 1 + id * chunk;
        int end = std::min(k + 1 + (id + 1) * chunk, n);

        // [1] L[i][k] 계산
        for (int i = start; i < end; ++i) 
            L[i][k] = A[i][k] / A[k][k];

        pthread_barrier_wait(barrier);  // wait for L update

        // [2] A[i][j] 갱신
        for (int i = start; i < end; ++i) {
            for (int jb = k + 1; jb < n; jb += block_size) {
                int j_end = std::min(jb + block_size, n);
                for (int j = jb; j < j_end; ++j) {
                    A[i][j] -= L[i][k] * U[k][j];
                }
            }
        }

        pthread_barrier_wait(barrier);  // step done
    }

    pthread_exit(nullptr);
}

void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

// Sequential LU decomposition
void lu_decomposition_seq(Matrix& A_input, Matrix& L, Matrix& U, int n) {
    printf("Running sequential version...\n");
    Matrix A = A_input;  

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k < n; ++k) {
        double max_val = std::abs(A[k][k]);
        int k_prime = k;
        
        // 1. Select pivot row 
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                k_prime = i;
            }
        }
        
        // 2. Check for singular matrix
        if (max_val == 0.0) {
            std::cerr << "LU Decomposition Error: Singular matrix!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // 3. Swap rows in A and L
        if (k_prime != k) {
            std::swap(A[k], A[k_prime]);
            std::swap(L[k], L[k_prime]);  
        }

        // 4. Compute U[k][j] 
        for (int j = k; j < n; ++j)
            U[k][j] = A[k][j];

        // 5. Compute L(i,k) = a(i,k) / U(k,k)
        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A[i][k] / A[k][k];
        }

        // 6. Update the submatrix A[i][j]
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    // 7. Set diagonal of L to 1
    for (int i = 0; i < n; ++i)
        L[i][i] = 1.0;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "LU decomposition(SEQ) time: " << elapsed.count() << " seconds\n";     
}

// Parallel LU decomposition using pthreads (stub)
void lu_decomposition_parallel(Matrix& A_input, Matrix& L, Matrix& U, int n, int t) {
    printf("Running Pthread version...\n");

    Matrix A = A_input;
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, t + 1); // t workers + main thread

    std::vector<pthread_t> threads(t);
    std::vector<args_t> thread_args(t);

    for (int i = 0; i < t; ++i) {
        thread_args[i] = {i, t, n, &A, &L, &U, &barrier};
        pthread_create(&threads[i], nullptr, thread_worker, &thread_args[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {        
        double max_val = std::abs(A[k][k]);
        int k_prime = k;

        // 1. Select pivot row (main thread only)
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                k_prime = i;
            }
        }

        if (max_val == 0.0) {
            std::cerr << "LU Decomposition Error: Singular matrix!" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (k_prime != k) {
            std::swap(A[k], A[k_prime]);
            std::swap(L[k], L[k_prime]);
        }

        // 2. U[k][j] 계산 (main thread)
        for (int j = k; j < n; ++j)
            U[k][j] = A[k][j];

        // 3. 스레드에 작업 시작 알림
        pthread_barrier_wait(&barrier);  // L[i][k] 계산 시작
        pthread_barrier_wait(&barrier);  // A[i][j] 계산 시작
        pthread_barrier_wait(&barrier);  // 다음 단계로 이동
    }

    // 4. L 대각선 설정
    for (int i = 0; i < n; ++i)
        L[i][i] = 1.0;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "LU decomposition(Pthread) time: " << elapsed.count() << " seconds\n";

    // join 및 정리
    for (int i = 0; i < t; ++i)
        pthread_join(threads[i], nullptr);

    pthread_barrier_destroy(&barrier);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <n> <r> <t> <p>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int r = atoi(argv[2]);
    int t = atoi(argv[3]);
    int p = atoi(argv[4]);

    srand(r);
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = rand() / (double)RAND_MAX;

#ifdef PARALLEL
    lu_decomposition_parallel(A, L, U, n, t);
#else
    lu_decomposition_seq(A, L, U, n);
#endif

    if (p == 1) {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    return 0;
}

// [전체 구조]
// main thread:
//   for each k in 0..n-1:
//     1. pivot 선택 및 행 교환 (main thread가 담당)
//     2. worker threads에게:
//        - L[i][k] 계산
//        - A[i][j] 업데이트
//     3. barrier로 동기화