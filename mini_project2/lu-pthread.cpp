
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>     
#include <pthread.h>

using Matrix = std::vector<std::vector<double>>;

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
void lu_decomposition_parallel(Matrix& A, Matrix& L, Matrix& U, int n, int t) {
    printf("Running Pthread version...\n");

    // [전체 구조]
    // main thread:
    //   for each k in 0..n-1:
    //     1. pivot 선택 및 행 교환 (main thread가 담당)
    //     2. worker threads에게:
    //        - L[i][k] 계산
    //        - A[i][j] 업데이트
    //     3. barrier로 동기화

    
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