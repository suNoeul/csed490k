
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>     // Time measurement
#include <omp.h>


void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

void lu_decomposition_omp(std::vector<std::vector<double>>& A,
                          std::vector<std::vector<double>>& L,
                          std::vector<std::vector<double>>& U,
                          int n, int t) {
    omp_set_num_threads(t);  

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k < n; ++k) {
        // 1. Select pivot row (row with the largest |a[i][k]| below the diagonal)
        double max_val = 0.0;
        int k_prime = k;

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

        // 3. Swap rows if needed (A and L)
        if (k_prime != k) {
            std::swap(A[k], A[k_prime]);
            std::swap(L[k], L[k_prime]);  
            // pivot은 안해도 되나?
        }

        // (+) Compute U[k][j] for j = k to n-1 (current row of U)
        for (int j = k; j < n; ++j)
            U[k][j] = A[k][j];

        // 4. Compute L(i,k) = a(i,k) / U(k,k)
        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A[i][k] / A[k][k];
        }

        // 5. Update the submatrix A[i][j] for i, j > k
        #pragma omp parallel for collapse(2)
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    // 6. Set diagonal of L to 1
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        L[i][i] = 1.0;
 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "LU decomposition time: " << elapsed.count() << " seconds\n";
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

    lu_decomposition_omp(A, L, U, n, t);

    if (p == 1) {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    return 0;
}
