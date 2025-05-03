
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>     

#ifdef PARALLEL
#include <omp.h>
#endif

using Matrix = std::vector<std::vector<double>>;

bool lu_validation(const Matrix& A, const Matrix& L, const Matrix& U, double tolerance = 1e-6) {
    int n = A.size();
    bool valid = true;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += L[i][k] * U[k][j];
            }
            if (std::abs(sum - A[i][j]) > tolerance) {
                std::cout << "Mismatch at (" << i << "," << j << "): " << "LU=" << sum << ", A=" << A[i][j] << "\n";
                valid = false;
            }
        }
    }

    if (valid) 
        std::cout << "Validation passed: L * U is approximately equal to A.\n";
    else 
        std::cout << "Validation failed: L * U does not match A.\n";

    return valid;
}

void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

void lu_decomposition(Matrix& A_input, Matrix& L, Matrix& U, int n, int t) { 
    /* Basic LU decomposition (A = LU) */

#ifdef PARALLEL
    omp_set_num_threads(t);
#endif

    Matrix A = A_input;    

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {
        // 1. Check for zero pivot 
        if (A[k][k] == 0.0) {
            std::cerr << "LU Decomposition Error: Zero pivot encountered at row " << k << std::endl;
            exit(EXIT_FAILURE);
        }

        // 2. Compute U[k][j] 
        for (int j = k; j < n; ++j)
            U[k][j] = A[k][j];

        // 3. Compute L[i][k] = A[i][k] / U[k][k] 
        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i)
            L[i][k] = A[i][k] / A[k][k];

        // 4. Update A[i][j] 
        #pragma omp parallel for collapse(2)
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    // 5. Set the diagonal of L to 1.0
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
    L[i][i] = 1.0;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "A = LU decomposition(OMP) time: " << elapsed.count() << " seconds\n";          
}

void lu_decomposition_omp(Matrix& A_input, Matrix& L, Matrix& U, int n, int t) {
    /* LU decomposition with partial pivoting (PA = LU) */

#ifdef PARALLEL
    omp_set_num_threads(t);
#endif

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
        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A[i][k] / A[k][k];
        }

        // 6. Update the submatrix A[i][j]
        #pragma omp parallel for collapse(2)
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    // 7. Set diagonal of L to 1
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        L[i][i] = 1.0;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "PA = LU decomposition(OMP) time: " << elapsed.count() << " seconds\n";       
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

// Matrix A = {
//     {1.8043e+09, 8.4693e+08, 1.6817e+09, 1.7146e+09},
//     {1.9577e+09, 4.2424e+08, 7.1989e+08, 1.6498e+09},
//     {5.9652e+08, 1.1896e+09, 1.0252e+09, 1.3505e+09},
//     {7.8337e+08, 1.1025e+09, 2.0449e+09, 1.9675e+09}
// };