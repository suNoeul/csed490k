#include <iostream>
#include <omp.h>

#include <random>  // C++11 RNG
#include <chrono> // 시간 측정용
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

#define DO_VERIFY 1 // 검증 모드 (0으로 바꾸면 컴파일 제외)

using namespace std;


void allocate_multiple_matrices(double*** matrices[], int n);
void free_multiple_matrices(double*** matrices[], int n);

double** allocate_matrix(int n);
void free_matrix(double** m, int n);

void extract_LU(double** a, double** L, double** U, int n);
void apply_permutation(double** src, double** dst, int* pi, int n);
void matrix_multiply(double** A, double** B, double** C, int n);
void compute_residual(double** PA, double** LU, double** R, int n);
double compute_l21_norm(double** R, int n);

void usage(const char *name) {
	std::cout << "usage: " << name << " matrix-size nworkers" << std::endl;
 	exit(-1);
}

void lu_decomposition(double** a, int* pi, int n){
  // pi 초기화
  for(int i = 0; i < n; i++)
    pi[i] = i;

    for (int k = 0; k < n; ++k) {
      // 1. Pivot 선택 (|a[i][k]| 최대 찾기)
      double max_val = 0.0;
      int k_prime = k;

      for (int i = k; i < n; ++i) {
          double abs_val = std::abs(a[i][k]);
          if (abs_val > max_val) {
              max_val = abs_val;
              k_prime = i;
          }
      }

      // 2. Singular 체크
      if (max_val == 0.0) {
          std::cerr << "LU Decomposition Error: Singular matrix!" << std::endl;
          exit(EXIT_FAILURE);
      }

      // 3. 행 교환 (π, A)
      std::swap(pi[k], pi[k_prime]);
      std::swap( a[k], a[k_prime]);  // 포인터 swap → 전체 행 교환

      // 4. L, U 계산
    #pragma omp parallel for
      for (int i = k + 1; i < n; ++i) {
          a[i][k] /= a[k][k];  // L(i,k) = a(i,k) / U(k,k)
      }

      // 5. 아래쪽 A 갱신 (Schur complement)
    #pragma omp parallel for collapse(2)
      for (int i = k + 1; i < n; ++i) {
          for (int j = k + 1; j < n; ++j) {
              a[i][j] -= a[i][k] * a[k][j];
          }
      }
  }
}

int main(int argc, char **argv) {
  const char *name = argv[0];
  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);
  int nworkers = atoi(argv[2]);

  std::cout << name << ": " << matrix_size << " " << nworkers << std::endl;

  omp_set_num_threads(nworkers);

  // 1) NxN 배열 Allocation
  int n = matrix_size;
  double** a = allocate_matrix(n);
#if DO_VERIFY
  double **orig, **L, **U, **PA, **LU, **R;
  double ***matrices[] = { &orig, &L, &U, &PA, &LU, &R };
  allocate_multiple_matrices(matrices, n);
#endif

  // 2) 병렬 난수 초기화 (use omp)
#pragma omp parallel
  {
    // 스레드 ID별로 고유한 시드값 사용
    int tid = omp_get_thread_num();
    std::mt19937 rng(time(NULL) + tid); // seed 다르게
    std::uniform_real_distribution<double> dist(0.0, 1.0);

  #pragma omp for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j){
          a[i][j] = dist(rng);
#if DO_VERIFY
          orig[i][j] = a[i][j];
#endif
        }
  }

  // LU 분해 시간 측정 시작
  auto start = std::chrono::high_resolution_clock::now();

  // 3) LU decomposition
  int* pi = new int[n];
  lu_decomposition(a, pi, n);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "LU decomposition time: " << elapsed.count() << " seconds\n";

  // 결과 검증
#if DO_VERIFY
  auto verity_start = std::chrono::high_resolution_clock::now();
  extract_LU(a, L, U, n);
  apply_permutation(orig, PA, pi, n);
  matrix_multiply(L, U, LU, n);
  compute_residual(PA, LU, R, n);
  double norm = compute_l21_norm(R, n);
  auto verity_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> verity_elapsed = verity_end - verity_start;

  std::cout << "Verification time: " << verity_elapsed.count() << " seconds\n";
  std::cout << "L2,1 norm of residual: " << norm << std::endl;

  free_multiple_matrices(matrices, n);
#endif

  // 4) Memory Free
  free_matrix(a, n);
  delete[] pi;
  return 0;
}

/* ---------------------------------------------------------------------------- */

void allocate_multiple_matrices(double*** matrices[], int n) {
  for (int i = 0; i < 6; ++i)
      *matrices[i] = allocate_matrix(n);
}

void free_multiple_matrices(double*** matrices[], int n) {
  for (int i = 0; i < 6; ++i)
      free_matrix(*matrices[i], n);
}

void extract_LU(double** a, double** L, double** U, int n) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
          if (i > j) {
              L[i][j] = a[i][j];
              U[i][j] = 0.0;
          } else if (i == j) {
              L[i][j] = 1.0;
              U[i][j] = a[i][j];
          } else {
              L[i][j] = 0.0;
              U[i][j] = a[i][j];
          }
      }
}

void apply_permutation(double** src, double** dst, int* pi, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
          dst[i][j] = src[pi[i]][j];
}

void matrix_multiply(double** A, double** B, double** C, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
          C[i][j] = 0.0;
          for (int k = 0; k < n; ++k)
              C[i][j] += A[i][k] * B[k][j];
      }
}

void compute_residual(double** PA, double** LU, double** R, int n) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
          R[i][j] = PA[i][j] - LU[i][j];
}

double compute_l21_norm(double** R, int n) {
  double norm = 0.0;

#pragma omp parallel for reduction(+:norm)
  for (int j = 0; j < n; ++j) {
      double col_sum_sq = 0.0;
      for (int i = 0; i < n; ++i)
          col_sum_sq += R[i][j] * R[i][j];
      norm += std::sqrt(col_sum_sq);
  }
  return norm;
}

double** allocate_matrix(int n) {
  double** m = new double*[n];
  for (int i = 0; i < n; ++i)
      m[i] = new double[n];
  return m;
}

void free_matrix(double** m, int n) {
  for (int i = 0; i < n; ++i) delete[] m[i];
  delete[] m;
}



