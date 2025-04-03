#include <iostream>
#include <omp.h>
#include <random>  // C++11 RNG
#include <cstdlib> // atoi, malloc
#include <ctime>   // time


void usage(const char *name) {
	std::cout << "usage: " << name << " matrix-size nworkers" << std::endl;
 	exit(-1);
}

void lu_decomposition(double** a, int* pi, int n){
  // pi 초기화
  for(int i = 0; i < n; i++)
    pi[i] = i;

  for(int k = 0; k < n; k++){
    
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
  int n = nworkers;
  double** a = new double*[n];
  for(int i = 0; i < n; i++)
    a[i] = new double[n];

  // 2) 병렬 난수 초기화 (use omp)
#pragma omp parallel
  {
    // 스레드 ID별로 고유한 시드값 사용
    int tid = omp_get_thread_num();
    std::mt19937 rng(time(NULL) + tid); // seed 다르게
    std::uniform_real_distribution<double> dist(0.0, 1.0);

  #pragma omp for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[i][j] = dist(rng);
  }

  // // (테스트 출력용) 상위 좌측 5x5만 출력
  // std::cout << "Top-left corner of matrix A:" << std::endl;
  // for (int i = 0; i < std::min(5, n); ++i) {
  //     for (int j = 0; j < std::min(5, n); ++j) {
  //         std::cout << a[i][j] << " ";
  //     }
  //     std::cout << "\n";
  // }

  // 3) LU decomposition
  int* pi = new int[n];
  lu_decomposition(a, pi, n);

  // 4) Memory Free
  for (int i = 0; i < n; ++i) 
    delete[] a[i];
  delete[] a;
  delete[] pi;

  return 0;
}

