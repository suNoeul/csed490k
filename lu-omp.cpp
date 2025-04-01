#include <iostream>
#include <omp.h>

void usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}


int main(int argc, char **argv)
{

  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  std::cout << name << ": " 
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

#pragma omp parallel
{
// uncomment the line below to remove the data race
// #pragma omp critical
	std::cout << "hello world from thread " 
                  << omp_get_thread_num() << std::endl;
}
return 0;
}
