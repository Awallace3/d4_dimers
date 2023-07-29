#include "helper.hpp"
#include "input.hpp"
#include "omp.h"
#include "stdio.h"
#include <Eigen/Dense>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


int main(int argc, char *argv[]) {
  printf("\nRunning: %s\n\n", argv[0]);
  if (argc == 1) {
    printf("You must pass a data path and number of threads like "
           "below:\n\t./hf data/t1 4\n");
    return 1;
  }
  return 0;
}
