#include "matrix.h"
#include "nn.h"
#include <stdio.h>

size_t and_table[][3] = {
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};

size_t or_table[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1}
};

size_t xor_table[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

size_t xnor_table[][3] = {
  {0, 0, 1},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};

int main(void) 
{
  srand(time(0));

  Mat input = mat_create_from(4, 3, xnor_table);
  SubMat in = mat_get_submat(input, (SubMatDim) {0, 0, 4, 2});
  SubMat out = mat_get_submat(input, (SubMatDim) {0, 2, 4, 1});

  size_t ar[] = {2, 2, 1};
  NN n = nn_create(ar, ARRAY_LEN(ar), &sigmoid);
  nn_randomize_params(n);

 // for(size_t i = 0; i < n.nnLayers; i++) {
 //   MAT_PRINT(n.ws[i]);
 //   MAT_PRINT(n.bs[i]);
 //   MAT_PRINT(n.is[i]);
 // }

  printf("first cost: %lf\n", nn_cost(n, in, out));
  double eps = 1e-1, rate= 1e-1;
  int iterations = 20000;
  while(iterations--) {
    nn_finite_diff_learn(n, in, out, eps, rate);
  }
  printf("after training cost: %lf\n", nn_cost(n, in, out));

#define DEBUG
#ifdef DEBUG
  for(size_t i = 0; i < 4; i++) {
    SubMat in_row = mat_get_row(in, i);
    SubMat out_row = mat_get_row(out, i);
    double res;
    nn_forward_propagation(n, in_row, &res);
    printf("%zu -> generated: %lf, expected: %lf\n", i, res, *out_row.data);
  }
#endif

  nn_destruct(n);
  mat_destruct(input);

  return 0;
}
