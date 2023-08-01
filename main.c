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

size_t xnor_table[][4] = {
  {0, 0, 1, 0},
  {0, 1, 0, 1},
  {1, 0, 0, 1},
  {1, 1, 1, 0}
};

int main(void) 
{
  srand(time(0));

  Mat input = mat_create_from(4, 4, xnor_table);
  SubMat in = mat_get_submat(input, (SubMatDim) {0, 0, 4, 2});
  SubMat out = mat_get_submat(input, (SubMatDim) {0, 2, 4, 2});

  size_t ar[] = {2, 2, 2};
  NN n = nn_create(ar, ARRAY_LEN(ar), &sigmoid, &deriv_sig);
  Gradient g = gg_create_from_nn(n);
  nn_randomize_params(n);

  printf("first cost: %lf\n", nn_cost(n, in, out));
  double eps = 1e-1, rate= 1;
  int iterations = 10000;
  while(iterations--) {
    nn_backward_propagation(n, g, in, out);
    nn_learn(n, g);
  }
  printf("after training cost: %lf\n", nn_cost(n, in, out));

#define DEBUG
#ifdef DEBUG
  for(int i = 0; i < 4; i++) {
    SubMat in_row = mat_get_row(in, i);
    SubMat out_row = mat_get_row(out, i);
    double res[2];
    nn_forward(n, in_row, res);
    printf("%d -> generated: %lf and %lf, expected: %lf and %lf\n", i, res[0], res[1], out_row.data[0], out_row.data[1]);
  }
#endif

  nn_destruct(n, g);
  mat_destruct(input);

  return 0;
}
