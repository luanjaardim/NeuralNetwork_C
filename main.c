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

size_t xnor_n_xor_table[][4] = {
  {0, 0, 1, 0},
  {0, 1, 0, 1},
  {1, 0, 0, 1},
  {1, 1, 1, 0}
};

int main(void) 
{
  srand(time(0));

  Mat input = mat_create_from(4, 4, xnor_n_xor_table);
  SubMat in = mat_get_submat(input, (SubMatDim) {0, 0, 4, 2});
  SubMat out = mat_get_submat(input, (SubMatDim) {0, 2, 4, 2});

#define DEBUG
#ifdef DEBUG
  //to use this function you need to provide every weight and bias of the NN, and the size of each layer, i recommend to previous save this on a file then import it.
  NN n = nn_back_up(&sigmoid);
  Gradient g = gg_create_from_nn(n);

  //for another layout this print should change
  for(int i = 0; i < 4; i++) {
    SubMat in_row = mat_get_row(in, i);
    SubMat out_row = mat_get_row(out, i);
    double res[2];
    nn_forward(n, in_row, res);
    printf("%d -> generated: %lf and %lf, expected: %lf and %lf\n", i, res[0], res[1], out_row.data[0], out_row.data[1]);

  }
  printf("cost: %lf\n", nn_cost(n, in, out));
  nn_destruct(n, g);
#else
  size_t ar[] = {2, 2, 2};
  NN n = nn_create(ar, ARRAY_LEN(ar), &sigmoid, &deriv_sig);
  Gradient g = gg_create_from_nn(n);
  nn_randomize_params(n);

  int iterations = 10000;
  while(iterations--) {
    nn_backward_propagation(n, g, in, out);
    nn_learn(n, g);
  }

  //this function is used to save the parameters of the current training, send this to a file to after use
  nn_save_up(n);

  //for another layout this print should change
  //for(int i = 0; i < 4; i++) {
  //  SubMat in_row = mat_get_row(in, i);
  //  SubMat out_row = mat_get_row(out, i);
  //  double res[2];
  //  nn_forward(n, in_row, res);
  //  printf("%d -> generated: %lf and %lf, expected: %lf and %lf\n", i, res[0], res[1], out_row.data[0], out_row.data[1]);
  //}

  nn_destruct(n, g);
#endif

  mat_destruct(input);

  return 0;
}
