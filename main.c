#include "matrix.h"
#include "nn.h"
#include "mnist.h"
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

//#define DEBUG
#ifdef DEBUG
  //to use this function you need to provide every weight and bias of the NN, and the size of each layer, i recommend to previous save this on a file then import it.
  NN n = nn_back_up(&sigmoid);
  Gradient g = gg_create_from_nn(n);

  Mat input = mat_create_from(NUM_TEST, SIZE, test_image);
  Mat output = mat_create(NUM_TEST, 10);
  for(int i = 0; i < NUM_TEST; i++) {
    for(int j = 0; j < 10; j++)
      MAT_AT(output, i, 0) = (double) 0.0f;
    MAT_AT(output, i, test_label[i]) = 1.0f;
  }

  //for another layout this print should change
  for(int i = 0; i < 1; i++) {
    SubMat in_row = mat_get_row(input, i);
    SubMat out_row = mat_get_row(output, i);
    double res[10];
    nn_forward(n, in_row, res);
    for(int j = 0; j < 10; j++){
      printf("%lf %lf", res[j], LINE_AT(out_row, j));
      printf("\n");
    }

  }
  printf("cost: %lf\n", nn_cost(n, input, output));
  nn_destruct(n, g);
#else
  load_mnist();

  Mat input = mat_create_from(NUM_TRAIN/6, SIZE, train_image);
  Mat output = mat_create(NUM_TRAIN/6, 10);
  for(int i = 0; i < NUM_TRAIN/6; i++) {
    for(int j = 0; j < 10; j++)
      MAT_AT(output, i, 0) = (double) 0.0f;
    MAT_AT(output, i, train_label[i]) = 1.0f;
  }

  size_t ar[] = {SIZE, 16, 16, 10};
  NN n = nn_create(ar, ARRAY_LEN(ar), &sigmoid, &deriv_sig);
  Gradient g = gg_create_from_nn(n);
  nn_randomize_params(n);

  int iterations = 2000;
  while(iterations--) {
    nn_backward_propagation(n, g, input, output);
    nn_learn(n, g);
    printf("%d\n", iterations);
  }

  //this function is used to save the parameters of the current training, send this to a file to after use
  nn_save_up(n);
  mat_destruct(input);
  mat_destruct(output);

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

  return 0;
}
