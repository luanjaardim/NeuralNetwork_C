#include "matrix.h"
#include "nn.h"
#include "mnist.h"
#include <stdio.h>

double and_table[][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 1.0, 0.0},
  {1.0, 0.0, 0.0},
  {1.0, 1.0, 1.0}
};

double or_table[][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 1.0, 1.0},
  {1.0, 0.0, 1.0},
  {1.0, 1.0, 1.0}
};

double xor_table[][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 1.0, 1.0},
  {1.0, 0.0, 1.0},
  {1.0, 1.0, 0.0}
};

double xnor_table[][4] = {
  {0.0, 0.0, 1.0, 0.0},
  {0.0, 1.0, 0.0, 1.0},
  {1.0, 0.0, 0.0, 1.0},
  {1.0, 1.0, 1.0, 0.0}
};

#define NUM_TO_TEST 1e3
#define NUM_TO_TRAIN NUM_TRAIN/6

int main(void) 
{
  srand(time(NULL));

  load_mnist();

//#define DEBUG
#ifdef DEBUG

  //to use this function you need to provide every weight and bias of the NN, and the size of each layer, i recommend to previous save this on a file(undefining DEBUG) then import it.
  NN n = nn_back_up(&sigmoid);
  Gradient g = gg_create_from_nn(n);

  Mat input = mat_create_from(NUM_TO_TEST, SIZE, test_image);
  Mat output = mat_create(NUM_TO_TEST, 1);
  mat_transpose(&output);
  for(int i = 0; i < NUM_TO_TEST; i++) {
    LINE_AT(output, i) = (double)test_label[i];
  }
  mat_transpose(&output);

  nn_accuracy(n, input, output);

  nn_destruct(n, g);
  mat_destruct(input);
  mat_destruct(output);

#else

  Mat input = mat_create_from(NUM_TO_TRAIN, SIZE, train_image);
  Mat output = mat_create(NUM_TO_TRAIN, 10);
  mat_fill(output, 0.0f);
  for(int i = 0; i < NUM_TO_TRAIN; i++) {
    MAT_AT(output, i, train_label[i]) = 1.0f;
  }

  size_t ar[] = {SIZE, 128, 32, 10};
  NN n = nn_create(ar, ARRAY_LEN(ar), &sigmoid, &deriv_sig);
  Gradient g = gg_create_from_nn(n);
  nn_randomize_params(n);
  nn_forward(n, mat_get_row(input, 0), NULL);

  int iterations = 50;
  while(iterations--) {
    nn_backward_propagation(n, g, input, output);
    nn_learn(n, g, 2);
    printf("current cost: %lf\n", nn_cost(n, input, output));
  }

  //this function is used to save the parameters of the current training, send this to a file to after use
  nn_save_up(n);

  mat_destruct(input);
  mat_destruct(output);
  nn_destruct(n, g);

#endif

  return 0;
}
