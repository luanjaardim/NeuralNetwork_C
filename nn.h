#ifndef NN_LIB
#define NN_LIB

#include "matrix.h"

typedef struct NeuralNetwork {
  Mat *is; //intermediates values
  Mat *ws; //weights
  Mat *bs; //bias
  size_t nnLayers;
  double (* activation_function)(double);
} NN;

NN nn_create(size_t *layers, size_t layersLen, double (* activation_function)(double));
void nn_randomize_params(NN n);
void nn_forward_propagation(NN n, Mat input_train, double *output);
void nn_destruct(NN n);
double nn_cost(NN n, Mat train_input, Mat train_output);
void nn_finite_diff_learn(NN n, Mat train_input, Mat train_output, double eps, double rate);

#define ARRAY_LEN(ar) sizeof(ar)/sizeof((ar)[0])
#define OUTPUT(n) (n).is[(n).nnLayers - 1]

double sigmoid(double x);
double reLU(double x);

#endif //NN_LIB
