#ifndef NN_LIB
#define NN_LIB

#include "matrix.h"

typedef struct NeuralNetwork {
  Mat *is; //intermediates values
  Mat *ws; //weights
  Mat *bs; //bias
  size_t nnLayers;
  double (* activation_function)(double);
  double (* deriv_act_function)(double);
} NN;

typedef NN Gradient;

NN nn_create(size_t *layers, size_t layersLen, double (* activation_function)(double), double (* deriv_act_function)(double));
Gradient gg_create_from_nn(NN n);
void nn_randomize_params(NN n);
void nn_forward(NN n, Mat input_train, double *output);
void nn_zero(NN n);
void nn_backward_propagation(NN n, Gradient g, Mat train_input, Mat train_output);
void nn_learn(NN n, Gradient g);
void nn_destruct(NN n, Gradient g);
double nn_cost(NN n, Mat train_input, Mat train_output);
void nn_finite_diff_learn(NN n, Mat train_input, Mat train_output, double eps, double rate);

#define ARRAY_LEN(ar) sizeof(ar)/sizeof((ar)[0])
#define OUTPUT(n) (n).is[(n).nnLayers]

double sigmoid(double x);
double deriv_sig(double x);
double reLU(double x);
double deriv_reLu(double x);

#endif //NN_LIB
