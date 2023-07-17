#include "nn.h"
#include "matrix.h"

NN nn_create(size_t *layers, size_t layersLen, double (* activate_function)(double))
{
  assert(layersLen >= 2); //must have at least an input and an output layer

  NN n = (NN){
    .is = (Mat *) malloc(sizeof(Mat) * (layersLen - 1)),
    .ws = (Mat *) malloc(sizeof(Mat) * (layersLen - 1)),
    .bs = (Mat *) malloc(sizeof(Mat) * (layersLen - 1)),
    .nnLayers = layersLen - 1,
    .activation_function = activate_function
  };

  for(size_t i = 0; i < n.nnLayers; i++) {
    n.is[i] = mat_create(1, layers[i+1]);
    n.bs[i] = mat_create(1, layers[i+1]);
    n.ws[i] = mat_create(layers[i], layers[i+1]);
  }

  return n;
}

void nn_destruct(NN n)
{
  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_destruct(n.is[i]);
    mat_destruct(n.bs[i]);
    mat_destruct(n.ws[i]);
  }
  free(n.is);
  free(n.bs);
  free(n.ws);
  n.is = NULL;
  n.bs = NULL;
  n.ws = NULL;
}

void nn_randomize_params(NN n)
{
  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_randomize(n.bs[i]);
    mat_randomize(n.ws[i]);
  }
}

void nn_forward_propagation(NN n, Mat input_train, double *output)
{
  assert(input_train.rows == 1 && input_train.cols == n.ws[0].rows);

  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_mul(n.is[i], input_train, n.ws[i]);
    mat_add(n.is[i], n.bs[i]);
    mat_activation_fn(n.is[i], n.activation_function);
    input_train = n.is[i];
  }

  memcpy((void *)output, (void *)OUTPUT(n).data, OUTPUT(n).cols*sizeof(double));
}

double nn_cost(NN n, Mat train_input, Mat train_output)
{
  assert(n.ws[0].rows == train_input.cols);
  assert(OUTPUT(n).cols == train_output.cols);

  float c = 0.0f;
  size_t len = train_input.rows;

  for(size_t i = 0; i < len; i++) {
    SubMat in = mat_get_row(train_input, i);
    SubMat out = mat_get_row(train_output, i);

    double output_vals[OUTPUT(n).cols];
    nn_forward_propagation(n, in, output_vals);

    for(size_t j = 0; j < out.cols; j++) {
      float diff = output_vals[j] - out.data[j];
      c += diff*diff;
    }
  }

  return c/(double)len;
}

void nn_finite_diff_learn(NN n, Mat train_input, Mat train_output, double eps, double rate){
  double c, saved, finite_diff;
  for(size_t i = 0; i < n.nnLayers; i++) {
    for(size_t j = 0; j < n.ws[i].rows; j++) {
      for(size_t k = 0; k < n.bs[i].cols; k++) {
        //trainin weights
        c = nn_cost(n, train_input, train_output);
        saved = MAT_AT(n.ws[i], j, k);
        MAT_AT(n.ws[i], j, k) += eps;
        finite_diff = (nn_cost(n, train_input, train_output) - c)/eps;
        MAT_AT(n.ws[i], j, k) = saved;
        MAT_AT(n.ws[i], j, k) -= rate*finite_diff;

        //training bias, always will have the same amount of cols of weights
        c = nn_cost(n, train_input, train_output);
        saved = MAT_AT(n.bs[i], 0, k);
        MAT_AT(n.bs[i], 0, k) += eps;
        finite_diff = (nn_cost(n, train_input, train_output) - c)/eps;
        MAT_AT(n.bs[i], 0, k) = saved;
        MAT_AT(n.bs[i], 0, k) -= rate*finite_diff;
      }
    }
  }
}

double sigmoid(double x) 
{
  return 1.0f/(1.0f + (1.0f/exp(x)));
}

double reLU(double x)
{
  return x < 0.0f ? 0 : x;
}
