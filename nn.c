#include "nn.h"
#include "matrix.h"

NN nn_create(size_t *layers, size_t layersLen, double (* activate_function)(double), double (* deriv_act_function)(double))
{
  assert(layersLen >= 2); //must have at least an input and an output layer

  NN n = (NN){
    .is = (Mat *) malloc(sizeof(Mat) * (layersLen)),
    .ws = (Mat *) malloc(sizeof(Mat) * (layersLen - 1)),
    .bs = (Mat *) malloc(sizeof(Mat) * (layersLen - 1)),
    .nnLayers = layersLen - 1,
    .activation_function = activate_function,
    .deriv_act_function = deriv_act_function
  };

  for(size_t i = 0; i < n.nnLayers; i++) {
    n.is[i+1] = mat_create(1, layers[i+1]);
    n.bs[i] = mat_create(1, layers[i+1]);
    n.ws[i] = mat_create(layers[i], layers[i+1]);
  }
  n.is[0] = mat_create(1, layers[0]);

  /* The first layer of the intermediate values has the input
   * The last layer of the intermediate values has the output
   *
   * There is one intermediate matrix more than bias and weight matrixes
   */

  return n;
}

Gradient gg_create_from_nn(NN n)
{
  Gradient g = (Gradient) {
    .nnLayers = n.nnLayers,
    .is = (Mat *) malloc(sizeof(Mat) * (n.nnLayers + 1)),
    .ws = (Mat *) malloc(sizeof(Mat) * (n.nnLayers)),
    .bs = (Mat *) malloc(sizeof(Mat) * (n.nnLayers)),
    .activation_function = n.activation_function,
    .deriv_act_function = n.deriv_act_function
  };
  for(size_t i = 0; i < g.nnLayers; i++) {
    g.is[i+1] = mat_create(n.is[i+1].rows, n.is[i+1].cols);
    g.ws[i] = mat_create(n.ws[i].rows, n.ws[i].cols);
    g.bs[i] = mat_create(n.bs[i].rows, n.bs[i].cols);
  }
  g.is[0] = mat_create(n.is[0].rows, n.is[0].cols);

  return g;
}

void nn_destruct(NN n, Gradient g)
{

  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_destruct(n.is[i+1]);
    mat_destruct(n.bs[i]);
    mat_destruct(n.ws[i]);
    mat_destruct(g.is[i+1]);
    mat_destruct(g.bs[i]);
    mat_destruct(g.ws[i]);
  }
  mat_destruct(n.is[0]);
  mat_destruct(g.is[0]);

  free(n.is);
  free(n.bs);
  free(n.ws);
  n.is = NULL;
  n.bs = NULL;
  n.ws = NULL;
  free(g.is);
  free(g.bs);
  free(g.ws);
  g.is = NULL;
  g.bs = NULL;
  g.ws = NULL;
}

void nn_zero(NN n)
{
  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_fill(n.is[i+1], 0.0f);
    mat_fill(n.ws[i], 0.0f);
    mat_fill(n.bs[i], 0.0f);
  }
  mat_fill(n.is[0], 0.0f);
}

void mat_softmax(Mat matrix) {
  assert(matrix.rows == 1);
  double sum = 0;
  for(int i = 0; i < matrix.cols; i++) sum += exp(LINE_AT(matrix, i));
  
  for(int i = 0; i < matrix.cols; i++) LINE_AT(matrix, i) = exp(LINE_AT(matrix, i))/sum;
}

void nn_randomize_params(NN n)
{
  for(size_t i = 0; i < n.nnLayers; i++) {
    mat_randomize(n.bs[i]);
    mat_randomize(n.ws[i]);
  }
}

void nn_forward(NN n, Mat input_train, double *output)
{
  assert(input_train.rows == n.is[0].rows && input_train.cols == n.is[0].cols);
  mat_copy(INPUT(n), input_train);

  for(size_t i = 0; i < n.nnLayers-1; i++) {
    mat_mul(n.is[i+1], n.is[i], n.ws[i]);
    mat_add(n.is[i+1], n.is[i+1], n.bs[i]);
    mat_apply_fn(n.is[i+1], n.is[i+1], n.activation_function);
  }
  mat_mul(OUTPUT(n), n.is[n.nnLayers - 1], n.ws[n.nnLayers - 1]);
  mat_add(OUTPUT(n), OUTPUT(n), n.bs[n.nnLayers - 1]);
  mat_softmax(OUTPUT(n));

  if(output != NULL)
    memcpy((void *)output, (void *)OUTPUT(n).data, OUTPUT(n).cols*sizeof(double));
}

void nn_backward_propagation(NN n, Gradient g, Mat train_input, Mat train_output)
{
  nn_zero(g);
  size_t m = train_input.rows;

  for(size_t sample = 0; sample < train_input.rows; sample++)
  {
    SubMat input_sample = mat_get_row(train_input, sample);
    SubMat output_sample = mat_get_row(train_output, sample);

    nn_forward(n, input_sample, NULL);

    //cleaning the intermediates values used on the previoues sample calculation
    for(size_t j = 0; j < g.nnLayers + 1; j++) {
      mat_fill(g.is[j], 0.0f);
    }

    for(size_t i = 0; i < output_sample.cols; i++) {
      LINE_AT(OUTPUT(g), i) = LINE_AT(OUTPUT(n), i) - LINE_AT(output_sample, i);
    }

    for(int l = n.nnLayers - 1; l >= 0; l--) 
    {
      float cur_act_value, diff_expected, deriv_cur_act_value;
      for(int i = 0; i < n.ws[l].cols; i++) {
        diff_expected = LINE_AT(g.is[l+1], i);
        cur_act_value = LINE_AT(n.is[l+1], i);
        deriv_cur_act_value = n.deriv_act_function(cur_act_value);

        LINE_AT(g.bs[l], i) += 2*diff_expected*deriv_cur_act_value;

        for(int j = 0; j < n.ws[l].rows; j++) {
          MAT_AT(g.ws[l], j, i) += 2*diff_expected*deriv_cur_act_value*LINE_AT(n.is[l], j);
          LINE_AT(g.is[l], j) += 2*diff_expected*deriv_cur_act_value*MAT_AT(n.ws[l], j, i);
        }
      }
    }
  }

  for(size_t i = 0; i < n.nnLayers; i++)
  {
    for (size_t j = 0; j < g.ws[i].rows; j++) {
        for (size_t k = 0; k < g.ws[i].cols; k++) {
            MAT_AT(g.ws[i], j, k) /= (double)m;
        }
    }
    for (size_t k = 0; k < g.bs[i].cols; k++) {
        LINE_AT(g.bs[i], k) /= (double)m;
    }
  }
}

void nn_learn(NN n, Gradient g, double rate)
{
  for(size_t i = 0; i < n.nnLayers; i++) {
    for(size_t j = 0; j < n.ws[i].cols; j++) {
      LINE_AT(n.bs[i], j) -= rate * LINE_AT(g.bs[i], j);
      for(size_t k = 0; k < n.ws[i].rows; k++)
        MAT_AT(n.ws[i], k, j) -= rate * MAT_AT(g.ws[i], k, j);
    }
  }
}

double nn_cost(NN n, Mat train_input, Mat train_output)
{
  assert(n.ws[0].rows == train_input.cols);
  assert(OUTPUT(n).cols == train_output.cols);
  assert(train_output.rows == train_input.rows);

  float c = 0.0f;
  size_t len = train_input.rows;

  double output_vals[OUTPUT(n).cols];
  for(size_t i = 0; i < len; i++) {
    SubMat in = mat_get_row(train_input, i);
    SubMat out = mat_get_row(train_output, i);

    nn_forward(n, in, output_vals);

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
        //training weights
        c = nn_cost(n, train_input, train_output);
        saved = MAT_AT(n.ws[i], j, k);
        MAT_AT(n.ws[i], j, k) += eps;
        finite_diff = (nn_cost(n, train_input, train_output) - c)/eps;
        MAT_AT(n.ws[i], j, k) = saved;
        MAT_AT(n.ws[i], j, k) -= rate*finite_diff;

      }
    }

    for(size_t k = 0; k < n.bs[i].cols; k++) {
      //training bias
      c = nn_cost(n, train_input, train_output);
      saved = MAT_AT(n.bs[i], 0, k);
      MAT_AT(n.bs[i], 0, k) += eps;
      finite_diff = (nn_cost(n, train_input, train_output) - c)/eps;
      MAT_AT(n.bs[i], 0, k) = saved;
      MAT_AT(n.bs[i], 0, k) -= rate*finite_diff;
    }
  }
}

void nn_accuracy(NN n, Mat test_input, Mat test_output) {
  size_t m = test_input.rows;
  double res[OUTPUT(n).cols];
  int correct = 0;
  for(int i = 0; i < m; i++) {
    SubMat in_row = mat_get_row(test_input, i);
    SubMat out_row = mat_get_row(test_output, i);
    nn_forward(n, in_row, res);

    int greatest = 0;
    for(int j = 1; j < OUTPUT(n).cols; j++)
      greatest = res[greatest] < res[j] ? j : greatest;

    if(greatest == *out_row.data) correct++;
    for(int j = 0; j < OUTPUT(n).cols; j++) printf("%.2lf ", res[j]);
    printf("%d %d\n", greatest, greatest == *out_row.data);
  }
  printf("total: %zu, correct infered: %d\n", m, correct);
}

void nn_save_up(NN n) {
  printf("%zu\n", n.nnLayers + 1);
  for(int i = 0; i < n.nnLayers + 1; i++) {
    printf("%zu ", n.is[i].cols);
  }
  printf("\n");

  for(int i = 0; i < n.nnLayers; i++) {
  //printf("peso\n");
    for(int j = 0; j < n.ws[i].rows; j++) {
      for(int k = 0; k < n.ws[i].cols; k++)
        printf("%lf ", MAT_AT(n.ws[i], j, k));
      printf("\n");
    }
  //printf("bias\n");
    for(int j = 0; j < n.bs[i].cols; j++)
      printf("%lf ", LINE_AT(n.bs[i], j));
    printf("\n");
  }
}

NN nn_back_up(double (* activate_function)(double)) {
  size_t count_layers;
  scanf("%zu", &count_layers);

  size_t arr[count_layers];
  for(int i = 0; i < count_layers; i++) {
    scanf("%zu", arr + i);
  }

  NN n = nn_create(arr, count_layers, activate_function, NULL); 
  //as we are only using the forward, we don't need the deriv function
  
  for(int i = 0; i < n.nnLayers; i++) {
    for(int j = 0; j < n.ws[i].rows; j++) {
      for(int k = 0; k < n.ws[i].cols; k++) {
        scanf("%lf", &MAT_AT(n.ws[i], j, k));
      }
    }
    for(int j = 0; j < n.bs[i].cols; j++)
      scanf("%lf", &LINE_AT(n.bs[i], j));
  }

  return n;
}

double sigmoid(double x) 
{
  return 1.0f/(1.0f + (exp(-x)));
}

double deriv_sig(double x)
{
  return (x) * (1 - (x));
}

double reLU(double x)
{
  return x < 0.01 * x  ? (0.01 * x) : x;
}

double deriv_reLu(double x)
{
  return x >= 0 ? 1.0f : 0.0f;
}
