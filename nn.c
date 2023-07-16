#include "nn.h"
#include "matrix.h"
#include <math.h>

double cost(Mat weights, Mat bias, Mat train_data)
{
  assert(weights.rows == train_data.cols - 1); //minus one cause we have the output at the last column
  float c = 0.0f;
  size_t n = train_data.rows;

  Mat tmp = mat_create(1, 1);
  for(size_t i = 0; i < n; i++) {
    SubMat in = mat_get_submat(train_data, (SubMatDim){i, 0, 1, train_data.cols - 1});
    SubMat out = mat_get_submat(train_data, (SubMatDim){i, train_data.cols - 1, 1, 1});
   // MAT_PRINT(in);
   // MAT_PRINT(out);
   // MAT_PRINT(weights);
    mat_mul(tmp, in, weights);
    mat_add(tmp, bias);
    mat_activation_fn(tmp, &sigmoid);
    float diff = *tmp.data - *out.data;
    c += diff*diff;
  }

  mat_destruct(&tmp);
  return c/(double)n;
}

double sigmoid(double x) 
{
  return 1.0f/(1.0f + (1.0f/exp(x)));
}

double reLU(double x)
{
  return x < 0.0f ? 0 : x;
}
