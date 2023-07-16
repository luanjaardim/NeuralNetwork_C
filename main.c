#include <stdio.h>

#include "matrix.h"
#include "nn.h"

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

//size_t xor_table[] = {
//  0, 0, 0,
//  0, 1, 1,
//  1, 0, 1,
//  1, 1, 0,
//};

int main(void) 
{
  srand(time(0));

  Mat input = mat_create(4, 3);
  for(int i = 0; i < input.rows; i++)
    for(int j = 0; j < input.rows; j++)
      MAT_AT(input, i, j) = or_table[i][j];
  Mat weights = mat_create(2, 1);
  Mat bias = mat_create(1, 1);

  mat_randomize(weights);
  mat_randomize(bias);

  double ds   = 1e-1;
  double rate = 1e-1;

  double c = cost(weights, bias, input);
  printf("%lf\n", c);
  for(size_t i = 0; i < 10000; i++) {
    c = cost(weights, bias, input);
    double savedw1 = MAT_AT(weights, 0, 0);
    double savedw2 = MAT_AT(weights, 1, 0);
    double savedb = MAT_AT(bias, 0, 0);

    MAT_AT(weights, 0, 0) += ds;
    double dw1 = (cost(weights, bias, input) - c)/ds;
    MAT_AT(weights, 0, 0) = savedw1;
    MAT_AT(weights, 1, 0) += ds; 
    double dw2 = (cost(weights, bias, input) - c)/ds;
    MAT_AT(weights, 1, 0) = savedw2;
    MAT_AT(bias, 0, 0) += ds; 
    double dwb = (cost(weights, bias, input) - c)/ds;
    MAT_AT(bias, 0, 0) = savedb;

    MAT_AT(weights, 0, 0) -= rate*dw1;
    MAT_AT(weights, 1, 0) -= rate*dw2; 
    MAT_AT(bias, 0, 0) -= rate*dwb;
  }

  c = cost(weights, bias, input);
  printf("%lf\n", c);
  MAT_PRINT(weights);
  MAT_PRINT(bias);

  Mat result = mat_create(1, 1);
  SubMat input_train = mat_get_submat(input, (SubMatDim){0,0,input.rows,2});
  for(size_t i = 0; i < input.rows; i++) {
    mat_mul(result, mat_get_row(input_train, i), weights);
    mat_add(result, bias);
    mat_activation_fn(result, &sigmoid);
    MAT_PRINT(result);
  }

  return 0;
}
