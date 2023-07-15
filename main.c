#include <stdio.h>

#include "nn.h"

int main() {

  Mat m = mat_create(2, 2);
  Mat m2 = mat_create(2, 2);
  mat_fill(m, 1.0f);
  mat_fill(m2, 2.0f);
  MAT_AT(m2, 1, 1) = 5.0f;
  MAT_AT(m, 1, 1) = 5.0f;

  MAT_PRINT(m);
  MAT_PRINT(m2);
  Mat ret = mat_create(2, 2);
  mat_mul(ret, m, m2);

  MAT_PRINT(ret);

  Mat sum = mat_create(2, 2);
  mat_fill(sum, 1.0f);
  mat_add(ret, sum);
  MAT_PRINT(ret);

  //SubMat may not be deallocated
  SubMat sub = mat_get_submat(ret, (SubMatDim){ .beginRow = 1, .qtdRows = 1, .beginCol = 1, .qtdCols = 1});

  MAT_PRINT(sub);

  mat_destruct(&ret);
  mat_destruct(&sum);
  mat_destruct(&m);
  mat_destruct(&m2);
  return 0;
}
