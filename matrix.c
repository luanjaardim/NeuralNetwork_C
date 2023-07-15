#include "matrix.h"

Mat mat_create(size_t rows, size_t cols)
{
  return (Mat) {
    .rows = rows,
    .cols = cols,
    .row_size = cols,
    .data = malloc(sizeof(double) * rows * cols)
  };
}

void mat_print(Mat m, const char* variable_name)
{
  printf("%s = {\n", variable_name);
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      printf("%*s%.4lf", (int)2, "",  MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("}\n");
}

void mat_fill(Mat m, double element)
{
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = element;
    }
  }
}

void mat_copy(Mat dest, Mat src)
{
  assert(dest.rows == src.rows);
  assert(dest.cols == src.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_mul(Mat dest, Mat m1, Mat m2)
{
  assert(m1.cols == m2.rows);
  assert(m1.rows == dest.rows);
  assert(m2.cols == dest.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) = 0.0f;
      for(size_t k = 0; k < m1.cols; k++)
        MAT_AT(dest, i, j) += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
    }
  }
}

void mat_add(Mat dest, Mat another)
{
  assert(dest.rows == another.rows);
  assert(dest.cols == another.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) += MAT_AT(another, i, j);
    }
  }
}

SubMat mat_get_submat(Mat origin, SubMatDim dim) {
  assert(dim.beginRow >= 0 && dim.beginRow < origin.rows);
  assert(dim.beginCol >= 0 && dim.beginCol < origin.cols);
  assert((dim.beginRow + dim.qtdRows) <= origin.rows);
  assert((dim.beginCol + dim.qtdCols) <= origin.cols);

  return (Mat) {
    .rows = dim.qtdRows,
    .cols = dim.qtdCols,
    .row_size = origin.row_size,
    .data = &MAT_AT(origin, dim.beginRow, dim.beginCol)
  };
}

void mat_destruct(Mat *m)
{
  free(m->data);
}
