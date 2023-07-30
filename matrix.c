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

Mat mat_create_from(size_t rows, size_t cols, size_t elements[rows][cols])
{
  Mat m = mat_create(rows, cols);
  for(size_t i = 0; i < rows; i++)
    for(size_t j = 0; j < cols; j++)
      MAT_AT(m, i, j) = (double) elements[i][j];
  return m;
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

// a double from 0 to 1
double random_value()
{
  return (double) rand() / (double) RAND_MAX;
}

void mat_randomize(Mat m)
{
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = random_value();
    }
  }
}

void mat_copy(Mat dest, Mat src)
{
  assert(dest.rows == src.rows);
  assert(dest.cols == src.cols);

  memcpy((void *)dest.data, (void *)src.data, sizeof(double)*src.rows*src.cols);
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

void mat_const_mul(Mat dest, double val)
{
  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) *= val;
    }
  }
}

void mat_add(Mat dest, Mat first, Mat second)
{
  assert(dest.rows == first.rows && first.rows == second.rows);
  assert(dest.cols == first.cols && first.cols == second.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) = MAT_AT(first, i, j) + MAT_AT(second, i, j);
    }
  }
}

void mat_sub(Mat dest, Mat first, Mat second)
{
  assert(dest.rows == first.rows && first.rows == second.rows);
  assert(dest.cols == first.cols && first.cols == second.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      MAT_AT(dest, i, j) = MAT_AT(first, i, j) - MAT_AT(second, i, j);
    }
  }
}

void mat_activation_fn(Mat m, double (* function)(double))
{
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = function(MAT_AT(m, i, j));
    }
  }
}

SubMat mat_get_submat(Mat origin, SubMatDim dim) {
  assert(dim.beginRow >= 0 && dim.beginRow < origin.rows);
  assert(dim.beginCol >= 0 && dim.beginCol < origin.cols);
  assert((dim.beginRow + dim.qtdRows) <= origin.rows);
  assert((dim.beginCol + dim.qtdCols) <= origin.cols);

  return (SubMat) {
    .rows = dim.qtdRows,
    .cols = dim.qtdCols,
    .row_size = origin.row_size,
    .data = &MAT_AT(origin, dim.beginRow, dim.beginCol)
  };
}

SubMat mat_get_row(Mat origin, size_t row)
{
  return mat_get_submat(origin, (SubMatDim){row, 0, 1, origin.cols});
}

SubMat mat_get_col(Mat origin, size_t col)
{
  return mat_get_submat(origin, (SubMatDim){0, col, origin.rows, 1});
}

void mat_destruct(Mat m)
{
  free(m.data);
}
