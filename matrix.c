#include "matrix.h"

Mat mat_create(size_t rows, size_t cols)
{
  return (Mat) {
    .rows = rows,
    .cols = cols,
    .row_size = cols,
    .data = malloc(sizeof(double) * rows * cols),
    .T = 0
  };
}

void mat_transpose(Mat *m)
{
  /* Change the flag to indicate transpose or normal matrix
   * and swap the values of rows and cols
   */

  m->T = !(m->T);
  size_t tmp = m->rows;
  m->rows = m->cols;
  m->cols = tmp;
}

double *mat_at(Mat m, size_t i, size_t j)
{
  /* When we transpose a matrix we just swap the 
   * values of cols and rows ans start to access the
   * Matrix by j and i, instead of i and j
   */

  assert((i >= 0 && i < m.rows) && (j>= 0 && j < m.cols));

  return (m.T) ? m.data + (j*m.row_size + i) : m.data + (i*m.row_size + j);
}

Mat mat_create_from(size_t rows, size_t cols, size_t elements[rows][cols])
{
  Mat m = mat_create(rows, cols);
  for(size_t i = 0; i < rows; i++)
    for(size_t j = 0; j < cols; j++)
      *mat_at(m, i, j) = (double) elements[i][j];
  return m;
}

void mat_print(Mat m, const char* variable_name)
{
  printf("%s = {\n", variable_name);
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      printf("%*s%.4lf", (int)2, "",  *mat_at(m, i, j));
    }
    printf("\n");
  }
  printf("}\n");
}

void mat_fill(Mat m, double element)
{
  for(size_t i = 0; i < m.rows; i++) {
    for(size_t j = 0; j < m.cols; j++) {
      *mat_at(m, i, j) = element;
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
      *mat_at(m, i, j) = random_value();
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
       *mat_at(dest, i, j) = 0.0f;
      for(size_t k = 0; k < m1.cols; k++)
        *mat_at(dest, i, j) += *mat_at(m1, i, k) * *mat_at(m2, k, j);
    }
  }
}

void mat_const_mul(Mat dest, double val)
{
  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      *mat_at(dest, i, j) *= val;
    }
  }
}

void mat_add(Mat dest, Mat first, Mat second)
{
  assert(dest.rows == first.rows && first.rows == second.rows);
  assert(dest.cols == first.cols && first.cols == second.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      *mat_at(dest, i, j) = *mat_at(first, i, j) + *mat_at(second, i, j);
    }
  }
}

void mat_sub(Mat dest, Mat first, Mat second)
{
  assert(dest.rows == first.rows && first.rows == second.rows);
  assert(dest.cols == first.cols && first.cols == second.cols);

  for(size_t i = 0; i < dest.rows; i++) {
    for(size_t j = 0; j < dest.cols; j++) {
      *mat_at(dest, i, j) = *mat_at(first, i, j) - *mat_at(second, i, j);
    }
  }
}

void mat_apply_fn(Mat dst, Mat src, double (* function)(double))
{
  assert(dst.rows == src.rows && dst.cols == src.cols);

  for(size_t i = 0; i < dst.rows; i++) {
    for(size_t j = 0; j < dst.cols; j++) {
      *mat_at(dst, i, j) = function(*mat_at(src, i, j));
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
    .data = mat_at(origin, dim.beginRow, dim.beginCol)
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
