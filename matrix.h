#ifndef MAT_LIB
#define MAT_LIB

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* A Matrix is a contiguous array, one dimension, that can 
 * act like a matrix rows x cols, two dimensions.
 *
 * A Matrix can be a submatrix of another Matrix,
 * that can be done by the using of the row_size field, that always
 * remember the number of columns of the original Matrix and
 * then allows the submatrix to proper indexing it's positions.
 */
typedef struct Matrix {
    size_t rows, cols, row_size;
    double *data;
    int T; //for transpose matrixes, 1 if it was transposed
} Mat;

typedef Mat SubMat;

typedef struct SubMatrixDimensions {
  size_t beginRow, beginCol, qtdRows, qtdCols;
} SubMatDim;

#define MAT_AT(m, i, j) (*mat_at((m), (i), (j)))
#define LINE_AT(m, i) (*mat_at((m), 0, (i)))
#define MAT_PRINT(m) mat_print((m), #m)

Mat mat_create(size_t rows, size_t cols);
void mat_transpose(Mat *m);
double *mat_at(Mat m, size_t i, size_t j);
Mat mat_create_from(size_t rows, size_t cols, double elements[rows][cols]);
void mat_print(Mat m, const char* variable_name);
void mat_fill(Mat m, double element);
void mat_randomize(Mat m);
void mat_copy(Mat dest, Mat src);
void mat_mul(Mat dest, Mat m1, Mat m2);
void mat_const_mul(Mat dest, double val);
void mat_add(Mat dest, Mat first, Mat second);
void mat_sub(Mat dest, Mat first, Mat second);
void mat_apply_fn(Mat dst, Mat src, double (* function)(double));
SubMat mat_get_submat(Mat origin, SubMatDim dim);
SubMat mat_get_row(Mat origin, size_t row);
SubMat mat_get_col(Mat origin, size_t col);
void mat_destruct(Mat m);

#endif // MAT_LIB
