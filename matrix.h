#ifndef NN_LIB
#define NN_LIB

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

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
} Mat;

typedef Mat SubMat;

typedef struct SubMatrixDimensions {
  size_t beginRow, beginCol, qtdRows, qtdCols;
} SubMatDim;

#define MAT_AT(m, i, j) (m).data[i*(m).row_size + j]
#define MAT_PRINT(m) mat_print((m), #m)

Mat mat_create(size_t rows, size_t cols);
void mat_print(Mat m, const char* variable_name);
void mat_fill(Mat m, double element);
void mat_copy(Mat dest, Mat src);
void mat_mul(Mat dest, Mat m1, Mat m2);
void mat_add(Mat dest, Mat another);
SubMat mat_get_submat(Mat origin, SubMatDim dim);
void mat_destruct(Mat *m);

#endif // NN_LIB
