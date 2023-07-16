#ifndef NN_LIB
#define NN_LIB

#include "matrix.h"

double sigmoid(double x);
double reLU(double x);
double cost(Mat weights, Mat bias, Mat train_data);

#endif //NN_LIB
