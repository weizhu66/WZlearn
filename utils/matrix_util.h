
#pragma once
#include<Eigen/Dense>
#include "math.h"
#include <set>
#include <map>
#include <vector>

typedef Eigen::MatrixXf MatrixType;

namespace MatrixUtils{
    MatrixType* add_first_col(const MatrixType &X);
    MatrixType matrix_sigmod(const MatrixType &X);
    std::set<int> unique(MatrixType &y);
    MatrixType* where_y_equals_c(MatrixType &X,MatrixType &y,int c);
    MatrixType* where_y_not_equals_c(MatrixType &X,MatrixType &y,int c);
    MatrixType col_var(MatrixType const &X);
    MatrixType boardcast_rows(MatrixType &X,int rows);
    MatrixType argmax_cols(MatrixType &X);
    int label_equals_c_count(MatrixType &y,int c);
}
