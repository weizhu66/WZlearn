
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

    template<typename T>
    std::set<T> unique(MatrixType &y);

    MatrixType* where_y_equals_c(MatrixType &X,MatrixType &y,int c);
    MatrixType* where_y_not_equals_c(MatrixType &X,MatrixType &y,int c);
    MatrixType* where_y_bigger_c(MatrixType &X,MatrixType &y, float c);
    MatrixType* where_y_smaller_c(MatrixType &X,MatrixType &y,float c);
    MatrixType col_var(MatrixType const &X);
    float calculate_val(MatrixType const &y);
    MatrixType boardcast_rows(MatrixType &X,int rows);
    MatrixType argmax_cols(MatrixType &X);
    int label_equals_c_count(MatrixType &y,int c);
}

template<typename T>
std::set<T> MatrixUtils::unique(MatrixType &y) {
    assert(y.cols()==1);
    std::set<T> set_;
    int rows = y.rows();
    for (int i = 0; i < rows ; ++i) {
        set_.insert(y(i,0));
    }
    return set_;
}
