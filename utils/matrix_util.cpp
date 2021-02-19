
#include "matrix_util.h"

MatrixType* MatrixUtils::add_first_col(const MatrixType &X) {
    int m_samples = X.rows();
    int n_features = X.cols();
    MatrixType* new_Xp = new MatrixType(n_features+1,m_samples);
    Eigen::VectorXf v(m_samples);
    v.setOnes();
    *new_Xp << v.transpose() ,X.transpose();
    (*new_Xp).transposeInPlace();
    return new_Xp;
}
template<typename T>
float sigmod(T t){
    return 1/(1+exp(-t));
}
/*
 *    sigmod:
 *          y = 1/(1+exp(-x))
 * */
MatrixType MatrixUtils::matrix_sigmod(const MatrixType &X) {
    MatrixType tmp = (-X).array().exp();
    tmp.array() += 1;
    tmp = tmp.array().pow(-1);
    return tmp;
}



MatrixType* MatrixUtils::where_y_equals_c(MatrixType &X,MatrixType &y,int c){
    assert(y.cols()==1);
    std::vector<int> v;
    for (int i = 0; i < y.rows() ; ++i) {
        if(c == y(i,0)){
            v.push_back(i);
        }
    }
    MatrixType* X_c = new MatrixType(v.size(),X.cols());
    for (int j = 0; j < v.size(); ++j) {
        X_c->row(j) << X.row(v[j]);
    }
    return X_c;
}

MatrixType* MatrixUtils::where_y_not_equals_c(MatrixType &X,MatrixType &y,int c){
    assert(y.cols()==1);
    std::vector<int> v;
    for (int i = 0; i < y.rows() ; ++i) {
        if(c != y(i,0)){
            v.push_back(i);
        }
    }
    MatrixType* X_c = new MatrixType(v.size(),X.cols());
    for (int j = 0; j < v.size(); ++j) {
        X_c->row(j) << X.row(v[j]);
    }
    return X_c;
}

MatrixType* MatrixUtils::where_y_bigger_c(MatrixType &X,MatrixType &y,float c){
    assert(y.cols()==1);
    std::vector<int> v;
    for (int i = 0; i < y.rows() ; ++i) {
        if(c <= y(i,0)){
            v.push_back(i);
        }
    }
    MatrixType* X_c = new MatrixType(v.size(),X.cols());
    for (int j = 0; j < v.size(); ++j) {
        X_c->row(j) << X.row(v[j]);
    }
    return X_c;
}

MatrixType* MatrixUtils::where_y_smaller_c(MatrixType &X,MatrixType &y,float c){
    assert(y.cols()==1);
    std::vector<int> v;
    for (int i = 0; i < y.rows() ; ++i) {
        if(c > y(i,0)){
            v.push_back(i);
        }
    }
    MatrixType* X_c = new MatrixType(v.size(),X.cols());
    for (int j = 0; j < v.size(); ++j) {
        X_c->row(j) << X.row(v[j]);
    }
    return X_c;
}
/*
 * 求每一列的方差
 * */
MatrixType MatrixUtils::col_var(MatrixType const &X){
    MatrixType _mean = X.colwise().mean();
    int cols = _mean.cols();
    int rows = X.rows();
    MatrixType vars(1,cols);
    for (int i = 0; i < cols ; ++i) {
        float mean = _mean(0,i);
        float v = (X.col(i).array() - mean).pow(2).sum()/rows;
        vars(0,i) = v;
    }
    return vars;
}
float MatrixUtils::calculate_val(MatrixType const &y){
    int rows = y.rows();
    float mean = y.array().mean();
    float var = (y.array() - mean).array().pow(2).sum()/rows;
    return var;
}

MatrixType MatrixUtils::boardcast_rows(MatrixType &X,int rows) {
    assert(X.rows()==1);
    MatrixType X_(rows,X.cols());
    for (int i = 1; i < rows; ++i) {
        X_.row(i) << X;
    }
    return X_;
}

MatrixType MatrixUtils::argmax_cols(MatrixType &X){
    MatrixType max_index(1,X.cols());
    for (int i = 0; i < X.cols() ; ++i) {
        int max_ = std::numeric_limits<int>::min();
        int index_ = 0;
        for (int j = 0; j < X.rows(); ++j) {
            if(X(j,i)>max_){
                max_ = X(j,i);
                index_ = j;
            }
        }
        max_index(0,i) = index_;
    }
    return max_index;
}

int MatrixUtils::label_equals_c_count(MatrixType &y,int c){
    int count = 0;
    for (int i = 0; i < y.rows() ; ++i) {
        if(y(i,0) == c){
            count ++;
        }
    }
    return count;
}

