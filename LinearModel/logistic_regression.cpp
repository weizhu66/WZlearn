//
// Created by asus on 2021-01-19.
//

#include "logistic_regression.h"

LogisticRegression::LogisticRegression(int iterations,float learning_rate) {
    this->learning_rate = learning_rate;
    this->iterations = iterations;
}

void LogisticRegression::fit(MatrixType *X, MatrixType *y) {
    int m_samples = (*X).rows();
    int n_features = (*X).cols();
    LogisticRegression::init_weights(n_features);
    MatrixType *new_Xp = MatrixUtils::add_first_col(*X);
    MatrixType new_X_trans = (*new_Xp).transpose();
    (*y).resize(m_samples,1);
    for(int i =0;i<this->iterations;i++){
        MatrixType h_x = (*new_Xp) * (*w);
        MatrixType y_pred = MatrixUtils::matrix_sigmod(h_x);
        MatrixType w_grad = new_X_trans * (y_pred - *y);
        *this->w -= this->learning_rate * w_grad;
    }
}

MatrixType LogisticRegression::predict(MatrixType *X) {
    MatrixType* new_Xp = MatrixUtils::add_first_col(*X);
    MatrixType h_x = *new_Xp * (*this->w);
    delete new_Xp;
    new_Xp = nullptr;
    MatrixType y_pred = MatrixUtils::matrix_sigmod(h_x).array().round();
    return y_pred;
}

void LogisticRegression::init_weights(int n_features){
    float limit = sqrt(n_features);
    MatrixType tmp_w = Eigen::MatrixXf::Random(n_features,1);
    int b = 0;
    this->w = new MatrixType(n_features+1,1);
    *this->w << b,tmp_w*limit;
}

LogisticRegression::~LogisticRegression() {
    if(this->w!= nullptr){
        delete w;
        w = nullptr;
    }
}

