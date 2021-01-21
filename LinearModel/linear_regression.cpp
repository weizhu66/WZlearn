//
// Created by asus on 2021-01-18.
//
#include "linear_regression.h"
#include <math.h>
#include <iostream>
LinearRegression::LinearRegression(int n_iterations, float learning_rate,
        Regularization* regularization) {
    this->n_iterations = n_iterations;
    this->learning_rate = learning_rate;
    this->losses = new vector<float>;
    this->regularization = regularization;
}

void LinearRegression::fit(MatrixType *X, MatrixType *y) {
    int m_samples = X->rows();
    int n_features = X->cols();
    LinearRegression::init_weights(n_features);
    MatrixType* new_Xp = MatrixUtils::add_first_col(*X);
    MatrixType new_X_trans = (*new_Xp).transpose();
    for(int i =0;i<this->n_iterations;i++){
        MatrixType y_pred = *(new_Xp) * (this->w);
        MatrixType y_bias = y_pred - *y;
        float loss;
        MatrixType tmp = 0.5 * y_bias.array()*y_bias.array();
        MatrixType w_grad = new_X_trans * y_bias;
        loss = tmp.mean();
        if(this->regularization!= nullptr){
            loss += this->regularization->loss(this->w);
            w_grad = w_grad + this->regularization->grad(this->w);
        }

        (*losses).push_back(loss);
//        cout << "iter:" << i <<"loss: " << loss << endl;
        this->w = (this->w) - this->learning_rate * w_grad;
    }
    delete new_Xp;
    new_Xp = nullptr;
}

void LinearRegression::init_weights(int n_features) {
    float limit = sqrt(n_features);
    MatrixType tmp_w = Eigen::MatrixXf::Random(n_features,1);
    int b = 0;
    this->w = MatrixType(n_features+1,1);
    this->w << b,tmp_w*limit;
}

MatrixType LinearRegression::predict(MatrixType *X){
    MatrixType* new_Xp = MatrixUtils::add_first_col(*X);
    MatrixType y_pred = *new_Xp * this->w;
    delete new_Xp;
    new_Xp = nullptr;
    return y_pred;
}

LinearRegression::~LinearRegression() {
    if(this->losses!=nullptr){
        delete losses;
        losses = nullptr;
    }
    if(this->regularization!= nullptr){
        delete regularization;
        regularization = nullptr;
    }
}

Regularization::Regularization() {

}
Regularization::Regularization(float alpha) {
        this->alpha = alpha;
}

float L1_regularization::loss(const MatrixType &w) {
    float loss = w.cwiseAbs().sum();
    return loss * this->alpha;
}

MatrixType L1_regularization::grad(const MatrixType &w) {
    return (w.array() < 0).select(-Eigen::MatrixXf::Ones(w.rows(),1),
            Eigen::MatrixXf::Ones(w.rows(),1)) * this->alpha;
}

L1_regularization::L1_regularization(float alpha):Regularization(alpha) {
//    this->alpha = alpha;
}


L2_regularization::L2_regularization(float alpha):Regularization(alpha) {

}

float L2_regularization::loss(const MatrixType &w) {
    float loss = (w.transpose()*w).sum();
    return 0.5 * this->alpha * loss;
}

MatrixType L2_regularization::grad(const MatrixType &w) {
    return w * this->alpha;
}
