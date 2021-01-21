//
// Created by asus on 2021-01-18.
//

#pragma once

#include "../utils/matrix_util.h"
#include <vector>
using namespace std;
#ifndef MatrixType
//#define MatrixType Eigen::MatrixXf;

#endif

class Regularization{
public:
    float alpha;
    virtual float loss(const MatrixType &w) = 0;
    virtual MatrixType grad(const MatrixType &w) = 0;
    Regularization();
    Regularization(float alpha);
};

class L1_regularization:public Regularization{
public:
    L1_regularization(float alpha);
    virtual float loss(const MatrixType &w);
    virtual MatrixType grad(const MatrixType &w);
};
class L2_regularization:public Regularization{
public:
    L2_regularization(float alpha);
    virtual float loss(const MatrixType &w);
    virtual MatrixType grad(const MatrixType &w);
};

class LinearRegression{
public:
    int n_iterations;
    float learning_rate;
    vector<float>* losses;
    MatrixType w;
    Regularization* regularization;

    LinearRegression(int n_iterations, float learning_rate,Regularization *regularization1= nullptr);
    ~LinearRegression();
    void fit(MatrixType *X,MatrixType *y);
    void init_weights(int n_features);
    MatrixType predict(MatrixType *X);
};

