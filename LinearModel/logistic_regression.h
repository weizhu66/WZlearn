//
// Created by asus on 2021-01-19.
//

#pragma once

#include "../utils/matrix_util.h"

class LogisticRegression{
public:
    float learning_rate;
    int iterations;
    MatrixType w;


    LogisticRegression(int iterations,float learning_rate);
    ~LogisticRegression();

    void fit(MatrixType *X,MatrixType *y);
    MatrixType predict(MatrixType *X);
    void init_weights(int n_features);

};
