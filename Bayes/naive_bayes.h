
#pragma once

#include "matrix_util.h"
#include <unordered_map>
#include <string>

using namespace std;
class NaiveBayes{
public:
    NaiveBayes();
    void fit(MatrixType *X,MatrixType *y);
    MatrixType _pdf(MatrixType *X,int n_class);
    MatrixType* get_prob(MatrixType *X);
    MatrixType predict(MatrixType *X);
    set<int> classes;
    unordered_map<int,float> prior_prob;
    unordered_map<int,unordered_map<string,MatrixType>> parameters;
};
