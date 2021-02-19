
#pragma once
#include "matrix_util.h"
#include <vector>
#include <algorithm> // std::move_backward
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock

#include "../data/data_loader.h"

namespace DataUtil{
    double accuracy_score(MatrixType const &y_pred,MatrixType const &y);
    double mean_squared_error(MatrixType const &y_pred,MatrixType const &y);

    std::vector<DataSet*> train_test_split(DataSet *dataSet,float train_percent, unsigned seed=0);
    DataSet* shuffle_data(DataSet *dataSet, unsigned seed);
    float cal_entropy(MatrixType &y);
    std::vector<MatrixType*> bootstrap_data(MatrixType &X_y,int bs);

    MatrixType* standardize(MatrixType const &X);
}