
#pragma once
#include "base_tree.h"
class DecisionTreeRegressor:public BaseTree{
public:
    DecisionTreeRegressor(int min_sample_split=2,
                          float min_impurity=1e-7,
                          int max_depth = std::numeric_limits<int>::max());
    ~DecisionTreeRegressor();
    std::vector<MatrixType*> divide(MatrixType* X_y,int feature_index, float threshold);

    float information_gain(MatrixType &y,MatrixType &y1,MatrixType &y2);

    float * vote_for_value(MatrixType &y);

    float _predict(MatrixType &x,TreeNode *node = nullptr);
};