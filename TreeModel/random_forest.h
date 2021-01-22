
#pragma once

#include "../utils/matrix_util.h"
#include "../TreeModel/decision_tree.h"
#include <algorithm>
class RandomForest{
public:
    int n_estimators;
    int min_sample_split;
    int max_depth;
    bool use_bootstrap = false;
    bool use_random_feature = false;
    std::vector<DecisionTreeClassifier*> tree_classifiers;

    RandomForest(int n_estimators = 200,int min_sample_split = 2,
                 int max_depth=std::numeric_limits<int>::max());
    void fit(MatrixType *X,MatrixType *y);

    MatrixType predict(MatrixType *X);

private:
    static bool sort_desc (map<int,int>::const_iterator a,map<int,int>::const_iterator b);
};