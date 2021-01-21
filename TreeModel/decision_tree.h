
#pragma once
#include "../utils/matrix_util.h"
#include "data_util.h"

#define FEATURE_KEY "feature_index"
#define THRESHOLD_KEY "threshold"
#define leftX "leftX"
#define leftY "leftY"
#define rightX "rightX"
#define rightY "rightY"

class TreeNode{
public:
    int feature_index;
    float threshold;
    int* value ;  //leaf node
    TreeNode* true_branch;
    TreeNode* false_branch;

    TreeNode(int feature_index=NULL,float threshold=NULL,
            int *value = nullptr,TreeNode* true_branch = nullptr,
            TreeNode* false_branch= nullptr);
    ~TreeNode();
};
class DecisionTreeClassifier{
public:
    TreeNode *root;
    int min_sample_split;
    float min_impurity;
    int max_depth;

    DecisionTreeClassifier(int min_sample_split=2,
                           float min_impurity=1e-7,
                           int max_depth = std::numeric_limits<int>::max());
    ~DecisionTreeClassifier();

    void fit(MatrixType *X,MatrixType *y);
    TreeNode* build_tree(MatrixType &X,MatrixType &y,int depth = 0);
    std::vector<MatrixType*> divide(MatrixType* X_y,int feature_index,int threshold);

    float information_gain(MatrixType &y,MatrixType &y1,MatrixType &y2);

    int vote_for_value(MatrixType &y);

    MatrixType predict(MatrixType *X);

    int _predict(MatrixType &x,TreeNode *node = nullptr);
};

