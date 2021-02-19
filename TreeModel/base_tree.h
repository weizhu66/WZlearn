
#pragma once
#include "../utils/matrix_util.h"
#include "../utils/data_util.h"
#include "../base/base.h"
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
    float * value ;  //leaf node
    TreeNode* true_branch;
    TreeNode* false_branch;

    TreeNode(int feature_index=-1,float threshold=-1,
             float *value = nullptr,TreeNode* true_branch = nullptr,
             TreeNode* false_branch= nullptr);
    ~TreeNode();
};

class BaseTree:public BaseModel {
public:
    TreeNode *root;
    int min_sample_split;
    float min_impurity;
    int max_depth;
    BaseTree(int min_sample_split=2,
                           float min_impurity=1e-7,
                           int max_depth = std::numeric_limits<int>::max());
    virtual ~BaseTree();

    virtual void fit(MatrixType *X,MatrixType *y);
    virtual MatrixType predict(MatrixType *X);

    TreeNode* build_tree(MatrixType &X,MatrixType &y,int depth = 0);
    virtual float _predict(MatrixType &x,TreeNode *node = nullptr)=0;

    virtual std::vector<MatrixType*> divide(MatrixType* X_y,int feature_index, float threshold)=0;

    virtual float information_gain(MatrixType &y,MatrixType &y1,MatrixType &y2)=0;

    virtual float * vote_for_value(MatrixType &y)=0;
};

