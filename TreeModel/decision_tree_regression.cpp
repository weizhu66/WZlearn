
#include "decision_tree_regression.h"
DecisionTreeRegressor::DecisionTreeRegressor(int min_sample_split, float min_impurity, int max_depth)
        :BaseTree(min_sample_split,min_impurity,max_depth){

}
DecisionTreeRegressor::~DecisionTreeRegressor() {
    if(this->root!=nullptr){
        delete root;
        root = nullptr;
    }
}
float DecisionTreeRegressor::information_gain(MatrixType &y, MatrixType &y1, MatrixType &y2) {
    float var = MatrixUtils::calculate_val(y);
    float var_1 = MatrixUtils::calculate_val(y1);
    float var_2 = MatrixUtils::calculate_val(y2);
    float l1 = (float)y1.rows()/y.rows();
    float l2 = (float)y2.rows()/y.rows();
    return var - (l1 * var_1 + l2 * var_2);
}

float* DecisionTreeRegressor::vote_for_value(MatrixType &y) {
    float* value = new float(y.array().mean());
    return value;
}
std::vector<MatrixType*> DecisionTreeRegressor::divide(MatrixType* X_y,int feature_index,float threshold){
    MatrixType tmp_ = X_y->col(feature_index);
    MatrixType *X1 = MatrixUtils::where_y_bigger_c(*X_y,tmp_,threshold);
    MatrixType *X2 = MatrixUtils::where_y_smaller_c(*X_y,tmp_,threshold);
    std::vector<MatrixType*> v;
    v.push_back(X1);
    v.push_back(X2);
    return v;
}

float DecisionTreeRegressor::_predict(MatrixType &x,TreeNode *node){
    if(node== nullptr) node = this->root;
    if(node->value!= nullptr){
        return *node->value;
    }
    float feature_value = x(0,node->feature_index);
    TreeNode* branch = node->false_branch;
    if(feature_value >= node->threshold){
        branch = node ->true_branch;
    }
    return _predict(x,branch);
}