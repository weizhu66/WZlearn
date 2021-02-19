
#include "base_tree.h"

TreeNode::TreeNode(int feature_index, float threshold, float *value, TreeNode *true_branch, TreeNode *false_branch) {
    this->value = value;
    this->feature_index = feature_index;
    this->threshold = threshold;
    this->true_branch = true_branch;
    this->false_branch = false_branch;
}

TreeNode::~TreeNode() {
    if(this->value){
        delete value;
        value = nullptr;
    }
    if(this->true_branch){
        delete true_branch;
        true_branch = nullptr;
    }
    if(this->false_branch){
        delete false_branch;
        false_branch = nullptr;
    }
}

BaseTree::BaseTree(int min_sample_split, float min_impurity, int max_depth) {
    this->min_sample_split = min_sample_split;
    this->min_impurity = min_impurity;
    this->max_depth = max_depth;
    this->root = nullptr;
}

BaseTree::~BaseTree() {
    if(this->root!=nullptr){
        delete root;
        root = nullptr;
    }
}

void BaseTree::fit(MatrixType *X, MatrixType *y) {
    this->root = build_tree(*X,*y);
}

TreeNode* BaseTree::build_tree(MatrixType &X, MatrixType &y, int depth) {
    float largest_gain = 0;

    std::map<string,float> best_feature_threshold;
    std::map<string,MatrixType> best_branch;

    int m_samples = X.rows();
    int n_features = X.cols();
    MatrixType* X_y = new MatrixType(m_samples,n_features+1);
    *X_y << X, y;
    if(m_samples >= this->min_sample_split &&depth <= this->max_depth ){
        for (int feature_index = 0; feature_index < n_features ; ++feature_index) {
            MatrixType X_feature_index = X.col(feature_index);
            std::set<float> feature_set = MatrixUtils::unique<float>(X_feature_index);
            for(auto iter=feature_set.begin();iter!=feature_set.end();iter++){
                std::vector<MatrixType*> divided = this->divide(X_y,feature_index,*iter);
                if(divided[0]->rows()>0 && divided[1]->rows()>0){

                    MatrixType y1 = divided[0]->rightCols(1);
                    MatrixType y2 = divided[1]->rightCols(1);
                    float gain = information_gain(y,y1,y2);
                    if(gain > largest_gain){
                        largest_gain = gain;
                        best_feature_threshold.insert(std::pair<string,float>{FEATURE_KEY,feature_index});
                        best_feature_threshold.insert(std::pair<string,float>{THRESHOLD_KEY,*iter});
                        best_branch.insert(std::pair<string,MatrixType>{leftX,divided[0]->leftCols(n_features)});
                        best_branch.insert(std::pair<string,MatrixType>{leftY,divided[0]->rightCols(1)});
                        best_branch.insert(std::pair<string,MatrixType>{rightX,divided[1]->leftCols(n_features)});
                        best_branch.insert(std::pair<string,MatrixType>{rightY,divided[1]->rightCols(1)});
                    }
                }
            }
        }
    }
    delete X_y;
    X_y = nullptr;
    if(largest_gain > this->min_impurity){
        TreeNode *true_br = build_tree(best_branch[leftX],best_branch[leftY],depth+1);
        TreeNode *false_br = build_tree(best_branch[rightX],best_branch[rightY],depth+1);
        return new TreeNode(best_feature_threshold[FEATURE_KEY],best_feature_threshold[THRESHOLD_KEY],
                            nullptr,true_br,false_br);
    }
    float *value_p = vote_for_value(y);
    return new TreeNode(-1,-1,value_p, nullptr,nullptr);
}

MatrixType BaseTree::predict(MatrixType *X){
    int m_samples = X->rows();
    MatrixType y_pred(m_samples,1);
    for (int i = 0; i < m_samples; ++i) {
        MatrixType x = X->row(i);
        y_pred(i,0) = this->_predict(x, nullptr);
    }
    return y_pred;
}
