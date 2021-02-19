
#include "decision_tree.h"

DecisionTreeClassifier::DecisionTreeClassifier(int min_sample_split, float min_impurity, int max_depth)
:BaseTree(min_sample_split,min_impurity,max_depth) {

}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    if(this->root!=nullptr){
        delete root;
        root = nullptr;
    }
}
float DecisionTreeClassifier::information_gain(MatrixType &y, MatrixType &y1, MatrixType &y2) {
    //    /*
//     * y.shape  (m_samples,1)
//     * */

    float prob = y1.rows()/y.rows();
    float entropy_ = DataUtil::cal_entropy(y);
    float information_gain_ = entropy_ - prob * DataUtil::cal_entropy(y1)
                                - (1-prob) * DataUtil::cal_entropy(y2);
    return information_gain_;
}
std::vector<MatrixType*> DecisionTreeClassifier::divide(MatrixType* X_y,int feature_index, float threshold){
    MatrixType tmp_ = X_y->col(feature_index);
    int thresholdi = (int) threshold;
    MatrixType *X1 = MatrixUtils::where_y_equals_c(*X_y,tmp_,thresholdi);
    MatrixType *X2 = MatrixUtils::where_y_not_equals_c(*X_y,tmp_,thresholdi);
    std::vector<MatrixType*> v;
    v.push_back(X1);
    v.push_back(X2);
    return v;
}
float* DecisionTreeClassifier::vote_for_value(MatrixType &y){
    float * most_common = new float;
    int max_count = 0;
    std::set<int> s = MatrixUtils::unique<int>(y);
    for(auto iter = s.begin();iter!= s.end();iter++){
        int count = MatrixUtils::label_equals_c_count(y,*iter);
        if(count > max_count){
            max_count = count;
            *most_common = *iter;
        }
    }
    return most_common;
}

float DecisionTreeClassifier::_predict(MatrixType &x,TreeNode *node){
    if(node== nullptr) node = this->root;
    if(node->value!= nullptr){
        return *node->value;
    }
    float feature_value = x(0,node->feature_index);
    TreeNode* branch = node->false_branch;
    if(feature_value == node->threshold){
        branch = node ->true_branch;
    }
    return _predict(x,branch);
}
