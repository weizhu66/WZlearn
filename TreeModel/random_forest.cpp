#include "random_forest.h"

RandomForest::RandomForest(int n_estimators, int min_sample_split, int max_depth) {
    this->n_estimators = n_estimators;
    this->min_sample_split = min_sample_split;
    this->max_depth = max_depth;
    this->tree_classifiers = std::vector<DecisionTreeClassifier*>();

    for (int i = 0; i < this->n_estimators ; ++i) {
        DecisionTreeClassifier* tree = new DecisionTreeClassifier(this->min_sample_split,1e-7,
                                        this->max_depth);
        tree_classifiers.push_back(tree);
    }
}
RandomForest::~RandomForest() {
    for(auto it = this->tree_classifiers.begin();it!=this->tree_classifiers.end();it++){
        delete *it;
    }
}
void RandomForest::fit(MatrixType *X, MatrixType *y) {
    MatrixType X_y = MatrixType(X->rows(),X->cols()+1);
    X_y << *X,*y;

    vector<MatrixType*> data_set = DataUtil::bootstrap_data(X_y,this->n_estimators);
    for (int i = 0; i < this->n_estimators; ++i) {
        MatrixType* sub_set = data_set[i];
        MatrixType* sub_X = new MatrixType(sub_set->leftCols(X->cols()));
        MatrixType* sub_y = new MatrixType(sub_set->rightCols(1));
        this->tree_classifiers[i]->fit(sub_X,sub_y);
        delete sub_set;
        delete sub_X;
        delete sub_y;
    }
}

MatrixType RandomForest::predict(MatrixType *X){
    MatrixType* all_preds = new MatrixType(X->rows(),this->n_estimators);
    MatrixType output_preds(X->rows(),1);
    for (int i = 0; i < this->n_estimators ; ++i) {
        DecisionTreeClassifier* clf = this->tree_classifiers[i];
        MatrixType pred = clf->predict(X);   //shape (m_samples,1)
        all_preds->col(i) << pred;
    }
    for (int i = 0; i < X->rows() ; ++i) {
        map<int,int> counter;
        for(int j = 0;j < this->n_estimators;j++){
            counter[(*all_preds)(i,j)]++;
        }
        vector<map<int,int>::const_iterator> v;
        for (auto it = counter.begin(); it !=counter.end() ;it ++) {
            v.emplace_back(it);
        }
        sort(v.begin(),v.end(),sort_desc);
        output_preds(i,0) = v[0]->first;
    }
    return output_preds;
}
bool RandomForest::sort_desc (map<int,int>::const_iterator a,map<int,int>::const_iterator b) {
    return (a->second > b->second); }

