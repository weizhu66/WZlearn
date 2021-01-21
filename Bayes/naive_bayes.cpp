
#include "naive_bayes.h"

#define MEAN "mean"
#define VAR "var"
NaiveBayes::NaiveBayes() {
    this->parameters = unordered_map<int,unordered_map<string,MatrixType>>();
    this->prior_prob = unordered_map<int,float>();
}

void NaiveBayes::fit(MatrixType *X, MatrixType *y) {
    string mean_string = MEAN;
    string var_string = VAR;

    this-> classes = MatrixUtils::unique(*y);
    for(auto iter = classes.begin();iter!=classes.end();iter++){
        MatrixType* X_c = MatrixUtils::where_y_equals_c(*X,*y,*iter);
        MatrixType X_c_mean = X_c->colwise().mean(); //shape (1,n_features)
        MatrixType X_c_var = MatrixUtils::col_var(*X_c);
        unordered_map<string,MatrixType> tmp_map;
        tmp_map.insert(pair<string,MatrixType>{mean_string,X_c_mean});
        tmp_map.insert(pair<string,MatrixType>{var_string,X_c_var});
        this->parameters.insert(pair<int,unordered_map<string,MatrixType>>{*iter,tmp_map} );
        this->prior_prob.insert(pair<int,float>{*iter,X_c->cols()-(*X).cols()});
        delete  X_c;
        X_c = nullptr;
    }
}

MatrixType NaiveBayes::_pdf(MatrixType *X, int n_class) {
    float eps = 0.0001;
    MatrixType mean = MatrixUtils::boardcast_rows(this->parameters[n_class][MEAN],
            X->rows()); //shape(m_samples,n_features)
    MatrixType var = MatrixUtils::boardcast_rows(this->parameters[n_class][VAR],
                                                  X->rows());
    MatrixType numerator = (-(((*X) - mean).array().pow(2)/(2*var.array() + eps))).array().exp();
    MatrixType denominator = (2*3.141592657*var.array() + eps).array().sqrt();
    MatrixType res = (numerator.array()/denominator.array()).array().log().rowwise().sum();
    //res shape(m_samples,1)
    return res.transpose();
}

MatrixType* NaiveBayes::get_prob(MatrixType *X){
    int n_classes = this->classes.size();
    MatrixType* output =new MatrixType(n_classes,X->rows());
    int i = 0;
    for (auto y = this->classes.begin() ; y!=this->classes.end(); ++y) {
        float prior = this->prior_prob[(*y)];
        MatrixType posterior = _pdf(X,*y);
        MatrixType pred = posterior.array() + prior;
        output->row(i++) << pred;
    }
    return output;
}

MatrixType NaiveBayes::predict(MatrixType *X) {
    MatrixType* output = get_prob(X); // output shape(n_classes,m_samples)
    MatrixType prediction = MatrixUtils::argmax_cols(*output);
    if(output!= nullptr){
        delete output;
        output = nullptr;
    }
    return prediction;
}

