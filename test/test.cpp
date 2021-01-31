//
// Created by asus on 2021-01-20.
//

#include "../data/data_loader.h"
#include "../utils/data_util.h"
#include "../Bayes/naive_bayes.h"
#include "../TreeModel/decision_tree.h"
#include "../LinearModel/logistic_regression.h"
#include "../TreeModel/random_forest.h"
#include "../LinearModel/linear_regression.h"
#include <iostream>
using namespace std;

void test_bayes(){
    DataLoader loader;
    DataSet* dataSet = loader.load_file("../data/iris.csv",",");
    NaiveBayes model;
    vector<DataSet*> v = DataUtil::train_test_split(dataSet,0.75,33);
    model.fit(v[0]->X,v[0]->y);
    MatrixType p = model.predict(v[1]->X);
    float acc = DataUtil::accuracy_score(p,*v[1]->y);
    cout << acc;
}
void test_logistic_regression(){
    DataLoader loader;
    DataSet* dataSet =loader.load_file("../data/testSet.txt","\t");
    vector<DataSet*> d = DataUtil::train_test_split(dataSet,0.8,9);
    DataSet* train_set = d[0];
    DataSet* test_set = d[1];
    LogisticRegression model(30000,0.0005);
    model.fit(train_set->X,train_set->y);
    MatrixType y_pred = model.predict(test_set->X);

    std::cout << y_pred << endl;
    double acc = DataUtil::accuracy_score(y_pred,*(test_set->y));
    std::cout << "acc: " << acc << endl;
}
void test_linear_regression(){
    MatrixType *X = new MatrixType(3,2);
    MatrixType *y = new MatrixType(3,1);
    *X << -10,-10,2,2,6,8;
    *y << -9,3,8;
    LinearRegression model(30000,0.0005);
    model.fit(X,y);
    MatrixType y_pred = model.predict(X);
    cout << y_pred << endl;
    cout << "coef: " << *model._coef << endl;
    cout << "b: " << *model._b << endl;
}
void test_decision_tree(){

}
void test_random_forest(){
    MatrixType *X = new MatrixType(10,3);
    MatrixType *y = new MatrixType(10,1);
    *X << 1,2,1,
            2,3,1,
            1,2,2,
            3,1,2,
            1,2,1,
            2,3,1,
            1,2,2,
            3,3,1,
            1,2,1,
            3,2,1;
    *y << 1,1,2,1,0,1,0,1,2,0;
    RandomForest model(10,2,3);
    MatrixType *x = new MatrixType(4,3);
    *x << 1,2,1,
            2,3,1,
            1,2,2,
            3,1,2;
    model.fit(X,y);
    MatrixType y_pred = model.predict(x);
    cout << y_pred.transpose() << endl;
    delete X;
    delete y;
}
int main(){
    test_bayes();
//    test_decision_tree();
//    test_logistic_regression();
//    test_random_forest();
//    test_linear_regression();

    return 0;
}
