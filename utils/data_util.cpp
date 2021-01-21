
#include "data_util.h"

double DataUtil::accuracy_score(MatrixType const &y_pred, MatrixType const &y) {
   /**
    * y_pred,y's shape require  N * 1
    *
    * */
    MatrixType m = (y_pred.array()==y.array()).select(Eigen::MatrixXf::Ones(y.rows(),1),
                                       Eigen::MatrixXf::Zero(y.rows(),1));
    return m.array().sum()/y.rows();
}

vector<DataSet*> DataUtil::train_test_split(DataSet *data_set, float train_percent,
        unsigned seed) {
    int rows = (*data_set->X).rows();
    data_set = shuffle_data(data_set,seed);
    int train_rows = rows * train_percent;
    int test_rows = rows - train_rows;
    MatrixType train_X = (*data_set->X).topRows(train_rows);
    MatrixType train_y = (*data_set->y).topRows(train_rows);
    MatrixType test_X = (*data_set->X).bottomRows(test_rows);
    MatrixType test_y = (*data_set->y).bottomRows(test_rows);
    MatrixType* train_Xp = new MatrixType(train_X);
    MatrixType* train_yp = new MatrixType(train_y);
    MatrixType* test_Xp = new MatrixType(test_X);
    MatrixType* test_yp = new MatrixType(test_y);
    DataSet *train_set = new DataSet(train_Xp,train_yp);
    DataSet *test_set = new DataSet(test_Xp,test_yp);
    vector<DataSet*> v;
    v.push_back(train_set);
    v.push_back(test_set);
    return v;
}

DataSet* DataUtil::shuffle_data(DataSet *data_set, unsigned seed) {
    int rows = (*data_set->X).rows();
    vector<int> v;
    for (int i = 0; i < rows; ++i) {
        v.push_back(i);
    }
//    seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
    std::shuffle (v.begin (), v.end (), std::default_random_engine (seed));
    for (int j = 0; j < rows/2; ++j) {
        (*data_set->X).row(v[j]).swap((*data_set->X).row(v[rows-j-1]));
        (*data_set->y).row(v[j]).swap((*data_set->y).row(v[rows-j-1]));
    }
    return data_set;
}

float DataUtil::cal_entropy(MatrixType &y) {
    /*
     * y.shape (m_samples,1)
     * */
    set<int> s = MatrixUtils::unique(y);
    float entropy = 0;
    for(auto it = s.begin();it!=s.end();it++){
        int count = MatrixUtils::label_equals_c_count(y,*it);
        float p = count/y.rows();
        entropy += -p * log(p)/log(2);
    }
    return entropy;
}
