//
// Created by asus on 2021-01-19.
//
#include "data_loader.h"
#include <fstream>
#include <iostream>
using namespace std;

vector<string> split(const string& str, const string& pattern){
    vector<string> ret;
    if(pattern.empty()) return ret;
    size_t start=0,index=str.find_first_of(pattern,0);
    while(index!=str.npos)
    {
        if(start!=index)
            ret.push_back(str.substr(start,index-start));
        start=index+1;
        index=str.find_first_of(pattern,start);
    }
    if(!str.substr(start).empty())
        ret.push_back(str.substr(start));
    return ret;
}

DataSet::DataSet(MatrixType *X, MatrixType *y) {
    this->X = X;
    this->y = y;
}
DataSet::~DataSet() {
    if(this->X!= nullptr){
        delete X;
        this->X = nullptr;
    }
    if(this->y!= nullptr){
        delete y;
        this->y = nullptr;
    }
}

DataLoader::DataLoader() {

}
DataSet* DataLoader::load_file(std::string const &file_path,string const &pattern) {
    ifstream infile;
    infile.open(file_path.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    string s;
    int rows = 0;
    int cols = 0;
    vector<vector<float>> v;
    while(getline(infile,s))
    {
        vector<string> splited = split(s,pattern);
        vector<float> tmp_vector;
        cols = splited.size();
        rows++;
        for(auto ss = splited.begin();ss!= splited.end();ss++){
            tmp_vector.push_back(atof((*ss).c_str()));
        }
        v.push_back(tmp_vector);
    }

    MatrixType *X = new MatrixType(rows,cols-1);
    MatrixType *y = new MatrixType(rows,1);
    for(int i = 0;i<rows;i++){
        for(int j = 0;j<cols;j++){
            if(j!=cols-1){
                (*X)(i,j) = v[i][j];
            } else{
                (*y)(i,0) = v[i][j];
            };
        }
    }
    infile.close();
    DataSet *dataSet = new DataSet(X,y);
    return dataSet;
}


