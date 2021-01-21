//
// Created by asus on 2021-01-19.
//
#pragma once
#include "matrix_util.h"
#include <string>
#include <vector>
using namespace std;

vector<string> split(const string& str, const string& pattern);
class DataSet{
public:
    MatrixType *X;
    MatrixType *y;

    DataSet(MatrixType *X,MatrixType *y);
    ~DataSet();
};
class DataLoader{
public:
    DataLoader();
//    DataSet dataSet;
    DataSet* load_file(string const &file_path,string const &pattern);
};
