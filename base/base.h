
#pragma once

#include "../utils/matrix_util.h"

class BaseModel{
public:
    BaseModel();
    virtual void fit(MatrixType* X,MatrixType* y)=0;
    virtual MatrixType predict(MatrixType* X)=0;
    virtual ~BaseModel();
};
