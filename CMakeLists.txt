cmake_minimum_required(VERSION 3.14)
project(WZlearn)

set(CMAKE_CXX_STANDARD 14)

include_directories(LinearModel)
include_directories(utils)
include_directories(Bayes)
include_directories(data)
include_directories(E:\\Download\\eigen-3.3.9)
add_executable(ML
        LinearModel/linear_regression.cpp
        LinearModel/linear_regression.h
        utils/matrix_util.cpp
        utils/matrix_util.h LinearModel/logistic_regression.cpp
        LinearModel/logistic_regression.h data/data_loader.h data/data_loader.cpp utils/data_util.h utils/data_util.cpp Bayes/naive_bayes.h Bayes/naive_bayes.cpp
        test/test.cpp TreeModel/decision_tree.h TreeModel/decision_tree.cpp TreeModel/random_forest.h TreeModel/random_forest.cpp TreeModel/decision_tree_regression.h TreeModel/decision_tree_regression.cpp base/base.h base/base.cpp TreeModel/base_tree.h TreeModel/base_tree.cpp)
