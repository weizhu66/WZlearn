cmake_minimum_required(VERSION 3.14)
project(WZlearn)

set(CMAKE_CXX_STANDARD 14)

include_directories(LinearModel)
include_directories(utils)
include_directories(Bayes)
include_directories(data)
# TODO: Configure Eigen path properly
# include_directories(${EIGEN_INCLUDE_DIR})
# Uncomment and set EIGEN_INCLUDE_DIR to your Eigen installation path
add_executable(ML
        LinearModel/linear_regression.cpp
        LinearModel/linear_regression.h
        utils/matrix_util.cpp
        utils/matrix_util.h LinearModel/logistic_regression.cpp
        LinearModel/logistic_regression.h data/data_loader.h data/data_loader.cpp utils/data_util.h utils/data_util.cpp Bayes/naive_bayes.h Bayes/naive_bayes.cpp
        test/test.cpp TreeModel/decision_tree.h TreeModel/decision_tree.cpp TreeModel/random_forest.h TreeModel/random_forest.cpp TreeModel/decision_tree_regression.h TreeModel/decision_tree_regression.cpp TreeModel/decision_tree.cpp base/base_tree.h utils/data_util.cpp base/base_tree.cpp)
