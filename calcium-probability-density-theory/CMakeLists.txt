cmake_minimum_required(VERSION 3.19)
project(calcium-probability-density-theory)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(calcium-probability-density-theory main.cpp enumerate.h)
target_link_libraries (calcium-probability-density-theory Eigen3::Eigen)