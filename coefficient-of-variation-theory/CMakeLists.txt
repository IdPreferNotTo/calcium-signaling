cmake_minimum_required(VERSION 3.19)
project(coefficient-of-variation-theory)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(coefficient-of-variation-theory main.cpp)
target_link_libraries (coefficient-of-variation-theory Eigen3::Eigen)