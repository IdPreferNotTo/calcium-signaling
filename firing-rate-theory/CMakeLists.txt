cmake_minimum_required(VERSION 3.19)
project(firing-rate_theory)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(firing-rate_theory main.cpp)
target_link_libraries (firing-rate_theory Eigen3::Eigen)