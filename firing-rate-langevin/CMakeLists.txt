cmake_minimum_required(VERSION 3.17)
project(firing-rate-langevin)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(firing-rate-langevin main.cpp)
target_link_libraries (firing-rate-langevin Eigen3::Eigen)