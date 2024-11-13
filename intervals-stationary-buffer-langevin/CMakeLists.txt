cmake_minimum_required(VERSION 3.22)
project(intervals-stationary-buffer-langevin)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(intervals-stationary-buffer-langevin main.cpp)
target_link_libraries (intervals-stationary-buffer-langevin Eigen3::Eigen)