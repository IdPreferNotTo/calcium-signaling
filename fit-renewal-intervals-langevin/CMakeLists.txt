cmake_minimum_required(VERSION 3.23)
project(fit-renewal-intervals-langevin)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(fit-renewal-intervals-langevin main.cpp)
target_link_libraries (fit-renewal-intervals-langevin Eigen3::Eigen)