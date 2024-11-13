cmake_minimum_required(VERSION 3.17)
project(fit-nonrenewal-intervals-langevin)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ALGLIB REQUIRED)

add_executable(fit-nonrenewal-intervals-langevin main.cpp)
target_link_libraries (fit-nonrenewal-intervals-langevin PRIVATE ALGLIB Eigen3::Eigen)