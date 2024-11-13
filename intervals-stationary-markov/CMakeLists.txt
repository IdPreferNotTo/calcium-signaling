cmake_minimum_required(VERSION 3.10)
project(intervals-stationary-markov)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost 1.49.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(intervals-stationary-markov main.cpp)