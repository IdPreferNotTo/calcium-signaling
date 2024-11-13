cmake_minimum_required(VERSION 3.22)
project(intervals-stationary-buffer-markov)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.49.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(intervals-stationary-buffer-markov main.cpp)
