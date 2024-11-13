cmake_minimum_required(VERSION 3.23)
project(firing-rate-markov)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost 1.49.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(firing-rate-markov main.cpp)