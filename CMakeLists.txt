cmake_minimum_required(VERSION 3.5)

project(Graph_Partition)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(run main.cpp)
