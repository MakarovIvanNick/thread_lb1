cmake_minimum_required(VERSION 3.25)
project(threadSIMD)

set(CMAKE_CXX_STANDARD 17)

add_executable(threadSIMD main.cpp sse2neon.h) #task 1
#add_executable(threadSIMD task2.cpp sse2neon.h) #task 2
