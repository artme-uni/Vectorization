cmake_minimum_required(VERSION 3.15)
project(vectorization_3)

set(CMAKE_CXX_STANDARD 14)

set(USE_BLAS openblas)
set(BLAS_LIBRARY /usr/local/opt/openblas/lib)

add_executable(vectorization_3 main.cpp)

