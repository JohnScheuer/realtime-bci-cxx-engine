cmake_minimum_required(VERSION 3.22)
project(realtime_bci_cxx_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(realtime_bci_engine
    src/main.cpp
    src/cuda_bci_decoder.cu
)

set_target_properties(realtime_bci_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
