cmake_minimum_required(VERSION 3.16)
project(qtda_stream CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(qtda_stream src/main.cpp)
target_include_directories(qtda_stream PRIVATE include)
target_compile_options(qtda_stream PRIVATE -O3 -Wall -Wextra)
